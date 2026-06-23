import numpy as np
import time
import zarr
import scipy.sparse as sp
from scipy.sparse import spmatrix

from optimizer import OptimizationModel
from data_handler import DataHandler
from domain import ContingencyDomain
from privacy import PrivacyMechanism
from constraints.sparse_constraint import SparseConstraint

from typing import List, Dict, Any, Tuple

def init_process(optimizer: Tuple[type, str, Dict], constraints_dict: Dict[int, List],
                 spill_dir: str, microdata_dir: str, parquet_path: str,
                 domain_dict: Dict[str, Any], hierarchical_columns: List[str], query_columns: List[str],
                 privacy_mechanism: PrivacyMechanism, query_matrix: spmatrix, query_sensitivity: int, check: bool,
                 zarr_path: str, noisy_array_name: str) -> None:
    '''Initialize global variables for parallel worker processes.

    Args:
        optimizer (Tuple[type, str, Dict]): Params to pass to the solver (result dtype, temporary files directory
                                            and solver options dict).
        constraints_dict (Dict[int, List]): Constraints mapped by level.
        spill_dir (str): Directory path for spilling vectors to disk.
        microdata_dir (str): Directory path for temporary microdata files.
        parquet_path (str): Path to the parquet file.
        domain_dict (Dict[str, Any]): Domain mapping for query columns.
        hierarchical_columns (List[str]): Hierarchical column names.
        query_columns (List[str]): Query column names.
        privacy_mechanism (PrivacyMechanism): Privacy mechanism instance for noise addition.
        query_matrix (spmatrix): The sparse query matrix Q used in optimization.
        query_sensitivity (int): Query sensitivity for noise addition.
        check (bool): Whether to check node correctness.
        zarr_path (str): Path to the Zarr group holding pre-computed noise vectors.
        noisy_array_name (str): Name of the noise array within the Zarr group.
    '''
    global _optimizer, _data_handler, _Q, _check, _privacy_mechanism, _query_sensitivity, _constraints, _noisy_arr

    _optimizer = OptimizationModel(*optimizer)

    _data_handler = DataHandler()
    _data_handler.spill_dir = spill_dir
    _data_handler.microdata_dir = microdata_dir
    _data_handler.hierarchical_columns = hierarchical_columns
    _data_handler.query_columns = query_columns
    _data_handler.file_path = parquet_path

    _data_handler.contingency_domain = ContingencyDomain(columns=query_columns, domains=domain_dict)
    _data_handler.n_cells = _data_handler.contingency_domain.n_cells

    _data_handler.create_data_view()

    _constraints = constraints_dict
    _Q = query_matrix
    _query_sensitivity = query_sensitivity
    _privacy_mechanism = privacy_mechanism
    _noisy_arr = zarr.open_group(zarr_path, mode="r")[noisy_array_name]
    _check = check

def _combine_child_constraints(num_children: int, contingency_vector: sp.csc_matrix, constraints: List, active_set: set) -> List[SparseConstraint]:
    '''Combine child publication constraints into joint SparseConstraints.

    Creates consistency constraints that ensure each parent cell equals the sum of corresponding child cells.

    The parent vector is a sparse CSC column vector, so consistency constraints are emitted only
    for its non-zero cells (its support). Cells where the parent is 0 need no constraint: the
    optimizer does not create child variables there, so they are structurally 0 and the
    consistency sum(children) == 0 holds automatically.

    This function translates child-local SparseConstraints (indexed 0..n_cells-1) to joint-space
    SparseConstraints (indexed over the global indices), filtering to only active cells
    (pruned cells are dropped from the coefficients).

    Args:
        num_children (int): Number of child nodes.
        contingency_vector (sp.csc_matrix): The parent's sparse cell-count vector, shape (n_cells, 1).
        constraints (List): List of Constraint objects (one list per child). Each constraint's
            to_sparse_constraint() method will be called to get SparseConstraint representations.
        active_set (set): Active joint-space global indices {k*n_cells + j} (parent support expanded
            over children). Cells outside it are pruned and dropped from constraints.

    Returns:
        List[SparseConstraint]: List of SparseConstraints for the joint optimization problem.
    '''
    n_cells = _data_handler.n_cells
    joint_constraints = []

    # Per-child constraints: convert each constraint to SparseConstraint via to_sparse_constraint()
    # Offset them to joint space (base + j) and filter to only active cells.
    start = 0
    for child_constraints in constraints:
        base = start

        for sparse_constraint in child_constraints:
            new_sparse_constraint = sparse_constraint.prune_to_active_space(base, active_set)
            if new_sparse_constraint is not None: joint_constraints.append(new_sparse_constraint)
        start += n_cells

    # Consistency constraints: parent value at each non-zero cell = sum of child values at that cell.
    # Authored directly in joint space with all indices active (parent support ensures this).
    for index, value in zip(contingency_vector.indices, contingency_vector.data):
        index = int(index)
        indices_to_sum = np.array([index + i * n_cells for i in range(num_children)])
        coefs = np.ones(len(indices_to_sum))

        joint_constraints.append(
            SparseConstraint(
                indices=indices_to_sum,
                coefs=coefs,
                sense="=",
                rhs=float(value)
            )
        )

    return joint_constraints

def _check_node_correctness(parent_vector: sp.csc_matrix, children_vectors: sp.csc_matrix) -> None:
    '''Checks that the sum of the values in the parent node vector
    is equal to the sum of the values in its children vectors.

    Args:
        parent_vector (sp.csc_matrix): Contingency vector of the parent node.
        children_vectors (sp.csc_matrix): Concatenated contingency vectors of the child nodes.
    '''
    parent_sum = parent_vector.data.sum()
    children_sum = children_vectors.data.sum()

    if parent_sum != children_sum:
        print(f"\nError: The sum of the children nodes' contingency vectors "
              f"({children_sum}) does not equal the parent node's contingency vector ({parent_sum}).")

def estimate_and_update_children(node_id: int, node_path: str, children_filter_dicts: List[Dict[str, Any]],
                                 children_ids: List[int], children_level: int, is_leaf: bool = False) -> float:
    '''Solve optimization for a node considering its children and update their vectors.

    Args:
        node_id (int): The unique ID of the parent node.
        node_path (str): Path to contigency vector file.
        children_filter_dicts (List[Dict[str, Any]]): List of filter dictionaries for each child.
        children_ids (List[int]): List of node IDs for each child (for pre-computed noise lookup).
        children_level (int): Level of all children (they all share the same level).
        is_leaf (bool): Whether children are leaf nodes. Defaults to False.

    Returns:
        float: Time spent writing microdata files
    '''
    contingency_vector = _data_handler.load_vector(node_path)

    # Materialize and combine children vectors and constraints
    children_vectors = []
    children_constraints = []

    for filter_dict, child_id in zip(children_filter_dicts, children_ids):
        child_vector, child_constraint = _data_handler.materialize_node_data(filter_dict, _constraints[children_level], _Q)

        # Try to use pre-computed noise, fallback to in-situ generation if not available
        try:
            _privacy_mechanism.add_noise_from_precomputed(_noisy_arr, child_vector, child_id)
        except:
            _privacy_mechanism.add_noise(child_vector, children_level, _query_sensitivity)

        children_vectors.append(child_vector)
        children_constraints.append(child_constraint)

    # Concatenate children vectors, and constraints are adapted to the new vector size.
    # Also create others to ensure consistency in the number of rows per category in the parent.
    # The number of rows in the parent category must match the sum of rows of that category across all children.
    joint_contingency_vector = np.concatenate(children_vectors)
    n_cells = _data_handler.n_cells
    num_children = len(children_filter_dicts)
    n_joint = num_children * n_cells

    # Cells where the parent is non-zero. By non-negativity + consistency, children can only
    # be non-zero on these cells. Expand the support to joint-space indices {k*n_cells + j}
    # so the optimizers instantiate variables only there. `active` stays an ordered list: the
    # optimizer aligns its solution positionally to it across the real -> rounding solves.
    support = contingency_vector.indices
    active = [k * n_cells + int(j) for k in range(num_children) for j in support]

    # Combine receives the active set so it can bake prune-to-0 + reindexing into the constraints.
    joint_constraints = _combine_child_constraints(num_children, contingency_vector, children_constraints, set(active))

    t1 = time.time()
    x_tilde = _optimizer.non_negative_real_estimation(
        noisy_measurements=joint_contingency_vector,
        node_id=node_id,
        constraints=joint_constraints,
        query_matrix=_Q,
        active=active
    )
    real_time = time.time() - t1

    t1 = time.time()
    joint_solution = _optimizer.rounding_estimation(
        x_tilde=x_tilde,
        node_id=node_id,
        constraints=joint_constraints,
        active=active,
        n=n_joint
    )
    rounding_time = time.time() - t1

    if _check: _check_node_correctness(contingency_vector, joint_solution)

    joint_contingency_vector = None
    joint_constraints = None

    microdata_time = 0.0
    if not is_leaf: _data_handler.update_child_vectors(joint_solution, children_filter_dicts)
    else:
        t_microdata = time.time()
        child_vectors = []
        start = 0

        for _ in children_filter_dicts:
            end = start + n_cells
            updated_vector = joint_solution[start:end]
            child_vectors.append(updated_vector)
            start = end

        _data_handler.write_microdata(node_id, child_vectors, children_filter_dicts)
        microdata_time = time.time() - t_microdata

    print(f'  [Node {node_id}] - real {real_time:.1f}s - rounding {rounding_time:.1f}s')

    return microdata_time
