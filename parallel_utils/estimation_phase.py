import numpy as np
import time
import scipy.sparse as sp
from dask.distributed import get_worker

from constraints.sparse_constraint import SparseConstraint

from typing import List, Dict, Any, Optional


def _combine_child_constraints(num_children: int, contingency_vector: sp.csc_matrix, constraints: List, active_set: set, n_cells: Optional[int] = None) -> List[SparseConstraint]:
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
        n_cells (Optional[int]): Number of contingency cells. Defaults to the worker-global
            _data_handler.n_cells; callers in the main process (no worker globals) must pass it.

    Returns:
        List[SparseConstraint]: List of SparseConstraints for the joint optimization problem.
    '''
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
                                 children_ids: List[int], children_level: int, is_leaf: bool = False) -> None:
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
    worker = get_worker()
    contingency_vector = worker.data_handler.load_vector(node_path)

    # Materialize and combine children vectors and constraints
    children_vectors = []
    children_constraints = []

    for filter_dict, child_id in zip(children_filter_dicts, children_ids):
        child_vector, child_constraint = worker.data_handler.materialize_node_data(filter_dict, worker.constraints[children_level], worker.Q)
        # Try to use pre-computed noise, fallback to in-situ generation if not available
        try:
            worker.privacy_mechanism.add_noise_from_precomputed(worker.noisy_arr, child_vector, child_id)
        except:
            worker.privacy_mechanism.add_noise(child_vector, children_level, worker.query_sensitivity)
        children_vectors.append(child_vector)
        children_constraints.append(child_constraint)

    # Constraints are adapted to the new vector size.
    # Also create others to ensure consistency in the number of rows per category in the parent.
    # The number of rows in the parent category must match the sum of rows of that category across all children.
    n_cells = worker.data_handler.n_cells
    num_children = len(children_filter_dicts)
    n_joint = num_children * n_cells

    # Cells where the parent is non-zero. By non-negativity + consistency, children can only
    # be non-zero on these cells. Expand the support to joint-space indices {k*n_cells + j}
    # so the optimizers instantiate variables only there. `active` stays an ordered list: the
    # optimizer aligns its solution positionally to it across the real -> rounding solves.
    support = contingency_vector.indices
    active = [k * n_cells + int(j) for k in range(num_children) for j in support]

    # Combine receives the active set so it can bake prune-to-0 + reindexing into the constraints.
    joint_constraints = _combine_child_constraints(num_children, contingency_vector,
                                                   children_constraints, set(active), worker.data_handler.n_cells)

    t1 = time.time()
    x_tilde = worker.optimizer.non_negative_real_estimation(
        noisy_measurements=children_vectors,
        node_id=node_id,
        constraints=joint_constraints,
        query_matrix=worker.Q,
        active=active
    )
    real_time = time.time() - t1

    t1 = time.time()
    joint_solution = worker.optimizer.rounding_estimation(
        x_tilde=x_tilde,
        node_id=node_id,
        constraints=joint_constraints,
        active=active,
        n=n_joint
    )
    rounding_time = time.time() - t1

    if worker.check_correctness: _check_node_correctness(contingency_vector, joint_solution)

    if not is_leaf: worker.data_handler.update_child_vectors(joint_solution, children_filter_dicts)
    else:
        child_vectors = []
        start = 0

        for _ in children_filter_dicts:
            end = start + n_cells
            updated_vector = joint_solution[start:end]
            child_vectors.append(updated_vector)
            start = end

        worker.data_handler.write_microdata(node_id, child_vectors, children_filter_dicts)

    print(f'  [Node {node_id}] - real {real_time:.1f}s - rounding {rounding_time:.1f}s')

