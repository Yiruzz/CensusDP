import numpy as np
import time
import zarr
from scipy.sparse import spmatrix

from optimizer import OptimizationModel
from data_handler import DataHandler
from domain import ContingencyDomain
from privacy import PrivacyMechanism

from typing import List, Callable, Dict, Any

def init_process(solver_options: dict, constraints_dict: Dict[int, List],
                 spill_dir: str, microdata_dir: str, parquet_path: str,
                 domain_dict: Dict[str, Any], hierarchical_columns: List[str], query_columns: List[str],
                 privacy_mechanism: PrivacyMechanism, query_matrix: spmatrix, query_sensitivity: int,
                 check: bool, zarr_path: str, noisy_array_name: str) -> None:
    '''Initialize global variables for parallel worker processes.

    Args:
        solver_options (dict): Dictionary of options to pass to the optimization solver.
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
    '''
    global _optimizer, _data_handler, _Q, _check, _privacy_mechanism, _query_sensitivity, _constraints, _noisy_arr 

    _optimizer = OptimizationModel(solver_options=solver_options)

    _data_handler = DataHandler()
    _data_handler.spill_dir = spill_dir
    _data_handler.microdata_dir = microdata_dir
    _data_handler.hierarchical_columns = hierarchical_columns
    _data_handler.query_columns = query_columns
    _data_handler.file_path = parquet_path

    _data_handler.contingency_domain = ContingencyDomain(columns=query_columns, domains=domain_dict)
    _data_handler.contingency_df_length = _data_handler.contingency_domain.n_cells

    _data_handler.create_data_view()

    _constraints = constraints_dict
    _Q = query_matrix
    _query_sensitivity = query_sensitivity
    _privacy_mechanism = privacy_mechanism
    _check = check

    _noisy_arr = zarr.open_group(zarr_path, mode="r")[noisy_array_name]

def _combine_child_constraints(num_children: int, contingency_vector: np.ndarray, constraints: List) -> List[Callable]:
    '''Combine child publication constraints into joint constraints.

    Creates consistency constraints that ensure each parent cell equals the sum of corresponding child cells.

    Args:
        num_children (int): Number of child nodes.
        contingency_vector (np.ndarray): The parent's contingency vector.
        constraints (List): List of child constraints.

    Returns:
        List: List of constraint functions for the joint optimization problem.
    '''   

    joint_constraints = []

    # Wrap child publication constraints with adjusted indices
    start = 0     
    for child_constraints in constraints:
        end = start + _data_handler.contingency_df_length
        for constraint in child_constraints:
            joint_constraints.append(
                lambda joint_array, s=start, e=end, c=constraint:
                    c({i - s: joint_array[i] for i in range(s, e)})
            )
        start = end

    # Add consistency constraints: parent value at each index = sum of child values at that index
    for index in range(_data_handler.contingency_df_length):
        indices_to_sum = [index + i * _data_handler.contingency_df_length for i in range(num_children)]
        joint_constraints.append(
            lambda joint_array, idxs=indices_to_sum, value=contingency_vector[index]:
                sum(joint_array[j] for j in idxs) == value
        )

    return joint_constraints 

def _check_node_correctness(parent_vector: np.ndarray, children_vectors: np.ndarray) -> None:
    '''Checks that the sum of the values in the parent node vector 
    is equal to the sum of the values in its children vectors.

    Args:
        parent_vector (np.ndarray): Contingency vector of the parent node.
        children_vectors (np.ndarray): Contingency vectors of the child nodes.
    '''
    parent_sum = np.sum(parent_vector)
    children_sum = np.sum(children_vectors)
    
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
        children_ids (List[int]): List of node IDs for each child (for noise lookup).
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
    joint_constraints = _combine_child_constraints(len(children_filter_dicts), contingency_vector, children_constraints)

    t1 = time.time()
    x_tilde = _optimizer.non_negative_real_estimation(
        noisy_measurements=joint_contingency_vector,
        node_id=node_id,
        constraints=joint_constraints,
        query_matrix=_Q
    )
    real_time = time.time() - t1

    t1 = time.time()
    joint_solution = _optimizer.rounding_estimation(
        x_tilde=x_tilde,
        node_id=node_id,
        constraints=joint_constraints
    )
    rounding_time = time.time() - t1

    if _check: _check_node_correctness(contingency_vector, joint_solution)

    joint_contingency_vector = None
    joint_constraints = None

    microdata_time = 0.0
    if not is_leaf: _data_handler.update_child_vectors(joint_solution, _data_handler.contingency_df_length, children_filter_dicts)
    else:
        t_microdata = time.time()
        child_vectors = []
        start = 0

        for _ in children_filter_dicts:
            end = start + _data_handler.contingency_df_length
            updated_vector = joint_solution[start:end]
            child_vectors.append(updated_vector)
            start = end

        _data_handler.write_microdata(node_id, child_vectors, children_filter_dicts)
        microdata_time = time.time() - t_microdata

    print(f'  [Node {node_id}] - real {real_time:.1f}s - rounding {rounding_time:.1f}s')

    return microdata_time