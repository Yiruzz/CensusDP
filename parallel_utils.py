import numpy as np
import time
import os
from scipy.sparse import spmatrix

from optimizer import OptimizationModel
from data_handler import DataHandler
from domain import ContingencyDomain
from privacy import PureDP, ZCDP, ApproximateDP, RenyiDP

from typing import List, Callable, Dict, Any, Optional

def init_process(solver_options: dict, spill_dir: str, microdata_dir: str, query_matrix: spmatrix,
                 parquet_path: str, hierarchical_columns: List[str], query_columns: List[str],
                 domain_dict: Dict[str, Any], constraints_dict: Optional[Dict[int, List]], privacy_name: str,
                 level_params: List[float], delta: Optional[float] = None,
                 alphas: Optional[List[float]] = None, query_sensitivity: int = 1,
                 check: bool = False) -> None:
    '''Initialize global variables for parallel worker processes.

    Args:
        solver_options (dict): Dictionary of options to pass to the optimization solver.
        spill_dir (str): Directory path for spilling vectors to disk.
        microdata_dir (str): Directory path for temporary microdata files.
        query_matrix (spmatrix): The sparse query matrix Q used in optimization.
        parquet_path (str): Path to the parquet file.
        hierarchical_columns (List[str]): Hierarchical column names.
        query_columns (List[str]): Query column names.
        domain_dict (Dict[str, Any]): Domain mapping for query columns.
        constraints_dict (Optional[Dict[int, List]]): Constraints mapped by level. TODO: implement constraints support.
        privacy_name (str): Name of the privacy mechanism ("PureDP", "ZCDP", "ApproximateDP", "RenyiDP").
        level_params (List[float]): Per-level privacy parameters.
        delta (Optional[float]): Delta parameter for ApproximateDP and RenyiDP.
        alphas (Optional[List[float]]): Alpha values for RenyiDP.
        query_sensitivity (int): Query sensitivity for noise addition.
        check (bool): Whether to check node correctness.
    '''
    global _optimizer, _data_handler, _Q, _vectors_length, _check, _privacy_mechanism, _query_sensitivity  # _constraints

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

    # Create worker-specific microdata file (CSV format)
    _data_handler.worker_microdata_file = os.path.join(microdata_dir, f'worker_{os.getpid()}.csv')
    # File will be created when first data is written

    _Q = query_matrix
    _query_sensitivity = query_sensitivity
    # TODO: constraints support - uncomment when ready
    # _constraints = constraints_dict

    # Instantiate privacy mechanism
    if privacy_name == "PureDP":
        _privacy_mechanism = PureDP(level_params)
    elif privacy_name == "ZCDP":
        _privacy_mechanism = ZCDP(level_params)
    elif privacy_name == "ApproximateDP":
        _privacy_mechanism = ApproximateDP(level_params, delta)
    elif privacy_name == "RenyiDP":
        _privacy_mechanism = RenyiDP(level_params, delta, alphas)
    else:
        raise ValueError(f"Unknown privacy mechanism: {privacy_name}")

    _check = check

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
    # start = 0
    # for child in range(num_children):
    #     end = start + _vectors_length
    #     for constraint in constraints[child]:
    #         joint_constraints.append(
    #             lambda joint_array, s=start, e=end, c=constraint:
    #                 c({i - s: joint_array[i] for i in range(s, e)})
    #         )
    #     start = end

    # Add consistency constraints: parent value at each index = sum of child values at that index
    for index in range(_data_handler.contingency_df_length):
        indices_to_sum = [index + i * _data_handler.contingency_df_length for i in range(num_children)]
        joint_constraints.append(
            lambda joint_array, idxs=indices_to_sum, value=contingency_vector[index]:
                sum(joint_array[j] for j in idxs) == value
        )

    return joint_constraints 

def _check_node_correctness(parent_vector: np.ndarray, children_vectors: np.ndarray) -> None:
    """
    Checks that the sum of the values in the parent node vector 
    is equal to the sum of the values in its children vectors.

    Args:
        parent_vector (np.ndarray): Contingency vector of the parent node.
        children_vectors (np.ndarray): Contingency vectors of the child nodes.
    """
    parent_sum = np.sum(parent_vector)
    children_sum = np.sum(children_vectors)
    
    if parent_sum != children_sum:
        print(f"\nError: The sum of the children nodes' contingency vectors "
              f"({children_sum}) does not equal the parent node's contingency vector ({parent_sum}).")

def estimate_and_update_children(geo_id: int, node_path: str, children_filter_dicts: List[Dict[str, Any]], children_level: int, is_leaf: bool = False, constraints: List[Callable] = []) -> Optional[float]:
    '''Solve optimization for a node considering its children and update their vectors.

    Args:
        geo_id (int): The geographic ID of the parent node.
        node_path (str): Path to contigency vector file.
        children_filter_dicts (List[Dict[str, Any]]): List of filter dictionaries for each child.
        children_level (int): Level of all children (they all share the same level).
        is_leaf (bool): Whether children are leaf nodes. Defaults to False.
        constraints (List[Callable]): List of constraint functions for the optimization. Defaults to empty List.

    Returns:
        Optional[float]: Time spent writing microdata files, or None if not writing.
    '''
    contingency_vector = _data_handler.load_vector(node_path)

    # Materialize and combine children vectors in memory
    children_vectors = []

    for filter_dict in children_filter_dicts:
        # TODO: constraints support - uncomment when ready
        # child_constraints = _constraints[children_level]
        child_constraints = []  # placeholder: use empty constraints for now

        # Materialize the child node in this worker
        child_vector, _ = _data_handler.materialize_node_data(filter_dict, child_constraints, _Q)
        _privacy_mechanism.add_noise(child_vector, children_level, _query_sensitivity)
        children_vectors.append(child_vector)

    # Concatenate children vectors in memory
    joint_contingency_vector = np.concatenate(children_vectors)

    joint_constraints = _combine_child_constraints(len(children_filter_dicts), contingency_vector, constraints)

    t1 = time.time()
    x_tilde = _optimizer.non_negative_real_estimation(
        noisy_measurements=joint_contingency_vector,
        node_id=geo_id,
        constraints=joint_constraints,
        query_matrix=_Q
    )
    real_time = time.time() - t1

    t1 = time.time()
    joint_solution = _optimizer.rounding_estimation(
        x_tilde=x_tilde,
        node_id=geo_id,
        constraints=joint_constraints
    )
    rounding_time = time.time() - t1

    if _check: _check_node_correctness(contingency_vector, joint_contingency_vector)

    microdata_time = 0.0

    if not is_leaf:
        _data_handler.update_child_vectors(joint_solution, _data_handler.contingency_df_length, children_filter_dicts)
    else:
        t_microdata = time.time()
        start = 0
        for filter_dict in children_filter_dicts:
            end = start + _data_handler.contingency_df_length
            updated_vector = joint_solution[start:end]
            _data_handler.append_microdata_to_worker_file(updated_vector, filter_dict)
            start = end
        microdata_time = time.time() - t_microdata

    print(f'  [Node {geo_id}] - real {real_time:.1f}s - rounding {rounding_time:.1f}s')

    return microdata_time