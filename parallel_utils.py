import numpy as np
import time
import os
from scipy.sparse import spmatrix

from optimizer import OptimizationModel
from data_handler import DataHandler
from domain import ContingencyDomain
from privacy import PrivacyMechanism

from typing import List, Callable, Dict, Any

def init_process(solver_options: dict, constraints_dict: Dict[int, List],
                 spill_dir: str, microdata_dir: str, parquet_path: str,
                 domain_dict: Dict[str, Any], hierarchical_columns: List[str], query_columns: List[str],
                 privacy_mechanism: PrivacyMechanism, query_matrix: spmatrix, query_sensitivity: int, check: bool) -> None:
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
    global _optimizer, _data_handler, _Q, _vectors_length, _check, _privacy_mechanism, _query_sensitivity, _constraints

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
    contingency_df_length = len(contingency_vector)

    # Wrap child publication constraints with adjusted indices
    start = 0     
    for child_constraints in constraints:
        end = start + contingency_df_length
        for constraint in child_constraints:
            joint_constraints.append(
                lambda joint_array, s=start, e=end, c=constraint:
                    c({i - s: joint_array[i] for i in range(s, e)})
            )
        start = end

    # Add consistency constraints: parent value at each index = sum of child values at that index
    for index in range(contingency_df_length):
        indices_to_sum = [index + i * contingency_df_length for i in range(num_children)]
        joint_constraints.append(
            lambda joint_array, idxs=indices_to_sum, value=contingency_vector[index]:
                sum(joint_array[j] for j in idxs) == value
        )

    return joint_constraints

def _real_and_round_estimation(optimizer: OptimizationModel, measurements: np.ndarray, node_id: int, constraints: List[Callable], Q: spmatrix) -> np.ndarray:
    '''Run non-negative real estimation followed by rounding estimation for a node.

    Args:
        optimizer (OptimizationModel): Solver instance used for both estimation steps.
        measurements (np.ndarray): Noisy contingency measurements to optimize over.
        node_id (int): Unique ID of the node being processed.
        constraints (List): Constraint functions to enforce during optimization.
        Q (spmatrix): Query matrix applied in the real estimation step.

    Returns:
        np.ndarray: Solution vector.
    '''
    t1 = time.time()
    x_tilde = optimizer.non_negative_real_estimation(
        noisy_measurements=measurements,
        node_id=node_id,
        constraints=constraints,
        query_matrix=Q
    )
    real_time = time.time() - t1

    t1 = time.time()
    solution = optimizer.rounding_estimation(
        x_tilde=x_tilde,
        node_id=node_id,
        constraints=constraints
    )
    rounding_time = time.time() - t1

    print(f'  [Node {node_id}] - real {real_time:.1f}s - rounding {rounding_time:.1f}s')
    return solution

def _split_child_vectors(joint_solution: np.ndarray, num_children: int, vec_length: int) -> List[np.ndarray]:
    '''Split a concatenated joint solution vector into individual child vectors.

    Args:
        joint_solution (np.ndarray): Concatenated solution vector for all children.
        num_children (int): Number of child nodes.
        vec_length (int): Length of each individual child vector.

    Returns:
        List[np.ndarray]: List of child solution vectors, one per child.
    '''
    slices, start = [], 0
    for _ in range(num_children):
        slices.append(joint_solution[start:start + vec_length])
        start += vec_length
    return slices

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

def estimate_and_update_children(node_id: int, node_path: str, children_filter_dicts: List[Dict[str, Any]], children_level: int, is_leaf: bool = False) -> float:
    '''Solve optimization for a node considering its children and update their vectors.

    Args:
        node_id (int): The unique ID of the parent node.
        node_path (str): Path to contigency vector file.
        children_filter_dicts (List[Dict[str, Any]]): List of filter dictionaries for each child.
        children_level (int): Level of all children (they all share the same level).
        is_leaf (bool): Whether children are leaf nodes. Defaults to False.

    Returns:
        float: Time spent writing microdata files
    '''
    contingency_vector = _data_handler.load_vector(node_path)

    # Materialize and combine children vectors and constraints
    children_vectors = []
    children_constraints = []

    for filter_dict in children_filter_dicts:
        child_vector, child_constraint = _data_handler.materialize_node_data(filter_dict, _constraints[children_level], _Q)
        _privacy_mechanism.add_noise(child_vector, children_level, _query_sensitivity)
        
        children_vectors.append(child_vector)
        children_constraints.append(child_constraint)

    # Concatenate children vectors, and constraints are adapted to the new vector size.
    # Also create others to ensure consistency in the number of rows per category in the parent.
    # The number of rows in the parent category must match the sum of rows of that category across all children.
    joint_contingency_vector = np.concatenate(children_vectors)
    joint_constraints = _combine_child_constraints(len(children_filter_dicts), contingency_vector, children_constraints)
    joint_solution = _real_and_round_estimation(_optimizer, joint_contingency_vector, node_id, joint_constraints, _Q)

    if _check: _check_node_correctness(contingency_vector, joint_solution)
    joint_contingency_vector = None
    joint_constraints = None

    microdata_time = 0.0
    if not is_leaf:
        _data_handler.update_child_vectors(joint_solution, _data_handler.contingency_df_length, children_filter_dicts)
    else:
        t_microdata = time.time()
        child_vectors = _split_child_vectors(joint_solution, len(children_filter_dicts), _data_handler.contingency_df_length)
        _data_handler.write_microdata(node_id, child_vectors, children_filter_dicts)
        microdata_time = time.time() - t_microdata

    return microdata_time