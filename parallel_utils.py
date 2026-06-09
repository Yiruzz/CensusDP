import numpy as np
import time
from scipy.sparse import spmatrix

from optimizer import OptimizationModel
from data_handler import DataHandler

from typing import List, Callable

def init_process(solver_options: dict, dir: str, query_matrix: spmatrix, length: int, check: bool = False) -> None:
    '''Initialize global variables for parallel worker processes.

    Args:
        solver_options (dict): Dictionary of options to pass to the optimization solver.
        dir (str): Directory path for spilling vectors to disk.
        query_matrix (spmatrix): The sparse query matrix Q used in optimization.
        length (int): The length of the contingency vectors.
    '''
    global _optimizer, _data_handler, _Q, _vectors_length, _check

    _optimizer = OptimizationModel(solver_options=solver_options)
    _data_handler = DataHandler()
    _data_handler.spill_dir = dir
    _Q = query_matrix
    _vectors_length = length
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
    for index in range(_vectors_length):
        indices_to_sum = [index + i * _vectors_length for i in range(num_children)]
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

def estimate_and_update_children(geo_id: int, node_path: str, children_paths: List[str], constraints: List[Callable] = []) -> None:
    '''Solve optimization for a node considering its children and update their vectors.

    Args:
        geo_id (int): The geographic ID of the parent node.
        node_path (str): File path to the parent node's contingency vector.
        children_paths (List[str]): List of file paths to children's contingency vectors.
        constraints (List[Callable]): List of constraint functions for the optimization. Defaults to empty List.
    '''

    contingency_vector = _data_handler.load_vector(node_path) 
    joint_contingency_vector = _data_handler.combine_child_vectors(children_paths)
    joint_constraints = _combine_child_constraints(len(children_paths), contingency_vector, constraints)

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

    _data_handler.update_child_vectors(joint_solution, _vectors_length, children_paths)

    print(f'  [Node {geo_id}] - real {real_time:.1f}s - rounding {rounding_time:.1f}s')