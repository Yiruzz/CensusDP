import numpy as np
from itertools import chain

from optimizer import OptimizationModel

from multiprocessing import shared_memory, get_context
from typing import Callable, Dict, List, Tuple, Any

def init_process(solver_name: str, solver_options: Dict[str, Any], optimizer_path: str, name: str, shape: Tuple[int, int], dtype: str) -> None:
    '''Initialize the process by creating a solver instance and
    attaching it to the shared memory space containing the
    contingency vectors.

    Args:
        solver_name (str): Name of the solver to use (e.g., 'gurobi').
        solver_options (Dict[str, Any]): Dictionary of options used to configure the solver.
        name (str): Name of the shared memory block.
        shape (Tuple[int, int]): Shape of the NumPy array view over the shared memory.
        dtype (str): Data type of the shared memory array.
    '''
    global process_solver, shm, arr

    process_solver = OptimizationModel(solver_name, solver_options, optimizer_path)

    shm = shared_memory.SharedMemory(name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return None

def combine_vectors(child_indices) -> np.ndarray:
    '''Retrieve the contingency vectors of the children of node and combine them into a single vector.

    Returns:
        np.ndarray: A 1D NumPy array containing the concatenated values.
    '''
    childs_contingency_vectors = [arr[idx] for idx in child_indices]
    joint_contingency_vector = np.concatenate(childs_contingency_vectors)
    return joint_contingency_vector

def generate_constraints(node_id, joint_contingency_vector) -> list[Callable]:
    """
    Retrieve the constraints of the children and store them in a list, 
    adjusting the indices to match the new joint contingency vector that will be applied.

    Returns:
        node: 
        list[Callable]: A list of callable objects with fixed parameters.
    """
    constraints = []

    # All vectors have the same length
    vector_length = arr.shape[1]
    contingency_vector = arr[node_id]

    # Consistency constraint: sum of children = parent
    for index in range(vector_length):
        # Parent's contingency vector value at 'index' must equal sum of children's values at 'index'
        # Precompute the indices to sum to avoid slice notation incompatible with Pyomo vars
        indices_to_sum = list(range(index, len(joint_contingency_vector), vector_length))
        constraints.append(lambda joint_array, idxs=indices_to_sum, value=contingency_vector[index]:
                                        sum(joint_array[j] for j in idxs) == value)

    return constraints

def update_vectors(child_indices,  joint_solution: np.ndarray) -> None:
    """
    Update the child vectors with the solution from the estimation phase. 
    The provided list will have sufficient size for all children of the node and will respect the order of the children.
    """
    vector_length = arr.shape[1]
    start = 0
    for idx in child_indices:
        end = start + vector_length
        arr[idx, :] = joint_solution[start:end]
        start = end

def solve(node_id: int, child_indices: List[int], child_constraints: List[Callable] = []) -> int:
    ''' Task executed in a separate process to solve a node in parallel.

    The solver instance is stored in a global variable for the process, 
    so it does not need to be passed as an argument each time.

    Args:
        node_id (int): Unique identifier for this hierarchical node.
        joint_contingency_vector (np.ndarray[int]): Contingency vector constructed from the node's children.
        joint_constraints (List[Callable]): List of constraints associated with the joint contingency vector.

    Returns:
        Tuple[int, np.ndarray[int]]: A tuple containing the node ID and the resulting solution of the problem.
                                     This allows mapping the solution back to the corresponding node.
    '''
    joint_contingency_vector = combine_vectors(child_indices)
    base_constraints = generate_constraints(
        node_id,
        joint_contingency_vector
    )

    estimated_solution = process_solver.non_negative_real_estimation(
        contingency_vector=joint_contingency_vector,
        node_id=node_id,
        constraints = chain(base_constraints, child_constraints)
    )

    joint_solution: np.ndarray = process_solver.rounding_estimation(
        x_tilde=estimated_solution,
        node_id=node_id,
        constraints = chain(base_constraints, child_constraints)
    )

    update_vectors(
        child_indices,
        joint_solution
    )

    return node_id