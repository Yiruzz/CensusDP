import numpy as np
import time

from optimizer import OptimizationModel

from multiprocessing import shared_memory, get_context
from typing import Callable, Dict, List, Tuple, Any

def init_process(solver_name: str, solver_options: Dict[str, Any], optimizer_path: str,
                 name: str, shape: Tuple[int, int], dtype: str) -> None:
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

def combine_vectors(child_indices: List[int]) -> np.ndarray:
    """Retrieve the contingency vectors of the children of node and combine them into a single vector.
    Args:
        child_indices (List[int]): List of indices representing the children of the node.
    
    Returns:
        np.ndarray: A 1D NumPy array containing the concatenated values.
    """
    childs_contingency_vectors = [arr[idx] for idx in child_indices]
    joint_contingency_vector = np.concatenate(childs_contingency_vectors)
    return joint_contingency_vector

def generate_constraints(node_id: int, child_constraints: Dict[int, Callable], joint_contingency_vector: np.ndarray) -> list[Callable]:
    """
    Retrieve the constraints of the children and store them in a list, 
    adjusting the indices to match the new joint contingency vector that will be applied.

    Args:
        node_id (int): Unique identifier of the hierarchical node. Used to access the corresponding contingency vector.
        child_indices (Dict[int, Callable]): List of indices representing the children of the node. Used to access and organize child-related data.
        joint_contingency_vector (np.ndarray): 1D NumPy array containing the concatenated child contingency vectors.

    Returns:
        np.ndarray: A 1D NumPy array containing the concatenated values.
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

    start = 0
    for idx in child_constraints.keys():
        end = start + vector_length
        for constraint in child_constraints[idx]:
            constraints.append(lambda joint_array, s=start, e=end, c=constraint: c({i - s: joint_array[i] for i in range(s, e)}))
        start = end

    return constraints

def update_vectors(child_indices: List[int], joint_contingency_vector: np.ndarray) -> None:
    """
    Update the child vectors with the solution obtained from the estimation phase.
    The provided list has sufficient size for all children of the node and preserves the
    order of the children.

    Args:
        child_indices (List[int]): List of indices representing the children of the node. Used to access and organize child-related data.
        joint_contingency_vector (np.ndarray): 1D NumPy array containing the concatenated child contingency vectors.
    """
    vector_length = arr.shape[1]
    start = 0
    for idx in child_indices:
        end = start + vector_length
        arr[idx, :] = joint_contingency_vector[start:end]
        start = end

def solve(node_id: int, child_constraints: Dict[int, Callable]) -> Tuple[int, float]:
    """Task executed in a separate process to solve a node in parallel.

    The solver instance is stored as a global variable within each process,
    so it does not need to be passed as an argument on each invocation.

    Args:
        node_id (int): Unique identifier of the hierarchical node. Used to name the problem instance and access the corresponding contingency vector.
        child_constraints (Dict[int, Callable]): Dictionary mapping each child node ID to its associated constraints.
                                                 Keys allow access to the contingency vectors, and values are used to construct
                                                 constraints over the joint contingency vector.

    Returns:
        Tuple[int, float]: Node ID of the node whose computation has already completed (used for tracking completion) and the time taken to resolve the computation.
    """

    t1 = time.time()
    joint_contingency_vector = combine_vectors(child_constraints.keys())
    constraints = generate_constraints(
        node_id,
        child_constraints,
        joint_contingency_vector
    )

    estimated_solution = process_solver.non_negative_real_estimation(
        contingency_vector=joint_contingency_vector,
        node_id=node_id,
        constraints = constraints
    )

    joint_solution = process_solver.rounding_estimation(
        x_tilde=estimated_solution,
        node_id=node_id,
        constraints = constraints
    )

    update_vectors(
        child_constraints.keys(),
        joint_solution
    )

    t2 = time.time()-t1
    return node_id, t2