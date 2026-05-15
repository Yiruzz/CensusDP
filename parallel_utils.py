import numpy as np
import time

from optimizer import OptimizationModel

from multiprocessing import shared_memory
from typing import Callable, Dict, List, Tuple, Any

def init_process(solver_name: str, solver_options: Dict[str, Any], optimizer_path: str,
                 name: str, shape: Tuple[int, int], dtype: str,
                 n_queries: int, n_cells: int, query_matrix: np.ndarray) -> None:
    '''Initialize the process by creating a solver instance and attaching it to the shared
    memory space containing the contingency vectors. The query matrix Q and the logical
    sizes (n_queries, n_cells) are stored as globals so they don't have to be sent on every
    task — they are pickled once per worker via initargs.

    Args:
        solver_name (str): Name of the solver to use (e.g., 'gurobi').
        solver_options (Dict[str, Any]): Dictionary of options used to configure the solver.
        optimizer_path (str): Path to the solver executable (or None).
        name (str): Name of the shared memory block.
        shape (Tuple[int, int]): Shape of the NumPy array view over the shared memory.
            shape[1] is vector_length = max(n_queries, n_cells); each row holds y in the
            first n_queries slots before estimation, and x_hat in the first n_cells slots after.
        dtype (str): Data type of the shared memory array.
        n_queries (int): Length of the noisy measurement vector y per node.
        n_cells (int): Length of the estimated cell-count vector x_hat per node.
        query_matrix (np.ndarray): Query matrix Q of shape (n_queries, n_cells).
    '''
    global process_solver, shm, arr, _n_queries, _n_cells, _Q

    process_solver = OptimizationModel(solver_name, solver_options, optimizer_path)

    shm = shared_memory.SharedMemory(name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    _n_queries = n_queries
    _n_cells = n_cells
    _Q = query_matrix
    return None

def combine_vectors(child_indices: List[int]) -> np.ndarray:
    """Concatenate each child's live noisy measurement y into a single 1D vector.

    Children are pre-estimation, so each row holds y in its first n_queries slots; the
    trailing padding (when vector_length > n_queries) is unused and excluded here.

    Args:
        child_indices (List[int]): Row indices of the children in the shared array.

    Returns:
        np.ndarray: Concatenation of children's y vectors, length n_children * n_queries.
    """
    childs_measurements = [arr[idx, :_n_queries] for idx in child_indices]
    return np.concatenate(childs_measurements)

def generate_constraints(node_id: int, child_constraints: Dict[int, List[Callable]]) -> List[Callable]:
    """Build the constraint list for the joint child estimation.

    Constraints are expressed in cell space (the solver's decision variable x is the
    concatenation of children's x_hat blocks, each of length n_cells). The parent has
    already been estimated, so the first n_cells slots of its row hold x_hat.

    Args:
        node_id (int): Row index of the parent in the shared array.
        child_constraints (Dict[int, List[Callable]]): Per-child constraint callables (each
            already expressed over cell-space indices 0..n_cells-1 by the data handler).

    Returns:
        List[Callable]: Constraints over the joint child vector of length n_children * n_cells.
    """
    constraints = []
    parent_cells = arr[node_id, :_n_cells]
    n_children = len(child_constraints)

    # Consistency: for each cell index, sum of children's x_hat at that cell == parent's x_hat at that cell.
    for index in range(_n_cells):
        indices_to_sum = list(range(index, n_children * _n_cells, _n_cells))
        constraints.append(lambda joint_array, idxs=indices_to_sum, value=int(parent_cells[index]):
                                        sum(joint_array[j] for j in idxs) == value)

    # Per-child user constraints — shift each child's cell-space indices into the joint vector.
    start = 0
    for idx in child_constraints.keys():
        end = start + _n_cells
        for constraint in child_constraints[idx]:
            constraints.append(lambda joint_array, s=start, e=end, c=constraint: c({i - s: joint_array[i] for i in range(s, e)}))
        start = end

    return constraints

def update_vectors(child_indices: List[int], joint_solution: np.ndarray) -> None:
    """Write each per-child x_hat block back into the first n_cells slots of that child's row.

    Args:
        child_indices (List[int]): Row indices of the children in the shared array.
        joint_solution (np.ndarray): Concatenated x_hat solution, length n_children * n_cells.
    """
    start = 0
    for idx in child_indices:
        end = start + _n_cells
        arr[idx, :_n_cells] = joint_solution[start:end]
        start = end

def solve(node_id: int, child_constraints: Dict[int, List[Callable]]) -> Tuple[int, float]:
    """Worker task: estimate the joint x_hat for a node's children.

    Reads each child's noisy y from shared memory, solves the real then rounding estimation
    using Q from the worker's globals, and writes the integer x_hat for each child back into
    the first n_cells slots of its row.

    Args:
        node_id (int): Row index of the parent (already estimated; holds x_hat in [:n_cells]).
        child_constraints (Dict[int, List[Callable]]): Per-child constraint callables.

    Returns:
        Tuple[int, float]: The node id processed and the wall-clock time spent on it.
    """
    t1 = time.time()
    child_ids = list(child_constraints.keys())
    joint_measurements = combine_vectors(child_ids)
    constraints = generate_constraints(node_id, child_constraints)

    x_tilde = process_solver.non_negative_real_estimation(
        noisy_measurements=joint_measurements,
        node_id=node_id,
        constraints=constraints,
        query_matrix=_Q,
    )

    joint_solution = process_solver.rounding_estimation(
        x_tilde=x_tilde,
        node_id=node_id,
        constraints=constraints,
    )

    update_vectors(child_ids, joint_solution)

    return node_id, time.time() - t1
