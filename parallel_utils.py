import numpy as np
import time

from optimizer import OptimizationModel
from constraints.constraint import SparseRow

from multiprocessing import shared_memory
from typing import Dict, List, Tuple, Any


def init_process(solver_name: str, solver_options: Dict[str, Any], optimizer_path: str,
                 name: str, shape: Tuple[int, int], dtype: str) -> None:
    """Initialize the worker process: create a solver instance and attach to shared memory."""
    global process_solver, shm, arr

    process_solver = OptimizationModel(solver_name, solver_options, optimizer_path)

    shm = shared_memory.SharedMemory(name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return None


def combine_vectors(child_indices: List[int]) -> np.ndarray:
    """Concatenate the contingency vectors of the children into one joint vector."""
    childs_contingency_vectors = [arr[idx] for idx in child_indices]
    return np.concatenate(childs_contingency_vectors)


def generate_constraints(node_id: int, child_constraints: Dict[int, List[SparseRow]]) -> List[SparseRow]:
    """Build the joint sparse constraints for the children optimization problem.

    Two kinds of rows are produced:
    1. Consistency: sum over each child of x[child_offset + i] == parent[i], one row per
       index i of the contingency vector.
    2. Per-child rows from `child_constraints`, with their indices shifted by the child's
       offset inside the joint vector.
    """
    rows: List[SparseRow] = []

    vector_length = arr.shape[1]
    parent = arr[node_id]
    n_children = len(child_constraints)
    starts = np.arange(n_children, dtype=np.int64) * vector_length

    # Consistency: each index i across all children
    for i in range(vector_length):
        idxs = starts + i
        rows.append(SparseRow(
            indices=idxs,
            coefs=np.ones(n_children, dtype=np.float64),
            sense='=',
            rhs=float(parent[i]),
        ))

    # Per-child user/aggregate constraints, shifted into the joint vector.
    for offset, child_id in enumerate(child_constraints.keys()):
        start = offset * vector_length
        for row in child_constraints[child_id]:
            rows.append(row.offset(start))

    return rows


def update_vectors(child_indices: List[int], joint_contingency_vector: np.ndarray) -> None:
    """Write the joint solution back to each child's slot in shared memory."""
    vector_length = arr.shape[1]
    start = 0
    for idx in child_indices:
        end = start + vector_length
        arr[idx, :] = joint_contingency_vector[start:end]
        start = end


def solve(node_id: int, child_constraints: Dict[int, List[SparseRow]]) -> Tuple[int, float]:
    """Worker task: build the joint problem for a node's children and solve it.

    Args:
        node_id: id of the parent node (its contingency vector lives in shared memory).
        child_constraints: mapping from child id to its list of SparseRow constraints.

    Returns:
        (node_id, elapsed_seconds)
    """
    t1 = time.time()

    joint_contingency_vector = combine_vectors(list(child_constraints.keys()))
    constraints = generate_constraints(node_id, child_constraints)

    estimated_solution = process_solver.non_negative_real_estimation(
        contingency_vector=joint_contingency_vector,
        node_id=node_id,
        constraints=constraints,
    )

    joint_solution = process_solver.rounding_estimation(
        x_tilde=estimated_solution,
        node_id=node_id,
        constraints=constraints,
    )

    update_vectors(list(child_constraints.keys()), joint_solution)

    return node_id, time.time() - t1
