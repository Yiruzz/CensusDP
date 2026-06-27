import numpy as np
from dask.distributed import get_worker

def generate_noise_row(node_id: int, level: int) -> None:
    '''Generate a single noise vector for one tree node and write it to the shared Zarr array.

    Reads worker-local state (privacy mechanism, data handler, Zarr array) set up by
    WorkerInitializer.setup(). Noise is generated in column blocks of COL_BLOCK_SIZE to
    bound peak memory per task when n_cells is very large.

    Args:
        node_id (int): BFS-order node ID used as the row index into the Zarr noise array.
        level (int): Tree level of the node; selects the per-level privacy parameter
            (e.g. epsilon or rho) from the privacy mechanism.
    '''
    worker = get_worker()

    COL_BLOCK_SIZE = 100_000
    out = np.zeros(worker.data_handler.n_cells, dtype=worker.dtype)
    for start in range(0, worker.data_handler.n_cells, COL_BLOCK_SIZE):
        end = min(start + COL_BLOCK_SIZE, worker.data_handler.n_cells)
        worker.privacy_mechanism.add_noise(out[start:end], level, worker.q_sensitivity)
    worker.noisy_arr[node_id, :] = out
