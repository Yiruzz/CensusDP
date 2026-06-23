import numpy as np
from privacy import MECHANISMS

from typing import List


def initialize_mechanism(privacy_mech_name: str, level_params: List[float], cells: int, data_type: str, sensitivity: int) -> None:
    '''Initialize the privacy mechanism in each worker process.

    Args:
        privacy_mech_name (str): Name of the privacy mechanism (e.g., 'Laplace', 'Gaussian')
        level_params (List[float]): Parameters for the mechanism at each hierarchical level
        cells (int): Total number of contingency cells (noise vector length)
        data_type (str): NumPy data type for the noise arrays (e.g., 'float64')
    '''
    global privacy_mechanism, n_cells, dtype, q_sensitivity
    privacy_mechanism = MECHANISMS[privacy_mech_name](level_params)
    n_cells = cells
    dtype = data_type
    q_sensitivity = sensitivity

def generate_noise_row(level: int) -> np.ndarray:
    '''Generate a single noise vector in parallel.

    Args:
        level (int): Hierarchical level of the node (determines which mechanism parameter to use)

    Returns:
        np.ndarray: Noise vector of shape (n_cells,) with dtype specified in initialize_mechanism
    '''
    COL_BLOCK_SIZE = 100_000
    out = np.zeros(n_cells, dtype=dtype)
    for start in range(0, n_cells, COL_BLOCK_SIZE):
        end = min(start + COL_BLOCK_SIZE, n_cells)
        privacy_mechanism.add_noise(out[start:end], level, q_sensitivity)
    return out
