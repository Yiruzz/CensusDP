import numpy as np

import opendp.prelude as dp
dp.enable_features("contrib")

# Numpy
# Ref: https://numpy.org/doc/2.4/reference/random/generated/numpy.random.normal.html
#      https://numpy.org/doc/2.4/reference/random/generated/numpy.random.laplace.html
def sample_dgauss_fast(scale: float, n_samples: int) -> np.ndarray:
    '''Generate samples of discrete Gaussian noise by sampling from a continuous Gaussian
    distribution and rounding the values.

    Args:
        scale (float): Noise scale parameter (scale = sensitivity / epsilon).
        n_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Integer-valued noise samples obtained by rounding
        Gaussian-distributed values.
    '''
    noise = np.random.normal(0.0, scale, n_samples)
    return np.round(noise).astype(int)

def sample_dlaplace_fast(scale: float, n_samples: int) -> np.ndarray:
    '''Generate samples of discrete Laplace noise by sampling from a continuous Laplace
    distribution and rounding the values.

    Args:
        scale (float): Noise scale parameter (scale = sensitivity / epsilon).
        n_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Integer-valued noise samples obtained by rounding
        Laplace-distributed values.
    '''
    noise = np.random.laplace(0.0, scale, n_samples)
    return np.round(noise).astype(int)

# OpenDP
# Ref: https://docs.opendp.org/en/stable/api/user-guide/measurements/additive-noise-mechanisms.html
def sample_dgauss_optimized(scale: float, n_samples: int) -> np.ndarray:
    '''Generate samples of discrete Gaussian noise using the optimized OpenDP mechanism.

    Args:
        scale (float): Noise scale parameter (scale = sensitivity / epsilon).
        n_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: A list of noisy values obtained using the Gaussian mechanism.
    '''
    mech =  dp.m.make_gaussian(
        input_domain=dp.vector_domain(dp.atom_domain(T=int)),
        input_metric=dp.l2_distance(T=int),
        scale=scale
    )
    return np.array(mech([0]*n_samples))

def sample_dlaplace_optimized(scale: float, n_samples: int) -> np.ndarray:
    '''Generate samples of discrete Laplace noise using the optimized OpenDP mechanism
    for the discrete Laplace distribution (i.e., using the geometric distribution).

    Args:
        scale (float): Noise scale parameter (scale = sensitivity / epsilon).
        n_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Integer-valued noise samples obtained by rounding
        Laplace-distributed values.
    '''
    mech = dp.m.make_geometric(
        input_domain=dp.vector_domain(dp.atom_domain(T=int)),
        input_metric=dp.l1_distance(T=int),
        scale=scale
    )
    return np.array(mech([0]*n_samples))