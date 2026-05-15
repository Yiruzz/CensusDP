"""Vectorized discrete-noise samplers used by the privacy mechanisms.

OpenDP mechanisms are pure Python objects whose construction is non-trivial: building one
per call (as the previous version did) made `add_noise` allocate a new mechanism for every
node in the tree, which dominated measurement-phase time on large hierarchies.

Each `(mechanism_kind, scale)` pair now maps to a single cached mechanism object. Scales
are derived from per-level privacy parameters and the (fixed) query sensitivity, so the
cache stays tiny — at most one entry per (mechanism, level) once the run is warmed up.
"""

import numpy as np
import opendp.prelude as dp

# NOTE: Enable the "contrib" features to access the optimized noise sampling mechanisms. 
#       Consider that this implementation may not be flotaing point safe.
dp.enable_features("contrib")

# Cache keyed by scale. Same input_domain / input_metric every time, so scale is the only
# distinguishing parameter.
_gaussian_cache: dict[float, object] = {}
_geometric_cache: dict[float, object] = {}


def _gaussian_mechanism(scale: float):
    mech = _gaussian_cache.get(scale)
    if mech is None:
        mech = dp.m.make_gaussian(
            input_domain=dp.vector_domain(dp.atom_domain(T=int)),
            input_metric=dp.l2_distance(T=int),
            scale=scale,
        )
        _gaussian_cache[scale] = mech
    return mech


def _geometric_mechanism(scale: float):
    mech = _geometric_cache.get(scale)
    if mech is None:
        mech = dp.m.make_geometric(
            input_domain=dp.vector_domain(dp.atom_domain(T=int)),
            input_metric=dp.l1_distance(T=int),
            scale=scale,
        )
        _geometric_cache[scale] = mech
    return mech


# OpenDP
# Ref: https://docs.opendp.org/en/stable/api/user-guide/measurements/additive-noise-mechanisms.html
def sample_dgauss_optimized(scale: float, n_samples: int) -> np.ndarray:
    '''Sample n_samples integer values from the discrete Gaussian with the given standard deviation.

    Args:
        scale (float): Standard deviation σ of the discrete Gaussian.
        n_samples (int): Number of independent samples to draw.

    Returns:
        np.ndarray: Integer-valued noise samples, shape (n_samples,).
    '''
    return np.asarray(_gaussian_mechanism(scale)([0] * n_samples))


def sample_dlaplace_optimized(scale: float, n_samples: int) -> np.ndarray:
    '''Sample n_samples integer values from the discrete (two-sided) geometric distribution,
    i.e. the discrete Laplace with the given b parameter.

    Args:
        scale (float): Discrete-Laplace b parameter.
        n_samples (int): Number of independent samples to draw.

    Returns:
        np.ndarray: Integer-valued noise samples, shape (n_samples,).
    '''
    return np.asarray(_geometric_mechanism(scale)([0] * n_samples))


# Numpy approximations — kept as fast non-OpenDP fallbacks for benchmarking / smoke tests.
# Ref: https://numpy.org/doc/2.4/reference/random/generated/numpy.random.normal.html
#      https://numpy.org/doc/2.4/reference/random/generated/numpy.random.laplace.html
def sample_dgauss_fast(scale: float, n_samples: int) -> np.ndarray:
    '''Round a continuous Gaussian — approximates discrete Gaussian; faster but not pure DP.'''
    noise = np.random.normal(0.0, scale, n_samples)
    return np.round(noise).astype(int)


def sample_dlaplace_fast(scale: float, n_samples: int) -> np.ndarray:
    '''Round a continuous Laplace — approximates discrete Laplace; faster but not pure DP.'''
    noise = np.random.laplace(0.0, scale, n_samples)
    return np.round(noise).astype(int)
