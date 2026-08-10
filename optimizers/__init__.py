"""Optimizer backends for the estimation phase.

Two interchangeable models solve the same two-stage problem (real-valued QP, then a
rounding MIP) over the same SparseConstraint inputs:

  - ``write_lp``:        emits Gurobi LP text directly (OptimizationModelLP).
  - ``pyoptinterface``:  builds the model through the pyoptinterface bindings.

pyoptinterface is an OPTIONAL dependency and is not listed in requirements.txt, so it is
imported lazily - only when that backend is actually selected. Importing it at module
level would make the whole pipeline unimportable on an installation that only uses
``write_lp``.
"""
from typing import Any, Tuple

BACKENDS = ("write_lp", "pyoptinterface")

# Gurobi solves the QP with barrier and leaves crossover off, so what comes back is an
# interior point, with every variable strictly positive. At the default 1e-8 that spreads a
# little mass across cells whose true value is zero, which costs the estimate its sparsity.
# 1e-12 have a better chance of preserving it. Sparsity is worth protecting for its own sake here: the support is what
# sizes everything downstream, so a polluted solution is both less accurate and more expensive.
#
# Overridable: caller-provided solver_options take precedence.
DEFAULT_SOLVER_OPTIONS = {"BarConvTol": 1e-12}


def build_optimizer(backend: str, params: Tuple[Any, ...]):
    """Instantiate the requested optimizer backend.

    Args:
        backend (str): One of BACKENDS.
        params (Tuple): (dtype, lp_problems_dir, solver_options), forwarded to the model.
            DEFAULT_SOLVER_OPTIONS is merged underneath solver_options.

    Returns:
        The optimizer model instance for the selected backend.

    Raises:
        ValueError: If the backend name is not recognised.
        ImportError: If 'pyoptinterface' is selected but the package is not installed.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown optimizer backend '{backend}'. Expected one of {BACKENDS}.")

    dtype, lp_problems_dir, solver_options = params
    params = (dtype, lp_problems_dir, {**DEFAULT_SOLVER_OPTIONS, **(solver_options or {})})

    if backend == "pyoptinterface":
        try:
            from .pyoptinterface import OptimizationModel
        except ImportError as exc:
            raise ImportError(
                "The 'pyoptinterface' backend requires the pyoptinterface package, which is "
                "not installed (pip install pyoptinterface). Use backend='write_lp' instead."
            ) from exc
        return OptimizationModel(*params)

    from .write_lp_directly import OptimizationModelLP
    return OptimizationModelLP(*params)
