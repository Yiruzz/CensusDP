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


def build_optimizer(backend: str, params: Tuple[Any, ...]):
    """Instantiate the requested optimizer backend.

    Args:
        backend (str): One of BACKENDS.
        params (Tuple): (dtype, lp_problems_dir, solver_options), forwarded to the model.

    Returns:
        The optimizer model instance for the selected backend.

    Raises:
        ValueError: If the backend name is not recognised.
        ImportError: If 'pyoptinterface' is selected but the package is not installed.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown optimizer backend '{backend}'. Expected one of {BACKENDS}.")

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
