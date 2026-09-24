"""Optimizer backends for the estimation phase.

Two interchangeable models solve the same two-stage problem (real-valued QP, then a
rounding MIP) over the same SparseConstraint inputs:

  - ``write_lp``:        emits Gurobi LP text directly (OptimizationModelLP).
  - ``pyoptinterface``:  builds the model through the pyoptinterface bindings.

pyoptinterface is an OPTIONAL dependency and is not listed in requirements.txt, so it is
imported lazily - only when that backend is actually selected. Importing it at module
level would make the whole pipeline unimportable on an installation that only uses
``write_lp``.

The ROUNDING METHOD is a separate axis from the backend, because the sweep does not replace a
backend - it composes with one, and still needs it to solve the QP:

  - ``mip``:    the global binary program, the backend's own rounding_estimation.
  - ``sweep``:  junction-tree sweep of 2-way transports (sweep_rounding.SweepRoundingModel),
                factored pipeline only. ~21x faster on the real census instance.
"""
from typing import Any, Optional, Tuple

BACKENDS = ("write_lp", "pyoptinterface")
ROUNDING_METHODS = ("mip", "sweep")

# Gurobi solves the QP with barrier and leaves crossover off, so what comes back is an
# interior point, with every variable strictly positive. At the default 1e-8 that spreads a
# little mass across cells whose true value is zero, which costs the estimate its sparsity.
# 1e-12 have a better chance of preserving it. Sparsity is worth protecting for its own sake here: the support is what
# sizes everything downstream, so a polluted solution is both less accurate and more expensive.
#
# Overridable: caller-provided solver_options take precedence.
DEFAULT_SOLVER_OPTIONS = {"BarConvTol": 1e-12}


def build_optimizer(backend: str, params: Tuple[Any, ...], rounding: str = "mip",
                    data_handler: Optional[Any] = None):
    """Instantiate the requested optimizer backend and rounding method.

    Args:
        backend (str): One of BACKENDS.
        params (Tuple): (dtype, lp_problems_dir, solver_options), forwarded to the model.
            DEFAULT_SOLVER_OPTIONS is merged underneath solver_options.
        rounding (str): One of ROUNDING_METHODS. 'mip' (the default) returns the backend
            unchanged; 'sweep' wraps it so the integer step is solved by sweeping the junction
            tree, which requires data_handler.
        data_handler (Optional[DataHandler]): Required by 'sweep', which reads the junction tree
            and the bag layout from it. Must already have build_marginal_domains applied.

    Returns:
        An object exposing non_negative_real_estimation and rounding_estimation.

    Raises:
        ValueError: If the backend or rounding name is not recognised, or if 'sweep' is asked
            for without a junction tree.
        ImportError: If 'pyoptinterface' is selected but the package is not installed.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown optimizer backend '{backend}'. Expected one of {BACKENDS}.")
    if rounding not in ROUNDING_METHODS:
        raise ValueError(
            f"Unknown rounding method '{rounding}'. Expected one of {ROUNDING_METHODS}.")

    dtype, lp_problems_dir, solver_options = params
    solver_options = {**DEFAULT_SOLVER_OPTIONS, **(solver_options or {})}
    params = (dtype, lp_problems_dir, solver_options)

    if backend == "pyoptinterface":
        try:
            from .pyoptinterface import OptimizationModel
        except ImportError as exc:
            raise ImportError(
                "The 'pyoptinterface' backend requires the pyoptinterface package, which is "
                "not installed (pip install pyoptinterface). Use backend='write_lp' instead."
            ) from exc
        model = OptimizationModel(*params)
    else:
        from .write_lp_directly import OptimizationModelLP
        model = OptimizationModelLP(*params)

    if rounding == "sweep":
        from .sweep_rounding import SweepRoundingModel
        return SweepRoundingModel(model, data_handler, solver_options)
    return model
