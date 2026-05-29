"""Shared utilities for the CensusDP empirical-evaluation framework.

Conventions
-----------
* "Truth" means the noise-free state of the algorithm — `x` per node (cell counts)
  or `Q @ x` per node (query answers).
* "Estimate" means the integer output of `estimation_phase()`: `x_hat` per node,
  with `y_hat = Q @ x_hat` computed on demand.
* Tree level numbering follows the codebase: 0 = root, then one per hierarchical
  column (so a 3-column hierarchy gives levels 0..3).

This module deliberately does not modify any pipeline file. It captures truth by
snapshotting `tree._contingency_vectors[:, :n_queries]` immediately after
`TopDown.initialize()` (which holds `Q @ x` at that point), and recovers per-cell
truth `x` via a separate Q=Identity tree build that is cached on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Ensure the project root is importable when scripts are run with `python -m tests.<name>`
# from anywhere. The project root is the parent of this tests/ folder.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Make stdout/stderr handle the non-ASCII glyphs we use in status lines (→, ±, …) even
# on Windows cp1252 consoles. Best-effort: ignored on streams that don't support reconfigure.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np 
import pandas as pd 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from topdown import TopDown
from privacy import PureDP, ZCDP, ApproximateDP, RenyiDP, PrivacyMechanism
from queries import QueryWorkload, col


# ─────────────────────────────────────────────────────────────────────────────
# Config file — datasets registry + experiment defaults
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH: Path = _PROJECT_ROOT / "tests" / "config.toml"

# Required keys per [datasets.X] table; the dataset entry is rejected if any are missing.
_REQUIRED_DATASET_KEYS: Tuple[str, ...] = ("path", "hierarchy", "sep", "default_queries")
# Required keys in [defaults]. Other fields are optional and use built-in fallbacks.
_REQUIRED_DEFAULTS_KEYS: Tuple[str, ...] = (
    "dataset", "process_until", "mechanism", "delta",
    "allocation", "trials", "extension", "solver_name",
)
# [defaults.budgets] must contain one entry per known mechanism.
_REQUIRED_BUDGET_KEYS: Tuple[str, ...] = (
    "PureDP", "ZCDP", "ApproximateDP", "RenyiDP",
)


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the TOML config and minimally validate it.

    The returned dict has two top-level keys: 'defaults' (dict) and 'datasets'
    (dict of dataset_name -> dict). Errors mention the config path so the user
    knows where to look.
    """
    path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"config file not found at {path} — "
            f"copy tests/config.toml from the repo or pass --config PATH."
        )
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    if "defaults" not in cfg or "datasets" not in cfg:
        raise ValueError(
            f"{path}: missing required top-level table(s); need [defaults] and at "
            f"least one [datasets.NAME] table."
        )
    missing_defaults = [k for k in _REQUIRED_DEFAULTS_KEYS if k not in cfg["defaults"]]
    if missing_defaults:
        raise ValueError(
            f"{path}: [defaults] is missing required key(s): {missing_defaults}."
        )
    if "budgets" not in cfg["defaults"]:
        raise ValueError(
            f"{path}: missing [defaults.budgets] table — needed to look up per-mechanism "
            f"default budgets. Required mechanism keys: {list(_REQUIRED_BUDGET_KEYS)}."
        )
    missing_budgets = [k for k in _REQUIRED_BUDGET_KEYS if k not in cfg["defaults"]["budgets"]]
    if missing_budgets:
        raise ValueError(
            f"{path}: [defaults.budgets] is missing required mechanism(s): {missing_budgets}."
        )
    if not cfg["datasets"]:
        raise ValueError(f"{path}: no datasets declared — add at least one [datasets.NAME] table.")
    for name, entry in cfg["datasets"].items():
        missing = [k for k in _REQUIRED_DATASET_KEYS if k not in entry]
        if missing:
            raise ValueError(
                f"{path}: [datasets.{name}] is missing required key(s): {missing}. "
                f"Required: {list(_REQUIRED_DATASET_KEYS)}."
            )
    # solver_options is optional; default to empty dict for consistency downstream.
    cfg["defaults"].setdefault("solver_options", {})
    return cfg


_CONFIG: Dict[str, Any] = load_config()
DATASETS: Dict[str, Dict[str, Any]] = _CONFIG["datasets"]
_DEFAULTS: Dict[str, Any] = _CONFIG["defaults"]
# Path of the currently loaded config, used in validation error messages.
_LOADED_CONFIG_PATH: Path = _DEFAULT_CONFIG_PATH


def _reload_config(path: Optional[Path]) -> None:
    """Re-populate module-level DATASETS / _DEFAULTS from a different config path."""
    global _CONFIG, DATASETS, _DEFAULTS, _LOADED_CONFIG_PATH
    if path is None or Path(path).resolve() == _LOADED_CONFIG_PATH.resolve():
        return
    _CONFIG = load_config(path)
    DATASETS = _CONFIG["datasets"]
    _DEFAULTS = _CONFIG["defaults"]
    _LOADED_CONFIG_PATH = Path(path).resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Budget allocation strategies across tree levels
# ─────────────────────────────────────────────────────────────────────────────

def _alloc_equal(L: int, total: float) -> List[float]:
    return [total / L] * L


def _alloc_exp_leaves(L: int, total: float) -> List[float]:
    aux = sum(2 ** i for i in range(L))
    return [(total / aux) * (2 ** i) for i in range(L)]


def _alloc_exp_root(L: int, total: float) -> List[float]:
    aux = sum(2 ** i for i in range(L))
    return [(total / aux) * (2 ** (L - 1 - i)) for i in range(L)]


def _alloc_square_leaves(L: int, total: float) -> List[float]:
    # Weights 1, 4, 9, …; level 0 gets the smallest share.
    weights = [(i + 1) ** 2 for i in range(L)]
    s = sum(weights)
    return [total * w / s for w in weights]


def _alloc_inv_nodes_factory(nodes_per_level: List[int]) -> Callable[[int, float], List[float]]:
    """Return a strategy giving each level a budget proportional to 1/n_nodes_at_that_level.

    Smaller-node levels (root) get more budget per node; leaves get less. Built as a factory
    because nodes_per_level depends on the tree, so it can't be a pure static function.
    """
    def _alloc(L: int, total: float) -> List[float]:
        if len(nodes_per_level) != L:
            raise ValueError(
                f"inv_nodes strategy needs nodes_per_level of length {L}, got {len(nodes_per_level)}"
            )
        weights = [1.0 / max(1, n) for n in nodes_per_level]
        s = sum(weights)
        return [total * w / s for w in weights]
    return _alloc


ALLOCATIONS: Dict[str, Callable[[int, float], List[float]]] = {
    "equal": _alloc_equal,
    "exp_leaves": _alloc_exp_leaves,
    "exp_root": _alloc_exp_root,
    "square_leaves": _alloc_square_leaves,
    # `inv_nodes` is registered lazily via a factory: experiments that want it pass the
    # actual tree's nodes-per-level vector through build_mechanism().
}


# ─────────────────────────────────────────────────────────────────────────────
# Mechanism factory
# ─────────────────────────────────────────────────────────────────────────────

def build_mechanism(
    name: str,
    total_budget: float,
    n_levels: int,
    allocation: Union[str, Callable[[int, float], List[float]]] = "exp_leaves",
    delta: Optional[float] = 1e-10,
) -> PrivacyMechanism:
    """Instantiate a `PrivacyMechanism` from a string spec.

    Args:
        name:         One of "PureDP", "ZCDP", "ApproximateDP", "RenyiDP".
        total_budget: Sum across levels (ε for PureDP/RenyiDP; ρ for ZCDP/ApproximateDP).
        n_levels:     Number of tree levels (= 1 + len(hierarchy)).
        allocation:   Either a key from ALLOCATIONS or a callable (L, total) -> list[float].
        delta:        Required for ApproximateDP / RenyiDP; ignored otherwise.

    Returns:
        A configured PrivacyMechanism instance.
    """
    alloc_fn = ALLOCATIONS[allocation] if isinstance(allocation, str) else allocation
    per_level = alloc_fn(n_levels, total_budget)
    if name == "PureDP":
        return PureDP(per_level)
    if name == "ZCDP":
        return ZCDP(per_level)
    if name == "ApproximateDP":
        if delta is None:
            raise ValueError("ApproximateDP requires a delta.")
        return ApproximateDP(per_level, delta=delta)
    if name == "RenyiDP":
        if delta is None:
            raise ValueError("RenyiDP requires a delta.")
        return RenyiDP(per_level, delta=delta)
    raise ValueError(f"Unknown mechanism: {name!r}")


def epsilon_to_rho(epsilon: float, delta: float) -> float:
    """Inverse Bun-Steinke: smallest rho with rho + 2*sqrt(rho*log(1/delta)) <= epsilon.

    Numerically stable closed form, rho = epsilon^2 / (c + sqrt(c^2 + epsilon))^2 with
    c = sqrt(log(1/delta)). Inverse of ApproximateDP.equivalent_epsilon — use to calibrate
    a zCDP/ApproxDP run so it matches a target (epsilon, delta)-DP guarantee.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}.")
    c = math.sqrt(math.log(1.0 / delta))
    return epsilon * epsilon / (c + math.sqrt(c * c + epsilon)) ** 2


def rho_to_epsilon(rho: float, delta: float) -> float:
    """Bun-Steinke: equivalent epsilon of a rho-zCDP mechanism at the given delta."""
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}.")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}.")
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    # All defaults come from the config file (tests/config.toml by default). Direct
    # dataclass construction without going through cfg_from_args() is opt-in: callers
    # supply explicit values or call ExperimentConfig.from_config().
    dataset: str = ""
    process_until: str = ""
    queries: List[str] = field(default_factory=list)
    mechanism: str = ""
    total_budget: float = 0.0
    delta: float = 0.0
    allocation: str = ""
    trials: int = 0
    extension: str = ""
    out_dir: str = ""
    solver_name: str = ""
    solver_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, defaults: Optional[Dict[str, Any]] = None) -> "ExperimentConfig":
        """Build a config populated from the [defaults] table of the loaded TOML.

        `total_budget` is looked up from the per-mechanism [defaults.budgets] table
        so the default privacy guarantee is consistent across mechanisms.
        """
        d = defaults if defaults is not None else _DEFAULTS
        mech = d["mechanism"]
        return cls(
            dataset=d["dataset"],
            process_until=d["process_until"],
            queries=[],
            mechanism=mech,
            total_budget=float(d["budgets"][mech]),
            delta=float(d["delta"]),
            allocation=d["allocation"],
            trials=int(d["trials"]),
            extension=d["extension"],
            out_dir=str(_PROJECT_ROOT / "tests" / "out"),
            solver_name=d["solver_name"],
            solver_options=dict(d.get("solver_options", {})),
        )

    # ---------- derived ----------

    def resolve_hierarchy(self) -> List[str]:
        if self.dataset not in DATASETS:
            raise ValueError(
                f"dataset {self.dataset!r} not found in {_LOADED_CONFIG_PATH} "
                f"(defined: {sorted(DATASETS)})."
            )
        hierarchy = DATASETS[self.dataset]["hierarchy"]
        if self.process_until not in hierarchy:
            raise ValueError(
                f"process_until={self.process_until!r} not in hierarchy {hierarchy} "
                f"for dataset {self.dataset!r} (configured in {_LOADED_CONFIG_PATH})."
            )
        idx = hierarchy.index(self.process_until)
        return hierarchy[: idx + 1]

    def resolve_queries(self) -> List[str]:
        return list(self.queries) if self.queries else list(DATASETS[self.dataset]["default_queries"])

    def n_levels(self) -> int:
        return 1 + len(self.resolve_hierarchy())

    def data_path(self) -> str:
        return DATASETS[self.dataset]["path"]

    def sep(self) -> str:
        return DATASETS[self.dataset]["sep"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI helper — every experiment script accepts the same baseline flags.
# ─────────────────────────────────────────────────────────────────────────────

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Standard flags every experiment accepts. Defaults come from tests/config.toml.

    `default=None` (not the config value) on flags that need post-parse resolution against
    the chosen dataset, like --process_until. cfg_from_args fills them in from the config
    after both args.dataset and args.config have been observed.
    """
    parser.add_argument("--config", default=None, type=Path,
                        help=f"Path to TOML config. Defaults to {_DEFAULT_CONFIG_PATH}.")
    parser.add_argument("--dataset", default=None,
                        help=f"Dataset name; one of the [datasets.X] tables in --config "
                             f"(default: from config).")
    parser.add_argument("--process_until", default=None,
                        help="Deepest hierarchy column to process; must be one of the "
                             "selected dataset's `hierarchy` entries.")
    parser.add_argument("--queries", nargs="+", default=None,
                        help="Query columns. Default: dataset's `default_queries`.")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--mechanism", choices=["ZCDP", "PureDP", "ApproximateDP", "RenyiDP"],
                        default=None)
    parser.add_argument("--budget", type=float, default=None,
                        help="Total privacy budget (ε for PureDP/RenyiDP, ρ for ZCDP/ApproximateDP).")
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--allocation", choices=list(ALLOCATIONS), default=None)
    parser.add_argument("--ext", choices=["png", "pdf", "svg"], default=None,
                        help="Plot file extension. png is the lightest; pdf is best for LaTeX.")
    parser.add_argument("--out", default=None,
                        help="Output directory. Defaults to tests/out/<exp_name>.")


def cfg_from_args(args: argparse.Namespace, exp_name: str) -> ExperimentConfig:
    """Build an ExperimentConfig with the precedence: CLI > config file > built-in.

    If `--config` was supplied, re-loads the module-level config from that path before
    extracting defaults. Validates `dataset` against the registry and `process_until`
    against the chosen dataset's hierarchy, with errors that name the config file path.
    """
    _reload_config(getattr(args, "config", None))
    cfg = ExperimentConfig.from_config(_DEFAULTS)
    # Override config-provided defaults with anything the CLI actually supplied.
    if args.dataset is not None:        cfg.dataset = args.dataset
    if args.process_until is not None:  cfg.process_until = args.process_until
    if args.queries:                    cfg.queries = list(args.queries)
    if args.mechanism is not None:
        cfg.mechanism = args.mechanism
        # A different mechanism uses a different native unit (eps vs rho), so refresh
        # the default budget from the per-mechanism table. --budget below still wins.
        if cfg.mechanism in _DEFAULTS["budgets"]:
            cfg.total_budget = float(_DEFAULTS["budgets"][cfg.mechanism])
    if args.budget is not None:         cfg.total_budget = args.budget
    if args.delta is not None:          cfg.delta = args.delta
    if args.allocation is not None:     cfg.allocation = args.allocation
    if args.trials is not None:         cfg.trials = args.trials
    if args.ext is not None:            cfg.extension = args.ext
    cfg.out_dir = args.out or str(_PROJECT_ROOT / "tests" / "out" / exp_name)
    if cfg.dataset not in DATASETS:
        raise ValueError(
            f"dataset {cfg.dataset!r} not found in {_LOADED_CONFIG_PATH} "
            f"(defined: {sorted(DATASETS)})."
        )
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Workload builders
# ─────────────────────────────────────────────────────────────────────────────

def workload_identity(contingency_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    n = len(contingency_df)
    Q = np.eye(n, dtype=np.int64)
    names = [f"cell_{i}" for i in range(n)]
    return Q, names


def workload_marginals_k(contingency_df: pd.DataFrame, columns: List[str], k: int) -> Tuple[np.ndarray, List[str]]:
    """All k-way marginals across the given columns.

    Each generated query counts the cells matching one specific combination of one chosen
    k-subset of `columns`. Implemented on top of QueryWorkload.value_counts.
    """
    from itertools import combinations
    qw = QueryWorkload()
    for combo in combinations(columns, k):
        qw.value_counts(list(combo))
    Q = qw.build(contingency_df).astype(np.int64)
    return Q, list(qw.query_names)


def workload_random_binary(
    contingency_df: pd.DataFrame,
    n_queries: int,
    density: float,
    target_sensitivity: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Random binary workload with controlled row density and column-sum cap.

    `density` is the per-row Bernoulli probability of including a cell. If `target_sensitivity`
    is set, columns that would exceed it have excess 1s thinned out — the result has L1
    sensitivity ≤ `target_sensitivity`.
    """
    rng = rng or np.random.default_rng(0)
    n_cells = len(contingency_df)
    Q = (rng.random((n_queries, n_cells)) < density).astype(np.int64)
    # Cells with no representation in Q leave LP variables unconstrained, which makes the
    # solver leave them uninitialised — repair by injecting at least one 1 per cell.
    empty_cols = np.where(Q.sum(axis=0) == 0)[0]
    for j in empty_cols:
        row = int(rng.integers(0, n_queries))
        Q[row, j] = 1
    if target_sensitivity is not None:
        col_sums = Q.sum(axis=0)
        for j in np.where(col_sums > target_sensitivity)[0]:
            ones = np.where(Q[:, j] == 1)[0]
            drop = rng.choice(ones, size=col_sums[j] - target_sensitivity, replace=False)
            Q[drop, j] = 0
    names = [f"rand_{i}" for i in range(n_queries)]
    return Q, names


def workload_mixed(
    contingency_df: pd.DataFrame,
    columns: List[str],
    n_random: int = 20,
    density: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[str]]:
    """1-way + 2-way marginals concatenated with random binary queries."""
    Q1, n1 = workload_marginals_k(contingency_df, columns, 1)
    if len(columns) >= 2:
        Q2, n2 = workload_marginals_k(contingency_df, columns, 2)
    else:
        Q2 = np.zeros((0, Q1.shape[1]), dtype=np.int64)
        n2 = []
    Qr, nr = workload_random_binary(contingency_df, n_random, density, rng=rng)
    Q = np.vstack([Q1, Q2, Qr])
    return Q, n1 + n2 + nr


def resolve_workload(spec: str, contingency_df: pd.DataFrame, query_columns: List[str],
                     rng: Optional[np.random.Generator] = None,
                     n_random: int = 20, density: float = 0.1) -> Tuple[np.ndarray, List[str]]:
    """String-to-workload dispatcher used by all experiment scripts.

    Supported specs: identity, marginals1, marginals2, marginals3, random, mixed.
    """
    if spec == "identity":
        return workload_identity(contingency_df)
    if spec.startswith("marginals"):
        k = int(spec.replace("marginals", ""))
        if k > len(query_columns):
            raise ValueError(f"marginals{k} requires at least {k} query columns; have {len(query_columns)}.")
        return workload_marginals_k(contingency_df, query_columns, k)
    if spec == "random":
        return workload_random_binary(contingency_df, n_random, density, rng=rng)
    if spec == "mixed":
        return workload_mixed(contingency_df, query_columns, n_random=n_random, density=density, rng=rng)
    raise ValueError(f"Unknown workload spec: {spec!r}")


def add_workload_args(parser: argparse.ArgumentParser, default: str = "identity") -> None:
    """Standard --workload / --n_random_queries / --random_density flags.

    Every experiment uses the same triple, so define it once here. `default` lets a script
    pick its own baseline (e.g., exp5 doesn't have a single workload).
    """
    parser.add_argument("--workload", default=default,
                        help="Workload spec: identity, marginals1, marginals2, ..., random, mixed.")
    parser.add_argument("--n_random_queries", type=int, default=20,
                        help="Number of queries when --workload is 'random' or 'mixed'.")
    parser.add_argument("--random_density", type=float, default=0.1,
                        help="Per-cell Bernoulli density for 'random' / 'mixed' workloads.")


def build_workload(cfg: "ExperimentConfig", args: argparse.Namespace,
                   verbose: bool = True) -> Tuple[np.ndarray, List[str], pd.DataFrame, int]:
    """Build Q for the (cfg, --workload) pair. Returns (Q, names, contingency_df, sensitivity).

    Centralizes the DataHandler dance every experiment does before run_once. Random
    workloads draw a fresh entropy seed each call — measurement-phase noise is non-
    reproducible (OpenDP samplers don't accept seeds), so we don't pretend the workload is.
    """
    from data_handler import DataHandler
    dh = DataHandler(file_path=str((_PROJECT_ROOT / cfg.data_path()).resolve()))
    dh.hierarchical_columns = cfg.resolve_hierarchy()
    dh.query_columns = cfg.resolve_queries()
    dh.read_data(cfg.resolve_hierarchy() + cfg.resolve_queries(), sep=cfg.sep())
    contingency_df = dh.generate_contingency_dataframe(cfg.resolve_queries())
    rng = np.random.default_rng()
    Q, names = resolve_workload(
        args.workload, contingency_df, cfg.resolve_queries(),
        rng=rng, n_random=args.n_random_queries, density=args.random_density,
    )
    sensitivity = int(Q.sum(axis=0).max())
    if verbose:
        print(f"Workload {args.workload!r}: Q.shape={Q.shape}, sensitivity={sensitivity}")
    return Q, names, contingency_df, sensitivity


# ─────────────────────────────────────────────────────────────────────────────
# Truth pass (Q = Identity) — cached on disk per (dataset, hierarchy, queries).
# ─────────────────────────────────────────────────────────────────────────────

_TRUTH_CACHE_DIR = _PROJECT_ROOT / "tests" / "out" / "_truth_cache"


def _truth_cache_key(cfg: ExperimentConfig) -> str:
    h = hashlib.sha1()
    payload = {
        "dataset": cfg.dataset,
        "hierarchy": cfg.resolve_hierarchy(),
        "queries": cfg.resolve_queries(),
        "process_until": cfg.process_until,
    }
    h.update(json.dumps(payload, sort_keys=True).encode())
    return h.hexdigest()[:16]


def truth_tree(cfg: ExperimentConfig, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Build (or load from cache) the per-node true cell counts x, plus node metadata.

    Runs a one-shot TopDown.initialize() with Q = Identity, captures the per-node x vectors
    and the level array, releases shared memory, and persists the result so subsequent
    experiments on the same (dataset, hierarchy, queries) reuse it.

    Returns a dict with keys:
        x_truth: np.ndarray  shape (n_nodes, n_cells), integer cell counts per node
        levels:  np.ndarray  shape (n_nodes,),         tree level of each node
        contingency_columns: list[str] (mirrors cfg.resolve_queries())
        n_cells: int
        n_nodes: int
        levels_start_idx: list[int] from tree._levels (node index where each level starts)
    """
    _TRUTH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _truth_cache_key(cfg)
    cache_file = _TRUTH_CACHE_DIR / f"{key}.npz"
    if cache_file.exists():
        if verbose:
            print(f"[truth_cache] hit  ({cache_file.name})")
        data = np.load(cache_file, allow_pickle=True)
        return {
            "x_truth": data["x_truth"],
            "levels": data["levels"],
            "contingency_columns": list(data["contingency_columns"]),
            "n_cells": int(data["n_cells"]),
            "n_nodes": int(data["n_nodes"]),
            "levels_start_idx": list(data["levels_start_idx"]),
        }
    if verbose:
        print(f"[truth_cache] miss → building Q=I truth tree …")
    hierarchy = cfg.resolve_hierarchy()
    queries = cfg.resolve_queries()
    n_levels = 1 + len(hierarchy)
    # Use a placeholder mechanism — we will not call measurement_phase, so any valid mechanism works.
    placeholder_mech = ZCDP([1.0] * n_levels)
    td = TopDown(
        data_path=str(_PROJECT_ROOT / cfg.data_path()),
        hierarchy=hierarchy,
        query_columns=queries,
        privacy_mechanism=placeholder_mech,
        out_path=str(_TRUTH_CACHE_DIR / "noisy_data.csv"),  # never written
        optimizer=cfg.solver_name,
        solver_options=cfg.solver_options,
    )
    # Override the sep used by data_handler.read_data via direct call. TopDown.initialize()
    # hardcodes sep=';' which matches both census CSVs, but we keep this explicit for clarity.
    td.initialize()
    n_cells = td.tree.n_cells
    # Q=I (default when no workload is set), so the first n_cells slots hold x.
    x_truth = td.tree._contingency_vectors[:, :n_cells].copy()
    levels = np.array([n.level for n in td.tree.nodes], dtype=np.int32)
    levels_start_idx = list(td.tree._levels)
    contingency_columns = list(td.data_handler.contingency_df.columns)
    _release_shared_memory(td)
    np.savez_compressed(
        cache_file,
        x_truth=x_truth,
        levels=levels,
        contingency_columns=np.array(contingency_columns),
        n_cells=n_cells,
        n_nodes=x_truth.shape[0],
        levels_start_idx=np.array(levels_start_idx, dtype=np.int64),
    )
    if verbose:
        print(f"[truth_cache] saved → {cache_file.name}  "
              f"({x_truth.shape[0]} nodes, n_cells={n_cells})")
    return {
        "x_truth": x_truth,
        "levels": levels,
        "contingency_columns": contingency_columns,
        "n_cells": n_cells,
        "n_nodes": x_truth.shape[0],
        "levels_start_idx": levels_start_idx,
    }


def _release_shared_memory(td: TopDown) -> None:
    """Best-effort cleanup of the shared-memory block backing the tree."""
    try:
        shm = td.tree._contingency_vectors_shm
        if shm is not None:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Single-run helper: runs init / measurement / estimation for one configuration.
# ─────────────────────────────────────────────────────────────────────────────

def run_once(
    cfg: ExperimentConfig,
    Q: Optional[np.ndarray] = None,
    mechanism: Optional[PrivacyMechanism] = None,
    trial_index: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one privatized TopDown pass and return the artefacts needed for metrics.

    Captures the noise-free `y_truth = Q @ x` right after initialize(), then runs the
    measurement and estimation phases. Microdata construction is skipped (slow + not
    needed for metrics). Shared memory is released before returning.

    Args:
        cfg:           Experiment config.
        Q:             Optional pre-built (n_queries, n_cells) workload matrix; identity used if None.
        mechanism:     Optional pre-built PrivacyMechanism; built from cfg if None.
        trial_index:   Used for tagging only — OpenDP RNG is not seedable.
        verbose:       If True, allow TopDown's progress prints; otherwise silence stdout.

    Returns a dict with:
        x_hat:    np.ndarray (n_nodes, n_cells), integer x_hat per node
        y_truth:  np.ndarray (n_nodes, n_queries), Q @ x  (noise-free measurements)
        y_noisy:  np.ndarray (n_nodes, n_queries), Q @ x + integer noise
        y_hat:    np.ndarray (n_nodes, n_queries), Q @ x_hat
        levels:   np.ndarray (n_nodes,), tree level per node
        levels_start_idx: list[int] (from tree._levels)
        timings:  dict {init, measure, estimate, total}
        Q:        the actual workload matrix used
        n_cells:  int
        n_queries:int
        sensitivity: int
        mechanism_report: str (privacy_mechanism.report_guarantee())
        trial_index: int
    """
    hierarchy = cfg.resolve_hierarchy()
    queries = cfg.resolve_queries()
    n_levels = 1 + len(hierarchy)
    mech = mechanism if mechanism is not None else build_mechanism(
        cfg.mechanism, cfg.total_budget, n_levels, cfg.allocation, cfg.delta
    )
    td = TopDown(
        data_path=str(_PROJECT_ROOT / cfg.data_path()),
        hierarchy=hierarchy,
        query_columns=queries,
        privacy_mechanism=mech,
        out_path=str(Path(cfg.out_dir) / "_unused_noisy.csv"),
        optimizer=cfg.solver_name,
        solver_options=cfg.solver_options,
    )
    if Q is not None:
        td.set_query_workload(Q)

    # ---- phase timings ----
    with _silenced(stdout=not verbose):
        t_init0 = time.time()
        td.initialize()
        t_init = time.time() - t_init0

        n_cells = td.tree.n_cells
        n_queries = td.tree.n_queries
        # Snapshot truth (Q @ x); subsequent phases overwrite this slot.
        y_truth = td.tree._contingency_vectors[:, :n_queries].copy()

        t_meas0 = time.time()
        td.measurement_phase()
        t_meas = time.time() - t_meas0

        y_noisy = td.tree._contingency_vectors[:, :n_queries].copy()

        t_est0 = time.time()
        td.estimation_phase()
        t_est = time.time() - t_est0

        x_hat = td.tree._contingency_vectors[:, :n_cells].astype(np.int64).copy()

    Q_actual = td.Q
    levels = np.array([n.level for n in td.tree.nodes], dtype=np.int32)
    levels_start_idx = list(td.tree._levels)
    sensitivity = td.query_sensitivity
    mech_report = td.privacy_mechanism.report_guarantee()
    _release_shared_memory(td)

    # Compute y_hat = Q @ x_hat for each node (vectorized).
    y_hat = (Q_actual @ x_hat.T).T.astype(np.int64)

    return {
        "x_hat": x_hat,
        "y_truth": y_truth,
        "y_noisy": y_noisy,
        "y_hat": y_hat,
        "levels": levels,
        "levels_start_idx": levels_start_idx,
        "timings": {"init": t_init, "measure": t_meas, "estimate": t_est, "total": t_init + t_meas + t_est},
        "Q": Q_actual,
        "n_cells": n_cells,
        "n_queries": n_queries,
        "sensitivity": sensitivity,
        "mechanism_report": mech_report,
        "trial_index": trial_index,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum()))


def tvd(truth: np.ndarray, est: np.ndarray) -> float:
    """Total Variation Distance between two count distributions on the same support."""
    total = max(1, int(truth.sum()))
    return 0.5 * np.abs(truth - est).sum() / total


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean()) if a.size else 0.0


def mre(truth: np.ndarray, est: np.ndarray, floor: int = 1) -> float:
    """Mean Relative Error, with denominator floored at `floor` to avoid div-by-zero."""
    denom = np.maximum(truth.astype(float), float(floor))
    return float((np.abs(est - truth) / denom).mean())


def mape(truth: np.ndarray, est: np.ndarray, floor: int = 1) -> float:
    return 100.0 * mre(truth, est, floor=floor)


def metrics_by_level(
    truth_matrix: np.ndarray,
    est_matrix: np.ndarray,
    levels: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Compute per-level mean metrics. truth_matrix / est_matrix are (n_nodes, n_features)."""
    out: Dict[int, Dict[str, float]] = {}
    unique_levels = np.unique(levels)
    for lvl in unique_levels:
        idx = np.where(levels == lvl)[0]
        t = truth_matrix[idx]
        e = est_matrix[idx]
        out[int(lvl)] = {
            "L1_mean": float(np.abs(t - e).sum(axis=1).mean()),
            "L2_mean": float(np.sqrt(((t - e) ** 2).sum(axis=1)).mean()),
            "MAE_mean": float(np.abs(t - e).mean()),
            "TVD_mean": float(np.mean([tvd(t[i], e[i]) for i in range(t.shape[0])])),
            "MAPE_mean": float(np.mean([mape(t[i], e[i]) for i in range(t.shape[0])])),
            "n_nodes": int(t.shape[0]),
        }
    return out


def query_errors(
    y_truth: np.ndarray,
    y_hat: np.ndarray,
    levels: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Per-level, per-query MAPE arrays. Returns dict[level -> array shape (n_queries,)]."""
    out: Dict[int, np.ndarray] = {}
    unique_levels = np.unique(levels)
    for lvl in unique_levels:
        idx = np.where(levels == lvl)[0]
        t = y_truth[idx].astype(float)
        e = y_hat[idx].astype(float)
        denom = np.maximum(t, 1.0)
        per_query = (np.abs(e - t) / denom).mean(axis=0) * 100.0
        out[int(lvl)] = per_query
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers and persistence
# ─────────────────────────────────────────────────────────────────────────────

def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    plt.rcParams.update({
        "axes.titleweight": "bold",
        "axes.labelweight": "regular",
        "figure.autolayout": True,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })


def save_fig(fig: plt.Figure, out_dir: Union[str, Path], name: str, ext: str = "png",
             footer: Optional[str] = None) -> Path:
    """Save `fig` and optionally stamp a one-line footer at the bottom (config summary)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if footer:
        # Make room for the footer without clipping autolayout content.
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.04)
        fig.text(0.5, 0.005, footer, ha="center", va="bottom",
                 fontsize=8, color="dimgray", alpha=0.9)
    path = out / f"{name}.{ext}"
    fig.savefig(path)
    plt.close(fig)
    return path


def experiment_footer(cfg: "ExperimentConfig", extra: str = "") -> str:
    """Build a one-line footer summarizing the run: dataset, queries, mechanism, budget."""
    hierarchy = " > ".join(cfg.resolve_hierarchy())
    parts = [
        f"Dataset: {cfg.dataset}",
        f"Hierarchy: {hierarchy}",
        f"Queries: {', '.join(cfg.resolve_queries())}",
        f"Mechanism: {cfg.mechanism}",
        f"Budget: {cfg.total_budget}",
        f"Allocation: {cfg.allocation}",
        f"Trials: {cfg.trials}",
    ]
    if extra:
        parts.append(extra)
    return "  |  ".join(parts)


def dump_results(out_dir: Union[str, Path], name: str, rows: List[Dict[str, Any]],
                 cfg: ExperimentConfig, extras: Optional[Dict[str, Any]] = None) -> Tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{name}.csv"
    json_path = out / f"{name}_config.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        payload: Dict[str, Any] = {"config": asdict(cfg)}
        if extras:
            payload["extras"] = extras
        json.dump(payload, f, indent=2, default=str)
    return csv_path, json_path


# ─────────────────────────────────────────────────────────────────────────────
# Misc utilities
# ─────────────────────────────────────────────────────────────────────────────

class _silenced:
    """Context manager that optionally redirects stdout/stderr to /dev/null."""
    def __init__(self, stdout: bool = True, stderr: bool = False):
        self._sout, self._serr = stdout, stderr
        self._dn = None
        self._orig: Tuple[Any, Any] = (None, None)

    def __enter__(self):
        if not (self._sout or self._serr):
            return self
        self._dn = open(os.devnull, "w")
        self._orig = (sys.stdout, sys.stderr)
        if self._sout:
            sys.stdout = self._dn
        if self._serr:
            sys.stderr = self._dn
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._dn is None:
            return False
        sys.stdout, sys.stderr = self._orig
        self._dn.close()
        return False


def nodes_per_level_from_tree(levels_start_idx: List[int], n_nodes: int) -> List[int]:
    """Convert tree._levels (start index of each level) into a per-level node count vector."""
    out = []
    for i in range(len(levels_start_idx)):
        start = levels_start_idx[i]
        end = levels_start_idx[i + 1] if i + 1 < len(levels_start_idx) else n_nodes
        out.append(end - start)
    return out


set_plot_style()
