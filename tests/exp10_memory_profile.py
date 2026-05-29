"""Experiment 10 — Memory usage of the TopDown pipeline.

Three sub-tests (select with --tests; default runs all three):

  matrix   Peak memory as the query matrix Q grows in number of *queries* (rows of Q),
           with the contingency domain held fixed. The memory analogue of exp9.
  cells    Peak memory as the per-node contingency vector grows. The knob is the number
           of *query columns*: we sweep a prefix of a query-column pool, so n_cells
           (= size of the Cartesian product) grows. Q is held at a fixed modest row count
           so the growth reflects the contingency vectors, not Q.
  profile  Per-phase memory profiling of a single run: initialize / measurement /
           estimation / construct_microdata. Reports each phase's peak RSS increase and,
           via tracemalloc, the top Python allocators (filename:lineno) per phase.

How memory is measured
----------------------
Primary metric = **peak RSS of the process tree** (this process + its recursive children),
sampled by a background thread (`MemorySampler`). RSS is the only metric that captures what
dominates here: numpy data buffers, the multiprocessing.shared_memory block backing
`tree._contingency_vectors`, and the estimation worker subprocesses spawned by
`subtree_estimation_phase`. It works on Windows, where `resource.getrusage` does not.

The `profile` test additionally uses `tracemalloc` for Python-level attribution. Caveat:
tracemalloc only sees the current process, so the estimation workers' memory shows up in the
RSS delta but NOT in the tracemalloc allocator table — the two views are complementary.

This module does not modify any pipeline file; it wraps the pipeline from the outside exactly
as `common.run_once` and `exp8._build_and_run` do.

Outputs (in --out, default tests/out/exp10_memory_profile/):
  - peak_rss_vs_n_queries.{ext}      (matrix)   + exp10_matrix.csv
  - peak_rss_vs_n_cells.{ext}        (cells)    + exp10_cells.csv
  - phase_memory_breakdown.{ext}     (profile)  + exp10_phase.csv
  - phase_rss_timeline.{ext}         (profile)  + exp10_phase_timeline.csv
  - exp10_phase_allocators.csv       (profile)
  - exp10_memory_profile_config.json
"""
from __future__ import annotations

import argparse
import gc
import json
import threading
import time
import tracemalloc
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tests.common import (
    DATASETS,
    _PROJECT_ROOT,
    _release_shared_memory,
    _silenced,
    add_common_args,
    add_workload_args,
    build_mechanism,
    cfg_from_args,
    experiment_footer,
    resolve_workload,
    run_once,
    save_fig,
    set_plot_style,
    workload_random_binary,
)

from topdown import TopDown

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only when the dep is absent
    psutil = None


_BYTES_PER_MB = 1024 * 1024

_PHASE_COLORS = {
    "initialize": "tab:blue",
    "measurement": "tab:orange",
    "estimation": "tab:green",
    "construct": "tab:red",
}


def _require_psutil() -> None:
    if psutil is None:
        raise SystemExit(
            "exp10 needs psutil for RSS-based memory measurement, but it is not installed.\n"
            "Install it into your environment with:\n\n    pip install psutil\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Peak-RSS sampler — background thread polling the whole process tree.
# ─────────────────────────────────────────────────────────────────────────────

class MemorySampler:
    """Context manager that tracks peak RSS of this process + its recursive children.

    A daemon thread polls every `interval_ms` and keeps the running max. `peak`/`delta`
    are bytes; `peak_mb`/`delta_mb` are MiB. Set `record_series=True` to retain a small
    (t, rss) trace for the timeline plot; otherwise only scalars are kept so nothing large
    is retained (the project's memory-efficiency constraint).
    """

    def __init__(self, interval_ms: float = 25.0, record_series: bool = False):
        self.interval = interval_ms / 1000.0
        self.record_series = record_series
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.baseline = 0
        self.peak = 0
        self.delta = 0
        self.series: List[Tuple[float, int]] = []
        self._t0 = 0.0

    def _tree_rss(self) -> int:
        total = 0
        try:
            total = self._proc.memory_info().rss
            for child in self._proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # workers come and go during estimation
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._tree_rss()
            if rss > self.peak:
                self.peak = rss
            if self.record_series:
                self.series.append((time.time() - self._t0, rss))
            self._stop.wait(self.interval)

    def __enter__(self) -> "MemorySampler":
        gc.collect()
        self._t0 = time.time()
        self.baseline = self._tree_rss()
        self.peak = self.baseline
        if self.record_series:
            self.series.append((0.0, self.baseline))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        rss = self._tree_rss()  # one last sample to catch a late peak
        if rss > self.peak:
            self.peak = rss
        self.delta = self.peak - self.baseline
        return False

    @property
    def peak_mb(self) -> float:
        return self.peak / _BYTES_PER_MB

    @property
    def delta_mb(self) -> float:
        return self.delta / _BYTES_PER_MB

    @property
    def baseline_mb(self) -> float:
        return self.baseline / _BYTES_PER_MB


def _build_contingency(cfg, queries: List[str]) -> pd.DataFrame:
    """Build the contingency DataFrame for an explicit query-column list (mirrors exp9)."""
    from data_handler import DataHandler
    hierarchy = cfg.resolve_hierarchy()
    dh = DataHandler(file_path=str((_PROJECT_ROOT / cfg.data_path()).resolve()))
    dh.hierarchical_columns = hierarchy
    dh.query_columns = queries
    dh.read_data(hierarchy + queries, sep=cfg.sep())
    return dh.generate_contingency_dataframe(queries)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — peak memory vs number of queries (rows of Q), n_cells fixed.
# ─────────────────────────────────────────────────────────────────────────────

def _run_matrix_sweep(cfg, n_queries_sweep: List[int], density: float, trials: int,
                      interval_ms: float) -> Tuple[List[dict], int]:
    contingency_df = _build_contingency(cfg, cfg.resolve_queries())
    n_cells = len(contingency_df)
    rows: List[dict] = []
    for n_q in n_queries_sweep:
        rng = np.random.default_rng()
        Q, _names = workload_random_binary(contingency_df, n_q, density, rng=rng)
        sensitivity = int(Q.sum(axis=0).max())
        print(f"\n[matrix] n_queries={n_q}  n_cells={n_cells}  Δ={sensitivity}")
        for t in range(trials):
            with MemorySampler(interval_ms=interval_ms) as ms:
                run_once(cfg, Q=Q, trial_index=t, verbose=False)
            rows.append({
                "n_queries": int(n_q),
                "n_cells": int(n_cells),
                "matrix_cells": int(n_q * n_cells),
                "sensitivity": sensitivity,
                "trial": t,
                "peak_mb": ms.peak_mb,
                "delta_mb": ms.delta_mb,
                "baseline_mb": ms.baseline_mb,
            })
            print(f"  trial {t}: peak={ms.peak_mb:.1f} MB  (Δ over baseline={ms.delta_mb:.1f} MB)")
    return rows, n_cells


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — peak memory vs n_cells (grown by adding query columns), Q rows fixed.
# ─────────────────────────────────────────────────────────────────────────────

def _run_cells_sweep(cfg, pool: List[str], cells_n_queries: int, density: float,
                     trials: int, interval_ms: float) -> List[dict]:
    if len(pool) < 3:
        print(f"[cells] WARNING: query pool {pool} has < 3 columns, so the sweep has only "
              f"{len(pool)} point(s). Pass --cells_query_pool with more columns "
              f"(e.g. P01 P02 P03A P03B) for a meaningful curve.")
    rows: List[dict] = []
    for k in range(1, len(pool) + 1):
        queries = list(pool[:k])
        sub_cfg = replace(cfg, queries=queries)
        contingency_df = _build_contingency(sub_cfg, queries)
        n_cells = len(contingency_df)
        rng = np.random.default_rng()
        Q, _names = workload_random_binary(contingency_df, cells_n_queries, density, rng=rng)
        sensitivity = int(Q.sum(axis=0).max())
        print(f"\n[cells] k={k}  queries={queries}  n_cells={n_cells}  "
              f"n_queries={cells_n_queries}  Δ={sensitivity}")
        for t in range(trials):
            with MemorySampler(interval_ms=interval_ms) as ms:
                run_once(sub_cfg, Q=Q, trial_index=t, verbose=False)
            rows.append({
                "k": k,
                "queries": ",".join(queries),
                "n_cells": int(n_cells),
                "n_queries": int(cells_n_queries),
                "sensitivity": sensitivity,
                "trial": t,
                "peak_mb": ms.peak_mb,
                "delta_mb": ms.delta_mb,
                "baseline_mb": ms.baseline_mb,
            })
            print(f"  trial {t}: peak={ms.peak_mb:.1f} MB  (Δ over baseline={ms.delta_mb:.1f} MB)")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — per-phase memory profile of one run.
# ─────────────────────────────────────────────────────────────────────────────

def _run_phase_profile(cfg, Q: Optional[np.ndarray], interval_ms: float,
                       top_n: int) -> Tuple[List[dict], List[dict], List[tuple]]:
    hierarchy = cfg.resolve_hierarchy()
    queries = cfg.resolve_queries()
    n_levels = 1 + len(hierarchy)
    mech = build_mechanism(cfg.mechanism, cfg.total_budget, n_levels, cfg.allocation, cfg.delta)
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

    phases = [
        ("initialize", td.initialize),
        ("measurement", td.measurement_phase),
        ("estimation", td.estimation_phase),
        ("construct", lambda: td.data_handler.construct_microdata(td.tree)),
    ]

    phase_rows: List[dict] = []
    alloc_rows: List[dict] = []
    timeline: List[tuple] = []
    t_offset = 0.0

    tracemalloc.start()
    try:
        with _silenced(stdout=True):
            for name, fn in phases:
                snap_before = tracemalloc.take_snapshot()
                with MemorySampler(interval_ms=interval_ms, record_series=True) as ms:
                    t0 = time.time()
                    fn()
                    elapsed = time.time() - t0
                snap_after = tracemalloc.take_snapshot()

                for (tt, rss) in ms.series:
                    timeline.append((name, t_offset + tt, rss / _BYTES_PER_MB))
                t_offset += elapsed

                phase_rows.append({
                    "phase": name,
                    "time_s": elapsed,
                    "peak_mb": ms.peak_mb,
                    "delta_mb": ms.delta_mb,
                    "baseline_mb": ms.baseline_mb,
                })

                stats = sorted(
                    snap_after.compare_to(snap_before, "lineno"),
                    key=lambda s: s.size_diff, reverse=True,
                )
                for st in stats[:top_n]:
                    frame = st.traceback[0]
                    alloc_rows.append({
                        "phase": name,
                        "location": f"{frame.filename}:{frame.lineno}",
                        "size_kb": st.size_diff / 1024.0,
                        "count": st.count_diff,
                    })
    finally:
        tracemalloc.stop()
        _release_shared_memory(td)

    return phase_rows, alloc_rows, timeline


def _print_allocators(alloc_rows: List[dict], top_n: int) -> None:
    if not alloc_rows:
        return
    print("\nTop Python allocators per phase (tracemalloc — main process only; estimation "
          "worker-process memory is NOT included here, but it is in the RSS delta above):")
    df = pd.DataFrame(alloc_rows)
    for phase in df["phase"].unique():
        sub = df[df["phase"] == phase].head(top_n)
        print(f"\n  [{phase}]")
        for r in sub.itertuples():
            print(f"    {r.size_kb:10.1f} KB  x{r.count:<7d}  {r.location}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _agg_peak(df: pd.DataFrame, xcol: str) -> pd.DataFrame:
    out = df.groupby(xcol)["peak_mb"].agg(["mean", "std"]).reset_index()
    return out.sort_values(xcol)


def _plot_scaling(df: pd.DataFrame, xcol: str, xlabel: str, title: str,
                  out_dir: Path, ext: str, name: str, footer: str = "") -> None:
    agg = _agg_peak(df, xcol)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(agg[xcol], agg["mean"], "-o", color="tab:purple", label="Peak RSS")
    if (agg["std"].fillna(0) > 0).any():
        ax.fill_between(agg[xcol], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        alpha=0.15, color="tab:purple")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Peak RSS (MB, process tree)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    save_fig(fig, out_dir, name, ext, footer=footer)


def _plot_phase_breakdown(phase_df: pd.DataFrame, out_dir: Path, ext: str,
                          footer: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [_PHASE_COLORS.get(p, "tab:gray") for p in phase_df["phase"]]
    ax.bar(phase_df["phase"], phase_df["delta_mb"], color=colors)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Peak RSS increase over phase baseline (MB)")
    ax.set_title("Per-phase memory (peak RSS delta)")
    save_fig(fig, out_dir, "phase_memory_breakdown", ext, footer=footer)


def _plot_timeline(timeline_df: pd.DataFrame, out_dir: Path, ext: str,
                   footer: str = "") -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(timeline_df["t"], timeline_df["mb"], "-", color="dimgray", lw=1.2, zorder=3)
    ymax = float(timeline_df["mb"].max())
    for phase, sub in timeline_df.groupby("phase", sort=False):
        color = _PHASE_COLORS.get(phase, "gray")
        ax.axvspan(sub["t"].min(), sub["t"].max(), color=color, alpha=0.12)
        ax.text(sub["t"].mean(), ymax * 1.01, phase, ha="center", va="bottom",
                fontsize=8, color=color)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RSS (MB, process tree)")
    ax.set_title("Process-tree RSS over the run")
    save_fig(fig, out_dir, "phase_rss_timeline", ext, footer=footer)


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Memory usage of the TopDown pipeline.")
    add_common_args(parser)
    add_workload_args(parser, default="random")  # used by the 'profile' test's Q
    parser.add_argument("--tests", nargs="+", choices=["matrix", "cells", "profile"],
                        default=["matrix", "cells", "profile"],
                        help="Which sub-tests to run (default: all three).")
    parser.add_argument("--n_queries_sweep", type=int, nargs="+",
                        default=[10, 50, 100, 250, 500, 1000],
                        help="[matrix] n_queries values to test (rows of Q).")
    parser.add_argument("--cells_query_pool", nargs="+", default=None,
                        help="[cells] query columns to sweep prefixes of. Default: the "
                             "dataset's resolved queries (pass several for a real curve).")
    parser.add_argument("--cells_n_queries", type=int, default=50,
                        help="[cells] fixed number of rows of Q while n_cells grows.")
    parser.add_argument("--sample_interval_ms", type=float, default=25.0,
                        help="RSS sampling interval for the background sampler.")
    parser.add_argument("--top_allocators", type=int, default=15,
                        help="[profile] number of top tracemalloc sites to keep per phase.")
    args = parser.parse_args()

    _require_psutil()
    set_plot_style()
    cfg = cfg_from_args(args, "exp10_memory_profile")
    # Keep sweeps fast by default: 2 tree levels (root + the dataset's root-most column).
    # Push deeper with --process_until for the per-phase profile.
    if args.process_until is None:
        cfg.process_until = DATASETS[cfg.dataset]["hierarchy"][0]
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {cfg.dataset}  |  hierarchy depth: {cfg.process_until} "
          f"({cfg.n_levels()} tree levels)  |  queries: {cfg.resolve_queries()}")
    print(f"Mechanism: {cfg.mechanism}  budget={cfg.total_budget}  |  tests: {args.tests}")
    print(f"RSS sampler interval: {args.sample_interval_ms} ms\n")

    base_footer = experiment_footer(cfg)
    extras: Dict[str, Any] = {"tests": args.tests, "sample_interval_ms": args.sample_interval_ms}

    if "matrix" in args.tests:
        rows, n_cells = _run_matrix_sweep(cfg, args.n_queries_sweep, args.random_density,
                                          cfg.trials, args.sample_interval_ms)
        pd.DataFrame(rows).to_csv(out_dir / "exp10_matrix.csv", index=False)
        extras["matrix"] = {"n_queries_sweep": args.n_queries_sweep,
                            "random_density": args.random_density, "n_cells": n_cells}
        footer = experiment_footer(cfg, extra=(
            f"Workload: random (density={args.random_density}) | n_cells={n_cells} | "
            f"n_queries sweep: {args.n_queries_sweep}"))
        _plot_scaling(pd.DataFrame(rows), "n_queries",
                      f"n_queries  (rows of Q; n_cells={n_cells} held fixed)",
                      "Peak memory vs query-matrix size", out_dir, cfg.extension,
                      "peak_rss_vs_n_queries", footer=footer)

    if "cells" in args.tests:
        pool = args.cells_query_pool if args.cells_query_pool else cfg.resolve_queries()
        rows = _run_cells_sweep(cfg, pool, args.cells_n_queries, args.random_density,
                                cfg.trials, args.sample_interval_ms)
        pd.DataFrame(rows).to_csv(out_dir / "exp10_cells.csv", index=False)
        extras["cells"] = {"query_pool": pool, "cells_n_queries": args.cells_n_queries,
                           "random_density": args.random_density}
        footer = experiment_footer(cfg, extra=(
            f"Workload: random ({args.cells_n_queries} rows, density={args.random_density}) | "
            f"query pool: {pool}"))
        _plot_scaling(pd.DataFrame(rows), "n_cells",
                      "n_cells  (contingency vector length; Q rows held fixed)",
                      "Peak memory vs contingency-vector size", out_dir, cfg.extension,
                      "peak_rss_vs_n_cells", footer=footer)

    if "profile" in args.tests:
        prof_queries = cfg.resolve_queries()
        contingency_df = _build_contingency(cfg, prof_queries)
        rng = np.random.default_rng()
        Q, _names = resolve_workload(args.workload, contingency_df, prof_queries, rng=rng,
                                     n_random=args.n_random_queries, density=args.random_density)
        sensitivity = int(Q.sum(axis=0).max())
        print(f"\n[profile] workload={args.workload!r}  Q.shape={Q.shape}  Δ={sensitivity}  "
              f"(this run constructs microdata, so it is the slowest test)")
        phase_rows, alloc_rows, timeline = _run_phase_profile(
            cfg, Q, args.sample_interval_ms, args.top_allocators)

        pd.DataFrame(phase_rows).to_csv(out_dir / "exp10_phase.csv", index=False)
        pd.DataFrame(alloc_rows).to_csv(out_dir / "exp10_phase_allocators.csv", index=False)
        timeline_df = pd.DataFrame(timeline, columns=["phase", "t", "mb"])
        timeline_df.to_csv(out_dir / "exp10_phase_timeline.csv", index=False)
        extras["profile"] = {"workload": args.workload, "Q_shape": list(Q.shape),
                             "sensitivity": sensitivity, "top_allocators": args.top_allocators}

        print("\nPer-phase peak RSS:")
        for r in phase_rows:
            print(f"  {r['phase']:<12s} peak={r['peak_mb']:8.1f} MB  "
                  f"Δ={r['delta_mb']:8.1f} MB  time={r['time_s']:6.2f} s")
        _print_allocators(alloc_rows, args.top_allocators)

        footer = experiment_footer(cfg, extra=(
            f"Workload: {args.workload} (Q.shape={tuple(Q.shape)}, Δ={sensitivity})"))
        _plot_phase_breakdown(pd.DataFrame(phase_rows), out_dir, cfg.extension, footer=footer)
        _plot_timeline(timeline_df, out_dir, cfg.extension, footer=footer)

    with open(out_dir / "exp10_memory_profile_config.json", "w") as f:
        json.dump({"config": asdict(cfg), "extras": extras}, f, indent=2, default=str)

    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()
