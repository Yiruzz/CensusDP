"""Experiment 9 — Runtime vs query-matrix size.

Holds the query columns (so n_cells is constant) and sweeps the number of rows of Q via
a random binary workload. Reports init / measurement / estimation / total wall-clock in
seconds, plus standard deviation across --trials.

By default it runs at 2 tree levels (root + the top of the configured dataset's hierarchy)
so the matrix-size effect is isolated from tree-traversal cost. Pass --process_until to go
deeper.

Outputs (in --out, default tests/out/exp9_matrix_size/):
  - runtime_vs_matrix_size.{ext}   — per-phase + total time vs n_queries
  - phase_breakdown.{ext}          — stacked bar of phases per n_queries
  - exp9_matrix_size.csv
  - exp9_matrix_size_config.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tests.common import (
    DATASETS,
    _PROJECT_ROOT,
    add_common_args,
    cfg_from_args,
    dump_results,
    experiment_footer,
    run_once,
    save_fig,
    set_plot_style,
    workload_random_binary,
)


_PHASES = [
    ("init", "Init", "tab:blue"),
    ("measure", "Measurement", "tab:orange"),
    ("estimate", "Estimation", "tab:green"),
    ("total", "Total", "black"),
]


def _build_contingency(cfg):
    """Build the contingency DataFrame once — fixed across the n_queries sweep."""
    from data_handler import DataHandler
    dh = DataHandler(file_path=str((_PROJECT_ROOT / cfg.data_path()).resolve()))
    dh.hierarchical_columns = cfg.resolve_hierarchy()
    dh.query_columns = cfg.resolve_queries()
    dh.read_data(cfg.resolve_hierarchy() + cfg.resolve_queries(), sep=cfg.sep())
    return dh.generate_contingency_dataframe(cfg.resolve_queries())


def _run_sweep(cfg, contingency_df, n_queries_sweep: List[int],
               density: float, trials: int) -> List[dict]:
    rows = []
    n_cells = len(contingency_df)
    for n_q in n_queries_sweep:
        rng = np.random.default_rng()
        Q, _names = workload_random_binary(contingency_df, n_q, density, rng=rng)
        sensitivity = int(Q.sum(axis=0).max())
        print(f"\n[matrix-size sweep] n_queries={n_q}  n_cells={n_cells}  Δ={sensitivity}")
        for t in range(trials):
            res = run_once(cfg, Q=Q, trial_index=t, verbose=False)
            timings = res["timings"]
            rows.append({
                "n_queries": int(n_q),
                "n_cells": int(n_cells),
                "matrix_size": int(n_q * n_cells),
                "sensitivity": sensitivity,
                "trial": t,
                **timings,
            })
            print(f"  trial {t}: init={timings['init']:.2f}s  meas={timings['measure']:.2f}s  "
                  f"est={timings['estimate']:.2f}s  total={timings['total']:.2f}s")
    return rows


def _agg_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["n_queries", "n_cells", "sensitivity"])
    out = g[["init", "measure", "estimate", "total"]].agg(["mean", "std"]).reset_index()
    out.columns = [c if isinstance(c, str) else "_".join([str(x) for x in c if x])
                   for c in out.columns]
    return out.sort_values("n_queries")


def _plot_runtime(df: pd.DataFrame, out_dir: Path, ext: str, footer: str = "") -> None:
    agg = _agg_mean_std(df)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for col, label, color in _PHASES:
        ax.plot(agg["n_queries"], agg[f"{col}_mean"], "-o", label=label, color=color)
        if (agg[f"{col}_std"] > 0).any():
            ax.fill_between(
                agg["n_queries"],
                agg[f"{col}_mean"] - agg[f"{col}_std"],
                agg[f"{col}_mean"] + agg[f"{col}_std"],
                alpha=0.15, color=color,
            )
    n_cells = int(agg["n_cells"].iloc[0])
    ax.set_xlabel(f"n_queries  (rows of Q; n_cells={n_cells} held fixed)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Runtime vs query-matrix size")
    ax.legend(loc="upper left", fontsize=9)
    save_fig(fig, out_dir, "runtime_vs_matrix_size", ext, footer=footer)


def _plot_stacked(df: pd.DataFrame, out_dir: Path, ext: str, footer: str = "") -> None:
    agg = _agg_mean_std(df)
    labels = [f"{int(r.n_queries)}" for r in agg.itertuples()]
    init = agg["init_mean"].values
    meas = agg["measure_mean"].values
    est = agg["estimate_mean"].values
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(labels) + 2), 5.5))
    ax.bar(labels, init, label="Init", color="tab:blue")
    ax.bar(labels, meas, bottom=init, label="Measurement", color="tab:orange")
    ax.bar(labels, est, bottom=init + meas, label="Estimation", color="tab:green")
    ax.set_xlabel("n_queries (rows of Q)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Phase breakdown vs query-matrix size")
    ax.legend()
    save_fig(fig, out_dir, "phase_breakdown", ext, footer=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime vs query-matrix size.")
    add_common_args(parser)
    parser.add_argument("--n_queries_sweep", type=int, nargs="+",
                        default=[10, 50, 100, 250, 500, 1000],
                        help="List of n_queries values to test (rows of Q).")
    parser.add_argument("--random_density", type=float, default=0.1,
                        help="Per-cell Bernoulli density of the random binary Q.")
    args = parser.parse_args()

    set_plot_style()
    cfg = cfg_from_args(args, "exp9_matrix_size")
    # Isolate matrix-size cost from tree-traversal cost: if --process_until wasn't given,
    # use the dataset's root-most hierarchy column (2 tree levels: root + that column).
    if args.process_until is None:
        cfg.process_until = DATASETS[cfg.dataset]["hierarchy"][0]
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contingency_df = _build_contingency(cfg)
    print(f"Contingency size: n_cells = {len(contingency_df)}  "
          f"(query columns: {cfg.resolve_queries()})")
    print(f"Hierarchy depth: {cfg.process_until}  ({cfg.n_levels()} tree levels)")
    print(f"Workload: random binary, density={args.random_density}, sweep={args.n_queries_sweep}")

    rows = _run_sweep(cfg, contingency_df, args.n_queries_sweep, args.random_density, cfg.trials)

    csv_path, json_path = dump_results(out_dir, "exp9_matrix_size", rows, cfg, extras={
        "workload": "random",
        "random_density": args.random_density,
        "n_queries_sweep": args.n_queries_sweep,
        "n_cells": int(len(contingency_df)),
    })
    print(f"\nRaw results → {csv_path}\nConfig     → {json_path}")

    df = pd.DataFrame(rows)
    footer = experiment_footer(cfg, extra=(
        f"Workload: random (density={args.random_density}) | "
        f"n_cells={len(contingency_df)} | n_queries sweep: {args.n_queries_sweep}"
    ))
    _plot_runtime(df, out_dir, cfg.extension, footer=footer)
    _plot_stacked(df, out_dir, cfg.extension, footer=footer)
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
