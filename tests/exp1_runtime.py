"""Experiment 1 — Runtime scaling.

Measures wall-clock for the three main phases (init, measurement, estimation) under two
sweeps:

  (a) Query-vector size: progressively adds query columns from --query_progression so the
      contingency-vector length grows from 2 (one binary column) up to the full product.
      Tree depth is fixed at --process_until.

  (b) Tree depth: fixes the query set and varies --depth_sweep across hierarchical levels
      (REGION → … → ZC_LOC). The contingency-vector length is constant.

For each (config, trial) we record per-phase seconds plus the total. Mean ± std is reported
across --trials.

Outputs (in --out, default tests/out/exp1_runtime/):
  - runtime_vs_size.{ext}        — total + per-phase vs n_cells, linear and log axes
  - runtime_vs_depth.{ext}       — total + per-phase vs hierarchy level
  - stacked_phases.{ext}         — stacked bar per query-size config
  - exp1_runtime.csv             — raw rows
  - exp1_runtime_config.json     — config snapshot
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
    add_common_args,
    add_workload_args,
    build_workload,
    cfg_from_args,
    dump_results,
    experiment_footer,
    run_once,
    save_fig,
    set_plot_style,
)


def _run_sweep_size(cfg_base, args, query_progression: List[str], trials: int) -> List[dict]:
    rows = []
    for k in range(1, len(query_progression) + 1):
        queries_now = query_progression[:k]
        cfg = _clone_cfg(cfg_base, queries=queries_now)
        print(f"\n[size sweep] queries={queries_now}  (process_until={cfg.process_until})")
        try:
            Q, _names, _cdf, sens = build_workload(cfg, args, verbose=False)
        except ValueError as e:
            print(f"  [skip] workload {args.workload!r} not buildable for {queries_now}: {e}")
            continue
        n_cells_observed = None
        for t in range(trials):
            res = run_once(cfg, Q=Q, trial_index=t, verbose=False)
            n_cells_observed = res["n_cells"]
            timings = res["timings"]
            rows.append({
                "sweep": "size",
                "queries": ",".join(queries_now),
                "n_query_cols": k,
                "process_until": cfg.process_until,
                "n_cells": res["n_cells"],
                "n_queries": res["n_queries"],
                "n_nodes": res["x_hat"].shape[0],
                "sensitivity": sens,
                "workload": args.workload,
                "trial": t,
                **timings,
            })
            print(f"  trial {t}: init={timings['init']:.2f}s  meas={timings['measure']:.2f}s  "
                  f"est={timings['estimate']:.2f}s  total={timings['total']:.2f}s")
        print(f"  → n_cells={n_cells_observed}, n_queries={Q.shape[0]}, Δ={sens}")
    return rows


def _run_sweep_depth(cfg_base, args, depth_sweep: List[str], trials: int) -> List[dict]:
    rows = []
    for proc_until in depth_sweep:
        cfg = _clone_cfg(cfg_base, process_until=proc_until)
        print(f"\n[depth sweep] process_until={proc_until}  (queries={cfg.resolve_queries()})")
        try:
            Q, _names, _cdf, sens = build_workload(cfg, args, verbose=False)
        except ValueError as e:
            print(f"  [skip] workload {args.workload!r} not buildable at {proc_until}: {e}")
            continue
        for t in range(trials):
            res = run_once(cfg, Q=Q, trial_index=t, verbose=False)
            timings = res["timings"]
            rows.append({
                "sweep": "depth",
                "queries": ",".join(cfg.resolve_queries()),
                "process_until": proc_until,
                "n_levels": cfg.n_levels(),
                "n_cells": res["n_cells"],
                "n_queries": res["n_queries"],
                "n_nodes": res["x_hat"].shape[0],
                "sensitivity": sens,
                "workload": args.workload,
                "trial": t,
                **timings,
            })
            print(f"  trial {t}: init={timings['init']:.2f}s  meas={timings['measure']:.2f}s  "
                  f"est={timings['estimate']:.2f}s  total={timings['total']:.2f}s")
    return rows


def _clone_cfg(cfg, **overrides):
    from dataclasses import replace
    return replace(cfg, **overrides)


def _agg_mean_std(df: pd.DataFrame, group_keys: List[str], value_cols: List[str]) -> pd.DataFrame:
    g = df.groupby(group_keys)
    out = g[value_cols].agg(["mean", "std"]).reset_index()
    out.columns = [c if isinstance(c, str) else "_".join([str(x) for x in c if x])
                   for c in out.columns]
    return out


def _plot_size(rows: List[dict], out_dir: Path, ext: str, footer: str = "") -> None:
    df = pd.DataFrame([r for r in rows if r["sweep"] == "size"])
    if df.empty:
        return
    agg = _agg_mean_std(df, ["n_query_cols", "n_cells"], ["init", "measure", "estimate", "total"])
    agg = agg.sort_values("n_cells")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, scale in zip(axes, ["linear", "log"]):
        for col, label, color in [
            ("init", "Init", "tab:blue"),
            ("measure", "Measurement", "tab:orange"),
            ("estimate", "Estimation", "tab:green"),
            ("total", "Total", "black"),
        ]:
            ax.plot(agg["n_cells"], agg[f"{col}_mean"], "-o", label=label, color=color)
            if (agg[f"{col}_std"] > 0).any():
                ax.fill_between(
                    agg["n_cells"],
                    agg[f"{col}_mean"] - agg[f"{col}_std"],
                    agg[f"{col}_mean"] + agg[f"{col}_std"],
                    alpha=0.15, color=color,
                )
        ax.set_xlabel("Contingency-vector size  (n_cells)")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Runtime vs query size — {scale} scale")
        ax.set_xscale(scale); ax.set_yscale(scale)
    axes[0].legend(loc="upper left", fontsize=9)
    save_fig(fig, out_dir, "runtime_vs_size", ext, footer=footer)


def _plot_depth(rows: List[dict], out_dir: Path, ext: str, hierarchy_order: List[str],
                footer: str = "") -> None:
    df = pd.DataFrame([r for r in rows if r["sweep"] == "depth"])
    if df.empty:
        return
    df["depth_order"] = df["process_until"].apply(lambda x: hierarchy_order.index(x))
    agg = _agg_mean_std(df, ["process_until", "depth_order", "n_levels"],
                        ["init", "measure", "estimate", "total"])
    agg = agg.sort_values("depth_order")

    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label, color in [
        ("init", "Init", "tab:blue"),
        ("measure", "Measurement", "tab:orange"),
        ("estimate", "Estimation", "tab:green"),
        ("total", "Total", "black"),
    ]:
        ax.plot(agg["process_until"], agg[f"{col}_mean"], "-o", label=label, color=color)
        if (agg[f"{col}_std"] > 0).any():
            ax.fill_between(
                agg["process_until"],
                agg[f"{col}_mean"] - agg[f"{col}_std"],
                agg[f"{col}_mean"] + agg[f"{col}_std"],
                alpha=0.15, color=color,
            )
    ax.set_xlabel("Deepest hierarchy level processed")
    ax.set_ylabel("Time (s)")
    ax.set_title("Runtime vs hierarchy depth")
    ax.legend(loc="upper left", fontsize=9)
    save_fig(fig, out_dir, "runtime_vs_depth", ext, footer=footer)


def _plot_stacked(rows: List[dict], out_dir: Path, ext: str, footer: str = "") -> None:
    df = pd.DataFrame([r for r in rows if r["sweep"] == "size"])
    if df.empty:
        return
    agg = _agg_mean_std(df, ["n_query_cols", "queries", "n_cells"],
                        ["init", "measure", "estimate"])
    agg = agg.sort_values("n_query_cols")
    labels = [f"{int(r.n_query_cols)} cols\n({int(r.n_cells)})" for r in agg.itertuples()]
    init = agg["init_mean"].values
    meas = agg["measure_mean"].values
    est  = agg["estimate_mean"].values
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels)), 5))
    ax.bar(labels, init, label="Init", color="tab:blue")
    ax.bar(labels, meas, bottom=init, label="Measurement", color="tab:orange")
    ax.bar(labels, est, bottom=init + meas, label="Estimation", color="tab:green")
    ax.set_ylabel("Time (s)")
    ax.set_title("Phase breakdown vs query size")
    ax.legend()
    save_fig(fig, out_dir, "stacked_phases", ext, footer=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime scaling experiment.")
    add_common_args(parser)
    add_workload_args(parser)
    parser.add_argument("--query_progression", nargs="+", default=None,
                        help="Ordered list of query columns; experiment runs cumulative prefixes.")
    parser.add_argument("--depth_sweep", nargs="+", default=None,
                        help="Hierarchy levels to sweep as process_until.")
    args = parser.parse_args()

    set_plot_style()
    cfg = cfg_from_args(args, "exp1_runtime")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hierarchy = DATASETS[cfg.dataset]["hierarchy"]
    query_progression = args.query_progression or DATASETS[cfg.dataset]["default_queries"]
    depth_sweep = args.depth_sweep or [hierarchy[0], cfg.process_until]
    # Remove duplicates, keep order
    seen = set()
    depth_sweep = [x for x in depth_sweep if (x in hierarchy) and not (x in seen or seen.add(x))]

    print(f"\nWorkload: {args.workload}  (n_random={args.n_random_queries}, density={args.random_density})")

    rows: List[dict] = []
    if query_progression:
        rows += _run_sweep_size(cfg, args, list(query_progression), cfg.trials)
    if depth_sweep:
        rows += _run_sweep_depth(cfg, args, depth_sweep, cfg.trials)

    csv_path, json_path = dump_results(out_dir, "exp1_runtime", rows, cfg, extras={
        "query_progression": list(query_progression),
        "depth_sweep": depth_sweep,
        "workload": args.workload,
        "n_random_queries": args.n_random_queries,
        "random_density": args.random_density,
    })
    print(f"\nRaw results → {csv_path}")
    print(f"Config     → {json_path}")

    footer = experiment_footer(cfg, extra=(
        f"Workload: {args.workload} | Query progression: {','.join(query_progression)} | "
        f"Depth sweep: {','.join(depth_sweep)}"
    ))
    _plot_size(rows, out_dir, cfg.extension, footer=footer)
    _plot_depth(rows, out_dir, cfg.extension, hierarchy, footer=footer)
    _plot_stacked(rows, out_dir, cfg.extension, footer=footer)
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
