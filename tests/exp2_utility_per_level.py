"""Experiment 2 — Per-level data-distribution distance to truth.

For a fixed workload Q, this experiment runs the privatized pipeline `--trials` times and
computes, for each tree level, the per-node Total Variation Distance (TVD), L1, and Mean
Absolute Error (MAE) between the noise-free cell-count vector x and the optimizer's
estimate x_hat. Results are averaged over nodes of each level then over trials.

Outputs (in --out, default tests/out/exp2_utility_per_level/):
  - tvd_per_level.{ext}        — mean ± std TVD per tree level
  - mae_per_level.{ext}        — same for MAE
  - l1_per_level.{ext}         — same for L1
  - exp2_per_level.csv         — raw per-trial rows
  - exp2_per_level_config.json — config snapshot
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tests.common import (
    add_common_args,
    cfg_from_args,
    dump_results,
    experiment_footer,
    metrics_by_level,
    resolve_workload,
    run_once,
    save_fig,
    set_plot_style,
    truth_tree,
)


def _per_node_truth_for_queries(truth: Dict, Q: np.ndarray) -> np.ndarray:
    """Compute y_truth = Q @ x_truth per node from cached x_truth.
    Currently unused (run_once already returns y_truth), but kept for symmetry.
    """
    return (Q @ truth["x_truth"].T).T


def _run_trials(cfg, Q: np.ndarray, truth: Dict, query_names: List[str]) -> List[dict]:
    rows = []
    for t in range(cfg.trials):
        print(f"\n--- trial {t+1}/{cfg.trials} ---")
        res = run_once(cfg, Q=Q, trial_index=t, verbose=False)
        levels = res["levels"]
        x_truth = truth["x_truth"]
        if x_truth.shape != res["x_hat"].shape:
            raise RuntimeError(
                f"shape mismatch: x_truth {x_truth.shape} vs x_hat {res['x_hat'].shape}. "
                "Likely a stale truth cache — delete tests/out/_truth_cache/ and retry."
            )
        per_level = metrics_by_level(x_truth, res["x_hat"], levels)
        for lvl, m in per_level.items():
            rows.append({
                "trial": t,
                "level": int(lvl),
                "n_nodes_at_level": m["n_nodes"],
                "L1": m["L1_mean"],
                "L2": m["L2_mean"],
                "MAE": m["MAE_mean"],
                "TVD": m["TVD_mean"],
                "MAPE": m["MAPE_mean"],
                **{f"time_{k}": v for k, v in res["timings"].items()},
                "n_queries": res["n_queries"],
                "n_cells": res["n_cells"],
                "sensitivity": res["sensitivity"],
                "mechanism": res["mechanism_report"],
            })
            print(f"  level {lvl}: TVD={m['TVD_mean']:.4f}  MAE={m['MAE_mean']:.4f}  "
                  f"L1={m['L1_mean']:.1f}  nodes={m['n_nodes']}")
    return rows


def _plot_metric_per_level(df: pd.DataFrame, metric: str, out_dir: Path, ext: str, ylabel: str, title: str,
                           footer: str = "") -> None:
    agg = df.groupby(["level"])[metric].agg(["mean", "std", "count"]).reset_index()
    mean = agg["mean"]
    std = agg["std"].fillna(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(agg["level"], mean, "-o", color="tab:blue", label=f"mean {metric}")
    if (std > 0).any():
        ax.fill_between(agg["level"], mean - std, mean + std,
                        alpha=0.2, color="tab:blue", label="± std")
    ax.set_xlabel("Tree level (0 = root)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(agg["level"])
    ax.legend()
    save_fig(fig, out_dir, f"{metric.lower()}_per_level", ext, footer=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-level distribution distance experiment.")
    add_common_args(parser)
    parser.add_argument("--workload", default="identity",
                        help="Workload spec: identity | marginals1 | marginals2 | random | mixed.")
    parser.add_argument("--n_random_queries", type=int, default=20)
    parser.add_argument("--random_density", type=float, default=0.1)
    args = parser.parse_args()

    set_plot_style()
    cfg = cfg_from_args(args, "exp2_utility_per_level")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Truth pass first; this builds the cache if missing.
    truth = truth_tree(cfg, verbose=True)

    # Materialize the workload using the same contingency_df as the truth (their columns must align).
    # We re-derive the contingency_df from the truth columns by re-running generate_contingency_dataframe
    # is expensive — instead, we reuse the workload helpers by passing the contingency frame from a fresh
    # DataHandler. Simpler: build a one-off TopDown.initialize() with the desired Q later in run_once.
    # To pre-build Q here we need the contingency_df; easiest path is a tiny helper that loads it
    # without running noise/estimation.
    from data_handler import DataHandler
    from pathlib import Path as _P
    dh = DataHandler(file_path=str(_P(cfg.data_path()).resolve()))
    dh.hierarchical_columns = cfg.resolve_hierarchy()
    dh.query_columns = cfg.resolve_queries()
    dh.read_data(cfg.resolve_hierarchy() + cfg.resolve_queries(), sep=cfg.sep())
    contingency_df = dh.generate_contingency_dataframe(cfg.resolve_queries())

    rng = np.random.default_rng()
    Q, names = resolve_workload(
        args.workload, contingency_df, cfg.resolve_queries(),
        rng=rng, n_random=args.n_random_queries, density=args.random_density,
    )
    print(f"\nWorkload {args.workload!r}: Q.shape={Q.shape}, sensitivity={int(Q.sum(axis=0).max())}")

    rows = _run_trials(cfg, Q, truth, names)

    csv_path, json_path = dump_results(out_dir, "exp2_per_level", rows, cfg, extras={
        "workload": args.workload,
        "n_random_queries": args.n_random_queries,
        "random_density": args.random_density,
        "Q_shape": list(Q.shape),
        "sensitivity": int(Q.sum(axis=0).max()),
    })
    print(f"\nRaw results → {csv_path}\nConfig     → {json_path}")

    df = pd.DataFrame(rows)
    footer = experiment_footer(cfg, extra=f"Workload: {args.workload} | Q.shape={Q.shape}, Δ={int(Q.sum(axis=0).max())}")
    _plot_metric_per_level(df, "TVD", out_dir, cfg.extension,
                           ylabel="Mean TVD  (0 – 1)",
                           title=f"TVD per tree level — {args.workload}",
                           footer=footer)
    _plot_metric_per_level(df, "MAE", out_dir, cfg.extension,
                           ylabel="Mean Absolute Error", title=f"MAE per tree level — {args.workload}",
                           footer=footer)
    _plot_metric_per_level(df, "L1", out_dir, cfg.extension,
                           ylabel="Mean L1 distance", title=f"L1 per tree level — {args.workload}",
                           footer=footer)
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
