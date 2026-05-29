"""Experiment 6 — Budget allocation strategies across tree levels.

For a fixed total budget and workload Q, compares several strategies for splitting that
budget across tree levels:

  equal           — same share at every level
  exp_leaves      — 2^i, more to deeper levels (current TopDown default)
  exp_root        — 2^(L-1-i), more to the root
  square_leaves   — i^2
  inv_nodes       — proportional to 1/n_nodes_at_level (smaller levels get more per node)

For each strategy: per-level TVD curves + summary bar of global query MAPE. The per-level
budget vector is annotated in the plot legend.

Outputs (in --out, default tests/out/exp6_budget_allocation/):
  - tvd_per_level_by_strategy.{ext}
  - mape_by_strategy.{ext}
  - exp6_budget_allocation.csv
  - exp6_budget_allocation_config.json
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tests.common import (
    ALLOCATIONS,
    _alloc_inv_nodes_factory,
    add_common_args,
    build_mechanism,
    cfg_from_args,
    dump_results,
    experiment_footer,
    mape,
    metrics_by_level,
    nodes_per_level_from_tree,
    resolve_workload,
    run_once,
    save_fig,
    set_plot_style,
    truth_tree,
)


def _allocation_for(name: str, n_levels: int, total: float, nodes_per_level: List[int]) -> List[float]:
    if name == "inv_nodes":
        return _alloc_inv_nodes_factory(nodes_per_level)(n_levels, total)
    return ALLOCATIONS[name](n_levels, total)


def _run(cfg, Q, truth, strategies: List[str], nodes_per_level: List[int]) -> List[dict]:
    rows = []
    n_levels = cfg.n_levels()
    for strat in strategies:
        per_level_budget = _allocation_for(strat, n_levels, cfg.total_budget, nodes_per_level)
        print(f"\n[strategy {strat!r}]  per-level={['%.3f' % b for b in per_level_budget]}")
        for t in range(cfg.trials):
            mech = build_mechanism(
                cfg.mechanism, cfg.total_budget, n_levels,
                allocation=lambda L, total, b=per_level_budget: b,
                delta=cfg.delta,
            )
            sub_cfg = replace(cfg, allocation="custom")  # purely informational; alloc fn is passed via mech
            res = run_once(sub_cfg, Q=Q, mechanism=mech, trial_index=t, verbose=False)
            per_level = metrics_by_level(truth["x_truth"], res["x_hat"], res["levels"])
            global_mape = float(np.mean([
                mape(res["y_truth"][i], res["y_hat"][i]) for i in range(res["y_truth"].shape[0])
            ]))
            for lvl, m in per_level.items():
                rows.append({
                    "strategy": strat,
                    "trial": t,
                    "level": int(lvl),
                    "TVD": m["TVD_mean"],
                    "MAE": m["MAE_mean"],
                    "global_MAPE": global_mape,
                    "budget_at_level": per_level_budget[int(lvl)],
                })
    return rows


def _plot_tvd_per_level(df: pd.DataFrame, out_dir: Path, ext: str, total_budget: float, mechanism: str,
                        footer: str = "") -> None:
    agg = df.groupby(["strategy", "level"])["TVD"].agg(["mean", "std"]).reset_index()
    agg["std"] = agg["std"].fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_colors=len(agg["strategy"].unique()))
    for i, strat in enumerate(sorted(agg["strategy"].unique())):
        sub = agg[agg["strategy"] == strat].sort_values("level")
        ax.plot(sub["level"], sub["mean"], "-o", label=strat, color=palette[i])
        if (sub["std"] > 0).any():
            ax.fill_between(
                sub["level"],
                sub["mean"] - sub["std"],
                sub["mean"] + sub["std"],
                alpha=0.12, color=palette[i],
            )
    ax.set_xlabel("Tree level (0 = root)")
    ax.set_ylabel("Mean TVD  (0 – 1)")
    ax.set_title(f"Per-level TVD by allocation strategy  (total={total_budget}, {mechanism})")
    ax.legend(title="Strategy", fontsize=9)
    save_fig(fig, out_dir, "tvd_per_level_by_strategy", ext, footer=footer)


def _plot_mape_summary(df: pd.DataFrame, out_dir: Path, ext: str, footer: str = "") -> None:
    agg = df.groupby(["strategy"])["global_MAPE"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(agg) + 3), 5))
    xs = np.arange(len(agg))
    ax.bar(xs, agg["mean"], yerr=agg["std"].fillna(0), capsize=4, color="tab:purple", alpha=0.85)
    ax.set_xticks(xs); ax.set_xticklabels(agg["strategy"])
    ax.set_ylabel("Global query MAPE (%)")
    ax.set_title("Global query MAPE by allocation strategy")
    save_fig(fig, out_dir, "mape_by_strategy", ext, footer=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare budget-allocation strategies.")
    add_common_args(parser)
    parser.add_argument("--workload", default="identity")
    parser.add_argument("--n_random_queries", type=int, default=20)
    parser.add_argument("--random_density", type=float, default=0.1)
    parser.add_argument("--strategies", nargs="+",
                        default=["equal", "exp_leaves", "exp_root", "square_leaves",
                                 "inv_nodes"])
    args = parser.parse_args()

    set_plot_style()
    cfg = cfg_from_args(args, "exp6_budget_allocation")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = truth_tree(cfg, verbose=True)
    n_nodes = int(truth["n_nodes"])
    nodes_per_level = nodes_per_level_from_tree(list(truth["levels_start_idx"]), n_nodes)
    print(f"Nodes per level: {nodes_per_level}")

    from data_handler import DataHandler
    from pathlib import Path as _P
    dh = DataHandler(file_path=str(_P(cfg.data_path()).resolve()))
    dh.hierarchical_columns = cfg.resolve_hierarchy()
    dh.query_columns = cfg.resolve_queries()
    dh.read_data(cfg.resolve_hierarchy() + cfg.resolve_queries(), sep=cfg.sep())
    contingency_df = dh.generate_contingency_dataframe(cfg.resolve_queries())

    rng = np.random.default_rng()
    Q, _names = resolve_workload(
        args.workload, contingency_df, cfg.resolve_queries(),
        rng=rng, n_random=args.n_random_queries, density=args.random_density,
    )
    print(f"\nWorkload {args.workload!r}: Q.shape={Q.shape}, sensitivity={int(Q.sum(axis=0).max())}")

    rows = _run(cfg, Q, truth, args.strategies, nodes_per_level)
    csv_path, json_path = dump_results(out_dir, "exp6_budget_allocation", rows, cfg, extras={
        "workload": args.workload,
        "strategies": args.strategies,
        "nodes_per_level": nodes_per_level,
    })
    print(f"\nRaw results → {csv_path}\nConfig     → {json_path}")

    df = pd.DataFrame(rows)
    footer = experiment_footer(cfg, extra=(
        f"Workload: {args.workload} | Q.shape={Q.shape}, Δ={int(Q.sum(axis=0).max())} | "
        f"Strategies: {','.join(args.strategies)} | Nodes/level: {nodes_per_level}"
    ))
    _plot_tvd_per_level(df, out_dir, cfg.extension, cfg.total_budget, cfg.mechanism, footer=footer)
    _plot_mape_summary(df, out_dir, cfg.extension, footer=footer)
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
