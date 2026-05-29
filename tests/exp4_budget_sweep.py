"""Experiment 4 — Utility vs total privacy budget at iso-(epsilon, delta).

Sweeps `--budgets` (default 0.1, 0.5, 1, 2, 5, 10, 20) as the target epsilon and reports:
  * Leaf-level TVD (mean across leaves, mean across trials)
  * Overall query MAPE — `Q @ x_hat` vs `Q @ x` averaged across all nodes and queries

Every mechanism in `--mechanisms` is calibrated to the SAME (target_epsilon, delta)-DP
guarantee at each sweep point:

  PureDP / RenyiDP : epsilon = target_eps directly
  ZCDP / ApproximateDP : rho = inverse Bun-Steinke(target_eps, delta)

so the x-axis is a single, shared epsilon and the curves are directly comparable. This
makes the PureDP-vs-Gaussian crossover visible: PureDP wins at low sensitivity / loose
epsilon; Gaussian mechanisms win as Δ grows or epsilon tightens.

Outputs (in --out, default tests/out/exp4_budget_sweep/):
  - utility_vs_budget.{ext}     — log-x utility curves
  - exp4_budget_sweep.csv
  - exp4_budget_sweep_config.json
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from tests.common import (
    add_common_args,
    add_workload_args,
    build_workload,
    cfg_from_args,
    dump_results,
    epsilon_to_rho,
    experiment_footer,
    mape,
    metrics_by_level,
    run_once,
    save_fig,
    set_plot_style,
    truth_tree,
)


def _iso_budget_for(mechanism: str, target_eps: float, delta: float) -> float:
    if mechanism in ("PureDP", "RenyiDP"):
        return target_eps
    if mechanism in ("ZCDP", "ApproximateDP"):
        return epsilon_to_rho(target_eps, delta)
    raise ValueError(f"Unknown mechanism: {mechanism!r}")


def _run(cfg, Q, truth, mechanism: str, target_epsilons: List[float]) -> List[dict]:
    rows = []
    for target_eps in target_epsilons:
        b = _iso_budget_for(mechanism, target_eps, cfg.delta)
        sub_cfg = replace(cfg, mechanism=mechanism, total_budget=b)
        unit = "epsilon" if mechanism in ("PureDP", "RenyiDP") else "rho"
        print(f"\n[{mechanism}]  target_eps={target_eps}  -> {unit}={b:.4g}")
        for t in range(cfg.trials):
            res = run_once(sub_cfg, Q=Q, trial_index=t, verbose=False)
            per_level = metrics_by_level(truth["x_truth"], res["x_hat"], res["levels"])
            leaf_lvl = int(res["levels"].max())
            leaf_tvd = per_level[leaf_lvl]["TVD_mean"]
            global_mape = float(np.mean([
                mape(res["y_truth"][i], res["y_hat"][i]) for i in range(res["y_truth"].shape[0])
            ]))
            rows.append({
                "mechanism": mechanism,
                "target_epsilon": float(target_eps),
                "per_mechanism_budget": float(b),
                "per_mechanism_unit": unit,
                "trial": t,
                "leaf_TVD": leaf_tvd,
                "global_MAPE": global_mape,
                "n_queries": res["n_queries"],
                "n_cells": res["n_cells"],
                "mechanism_report": res["mechanism_report"],
            })
            print(f"   trial {t}: leaf_TVD={leaf_tvd:.4f}  global_MAPE={global_mape:.2f}%")
    return rows


def _fmt_eps(v: float) -> str:
    """Compact label for a target epsilon: integers as '10', fractions as '0.5' / '2.5'."""
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _plot(df: pd.DataFrame, out_dir: Path, ext: str, delta: float, footer: str = "") -> None:
    epsilons = sorted(df["target_epsilon"].unique())
    tick_labels = [_fmt_eps(e) for e in epsilons]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, ylabel in zip(
        axes,
        ["leaf_TVD", "global_MAPE"],
        ["Leaf-level TVD",
         "Global query MAPE (%)"],
    ):
        agg = df.groupby(["mechanism", "target_epsilon"])[metric].agg(["mean", "std"]).reset_index()
        agg["std"] = agg["std"].fillna(0)
        for mech, sub in agg.groupby("mechanism"):
            sub = sub.sort_values("target_epsilon")
            ax.plot(sub["target_epsilon"], sub["mean"], "-o", label=mech)
            if (sub["std"] > 0).any():
                ax.fill_between(
                    sub["target_epsilon"],
                    sub["mean"] - sub["std"],
                    sub["mean"] + sub["std"],
                    alpha=0.15,
                )
        # Linear x-axis so the visual gap matches the actual numeric distance — readers
        # asked to see the *trend* between sweep points, not equal-width log intervals.
        ax.set_xscale("linear")
        ax.set_xticks(epsilons)
        ax.set_xticklabels(tick_labels)
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel(f"Target ε  (δ={delta:g})")
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.legend(fontsize=9, title="Mechanism")
    save_fig(fig, out_dir, "utility_vs_budget", ext, footer=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Utility vs privacy budget at iso-(epsilon, delta).")
    add_common_args(parser)
    add_workload_args(parser)
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
                        help="Target epsilon values to sweep. ZCDP/ApproxDP get matching rho.")
    parser.add_argument("--mechanisms", nargs="+", default=None,
                        help="If set, overrides --mechanism and overlays multiple. "
                             "e.g. --mechanisms ZCDP PureDP ApproximateDP RenyiDP")
    args = parser.parse_args()

    set_plot_style()
    cfg = cfg_from_args(args, "exp4_budget_sweep")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = truth_tree(cfg, verbose=True)
    Q, _names, _cdf, sensitivity = build_workload(cfg, args, verbose=True)

    mechs = args.mechanisms or [cfg.mechanism]
    print(f"\nIso-(target_epsilon, delta={cfg.delta:g}) budgeting across {len(args.budgets)} "
          f"epsilon values for mechanisms: {mechs}")
    rows: List[dict] = []
    for mech in mechs:
        rows += _run(cfg, Q, truth, mech, args.budgets)

    csv_path, json_path = dump_results(out_dir, "exp4_budget_sweep", rows, cfg, extras={
        "workload": args.workload,
        "n_random_queries": args.n_random_queries,
        "random_density": args.random_density,
        "target_epsilons": args.budgets,
        "mechanisms": mechs,
        "Q_shape": list(Q.shape),
        "sensitivity": sensitivity,
    })
    print(f"\nRaw results → {csv_path}\nConfig     → {json_path}")

    df = pd.DataFrame(rows)
    footer = experiment_footer(cfg, extra=(
        f"Workload: {args.workload} | Q.shape={Q.shape}, Δ={sensitivity} | "
        f"Mechanisms: {','.join(mechs)} | Target ε: {args.budgets} | δ={cfg.delta:g}"
    ))
    _plot(df, out_dir, cfg.extension, cfg.delta, footer=footer)
    print(f"Plot saved to {out_dir}")


if __name__ == "__main__":
    main()
