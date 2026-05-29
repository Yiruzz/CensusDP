"""Experiment 5 — Query matrix design.

Compares several workload matrices on the same dataset and budget:
  - identity        : np.eye(n_cells), one query per contingency cell
  - marginalsK      : all K-way marginals over query columns (K = 1, 2, …)
  - random          : random binary, density --random_density, with sensitivity cap
  - mixed           : 1-way + 2-way marginals concatenated with `random`

For each workload it reports the workload's number of queries and L1-sensitivity, plus
utility metrics (leaf TVD, global MAPE) averaged over --trials.

Outputs (in --out, default tests/out/exp5_query_matrix/):
  - tvd_by_workload.{ext}        — bar chart of leaf TVD per workload
  - mape_by_workload.{ext}       — bar chart of global MAPE per workload
  - exp5_query_matrix.csv
  - exp5_query_matrix_config.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tests.common import (
    add_common_args,
    cfg_from_args,
    dump_results,
    experiment_footer,
    mape,
    metrics_by_level,
    resolve_workload,
    run_once,
    save_fig,
    set_plot_style,
    truth_tree,
)


def _build_all_workloads(specs: List[str], contingency_df, query_columns: List[str],
                         n_random: int, density: float, rng):
    out = []
    for spec in specs:
        try:
            Q, names = resolve_workload(spec, contingency_df, query_columns,
                                        rng=rng, n_random=n_random, density=density)
        except ValueError as e:
            print(f"[skip] workload {spec!r}: {e}")
            continue
        out.append((spec, Q, names))
    return out


def _run_trials(cfg, workloads, truth) -> List[dict]:
    rows = []
    for spec, Q, names in workloads:
        sens = int(Q.sum(axis=0).max())
        print(f"\n[workload {spec!r}]  n_queries={Q.shape[0]}, sensitivity={sens}")
        for t in range(cfg.trials):
            res = run_once(cfg, Q=Q, trial_index=t, verbose=False)
            per_level = metrics_by_level(truth["x_truth"], res["x_hat"], res["levels"])
            leaf_lvl = int(res["levels"].max())
            leaf_tvd = per_level[leaf_lvl]["TVD_mean"]
            global_mape = float(np.mean([
                mape(res["y_truth"][i], res["y_hat"][i]) for i in range(res["y_truth"].shape[0])
            ]))
            rows.append({
                "workload": spec,
                "n_queries": int(Q.shape[0]),
                "sensitivity": sens,
                "trial": t,
                "leaf_TVD": leaf_tvd,
                "global_MAPE": global_mape,
                "init": res["timings"]["init"],
                "measure": res["timings"]["measure"],
                "estimate": res["timings"]["estimate"],
                "total": res["timings"]["total"],
            })
            print(f"   trial {t}: leaf_TVD={leaf_tvd:.4f}  global_MAPE={global_mape:.2f}%  "
                  f"total={res['timings']['total']:.1f}s")
    return rows


def _bar(df: pd.DataFrame, metric: str, ylabel: str, title: str, out_dir: Path, ext: str, name: str,
         footer: str = "") -> None:
    agg = df.groupby("workload").agg(
        mean=(metric, "mean"),
        std=(metric, "std"),
        n_queries=("n_queries", "first"),
        sensitivity=("sensitivity", "first"),
    ).reset_index()
    agg["std"] = agg["std"].fillna(0)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(agg) + 3), 5))
    xs = np.arange(len(agg))
    ax.bar(xs, agg["mean"], yerr=agg["std"], capsize=4, color="tab:blue", alpha=0.85)
    labels = [f"{w}\nQ={int(nq)}, Δ={int(s)}" for w, nq, s in
              zip(agg["workload"], agg["n_queries"], agg["sensitivity"])]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_fig(fig, out_dir, name, ext, footer=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare query matrix designs.")
    add_common_args(parser)
    parser.add_argument("--workloads", nargs="+",
                        default=["identity", "marginals1", "marginals2", "random", "mixed"])
    parser.add_argument("--n_random_queries", type=int, default=20)
    parser.add_argument("--random_density", type=float, default=0.1)
    args = parser.parse_args()

    set_plot_style()
    cfg = cfg_from_args(args, "exp5_query_matrix")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = truth_tree(cfg, verbose=True)

    from data_handler import DataHandler
    from pathlib import Path as _P
    dh = DataHandler(file_path=str(_P(cfg.data_path()).resolve()))
    dh.hierarchical_columns = cfg.resolve_hierarchy()
    dh.query_columns = cfg.resolve_queries()
    dh.read_data(cfg.resolve_hierarchy() + cfg.resolve_queries(), sep=cfg.sep())
    contingency_df = dh.generate_contingency_dataframe(cfg.resolve_queries())

    rng = np.random.default_rng()
    workloads = _build_all_workloads(
        args.workloads, contingency_df, cfg.resolve_queries(),
        n_random=args.n_random_queries, density=args.random_density, rng=rng,
    )
    if not workloads:
        raise SystemExit("No valid workloads to run.")

    rows = _run_trials(cfg, workloads, truth)
    csv_path, json_path = dump_results(out_dir, "exp5_query_matrix", rows, cfg, extras={
        "workloads": args.workloads,
        "n_random_queries": args.n_random_queries,
        "random_density": args.random_density,
    })
    print(f"\nRaw results → {csv_path}\nConfig     → {json_path}")

    df = pd.DataFrame(rows)
    footer = experiment_footer(cfg, extra=f"Workloads: {','.join(args.workloads)}")
    _bar(df, "leaf_TVD", "Leaf-level TVD  (0 – 1)",
         "Leaf TVD by workload",
         out_dir, cfg.extension, "tvd_by_workload", footer=footer)
    _bar(df, "global_MAPE", "Global query MAPE (%)", "Global query MAPE by workload",
         out_dir, cfg.extension, "mape_by_workload", footer=footer)
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
