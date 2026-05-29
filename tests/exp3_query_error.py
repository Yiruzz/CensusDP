"""Experiment 3 — Per-query percentage error vs tree level.

For each query in Q this experiment computes the mean percentage error
|y_hat - y_truth| / max(1, y_truth) averaged over the nodes at each tree level. y_truth is
the noise-free `Q @ x`; y_hat is `Q @ x_hat`. Averaging is then taken across --trials.

Outputs (in --out, default tests/out/exp3_query_error/):
  - per_query_mape.{ext}        — one line per query, MAPE vs tree level
  - exp3_query_error.csv        — raw rows: trial, level, query_index, query_name, MAPE
  - exp3_query_error_config.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # color palette only

from tests.common import (
    add_common_args,
    cfg_from_args,
    dump_results,
    experiment_footer,
    query_errors,
    resolve_workload,
    run_once,
    save_fig,
    set_plot_style,
)


def _run_trials(cfg, Q: np.ndarray, query_names: List[str], top_queries: int) -> List[dict]:
    rows = []
    for t in range(cfg.trials):
        print(f"\n--- trial {t+1}/{cfg.trials} ---")
        res = run_once(cfg, Q=Q, trial_index=t, verbose=False)
        per_level = query_errors(res["y_truth"], res["y_hat"], res["levels"])
        # Pick which queries to retain. If top_queries is set, pick by largest mean truth magnitude.
        keep_idx = _select_top_queries(res["y_truth"], top_queries)
        for lvl, mape_arr in per_level.items():
            for qi in keep_idx:
                rows.append({
                    "trial": t,
                    "level": int(lvl),
                    "query_index": int(qi),
                    "query_name": query_names[qi] if qi < len(query_names) else f"q_{qi}",
                    "MAPE": float(mape_arr[qi]),
                })
    return rows


def _select_top_queries(y_truth: np.ndarray, top_queries: int) -> List[int]:
    n_queries = y_truth.shape[1]
    if top_queries <= 0 or top_queries >= n_queries:
        return list(range(n_queries))
    mean_truth = y_truth.mean(axis=0)
    return list(np.argsort(-mean_truth)[:top_queries])


def _plot_lines(df: pd.DataFrame, out_dir: Path, ext: str, footer: str = "") -> None:
    agg = df.groupby(["query_name", "level"])["MAPE"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    queries = sorted(agg["query_name"].unique())
    palette = sns.color_palette("tab10", n_colors=max(10, len(queries)))
    for i, q in enumerate(queries):
        sub = agg[agg["query_name"] == q].sort_values("level")
        ax.plot(sub["level"], sub["MAPE"], "-o", label=q, color=palette[i % len(palette)])
    ax.set_xlabel("Tree level (0 = root)")
    ax.set_ylabel("Mean Absolute Percentage Error (%)")
    ax.set_title("Per-query MAPE vs tree level")
    if len(queries) <= 15:
        ax.legend(fontsize=8, ncol=2)
    save_fig(fig, out_dir, "per_query_mape", ext, footer=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-query percentage error experiment.")
    add_common_args(parser)
    parser.add_argument("--workload", default="marginals1",
                        help="Workload spec: identity | marginals1 | marginals2 | random | mixed.")
    parser.add_argument("--n_random_queries", type=int, default=20)
    parser.add_argument("--random_density", type=float, default=0.1)
    parser.add_argument("--top_queries", type=int, default=10,
                        help="Limit plot to the top-N queries by truth magnitude. 0 = all.")
    args = parser.parse_args()

    set_plot_style()
    cfg = cfg_from_args(args, "exp3_query_error")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    rows = _run_trials(cfg, Q, names, args.top_queries)
    csv_path, json_path = dump_results(out_dir, "exp3_query_error", rows, cfg, extras={
        "workload": args.workload,
        "n_random_queries": args.n_random_queries,
        "random_density": args.random_density,
        "top_queries": args.top_queries,
        "Q_shape": list(Q.shape),
    })
    print(f"\nRaw results → {csv_path}\nConfig     → {json_path}")

    df = pd.DataFrame(rows)
    footer = experiment_footer(cfg, extra=f"Workload: {args.workload} | top_queries={args.top_queries}")
    _plot_lines(df, out_dir, cfg.extension, footer=footer)
    print(f"Plot saved to {out_dir}")


if __name__ == "__main__":
    main()
