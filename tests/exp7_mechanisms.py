"""Experiment 7 — DP mechanism comparison at iso-(epsilon, delta).

By default, every mechanism is calibrated to the SAME (target epsilon, delta)-DP guarantee:

  PureDP        — epsilon = target_eps                         (no delta)
  RenyiDP       — Sum eps_L = target_eps at the given delta    (joint-alpha RDP)
  ZCDP          — Sum rho_L = inverse Bun-Steinke(target_eps, delta)
  ApproximateDP — Sum rho_L = inverse Bun-Steinke(target_eps, delta)

so the four bars represent four mechanisms operating at the same privacy guarantee. This
makes the comparison meaningful.

Per-mechanism overrides (--budgets_pure / --budgets_zcdp / --budgets_approx /
--budgets_renyi) still work and bypass the iso-(eps, delta) conversion when supplied.

Outputs (in --out, default tests/out/exp7_mechanisms/):
  - tvd_by_mechanism.{ext}
  - mape_by_mechanism.{ext}
  - exp7_mechanisms.csv
  - exp7_mechanisms_config.json
"""
from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tests.common import (
    _DEFAULTS,
    add_common_args,
    cfg_from_args,
    dump_results,
    epsilon_to_rho,
    experiment_footer,
    mape,
    metrics_by_level,
    resolve_workload,
    rho_to_epsilon,
    run_once,
    save_fig,
    set_plot_style,
    truth_tree,
)


def _iso_budget_for(mechanism: str, target_eps: float, delta: float) -> float:
    """Return the per-mechanism parameter that realizes (target_eps, delta)-DP."""
    if mechanism in ("PureDP", "RenyiDP"):
        return target_eps
    if mechanism in ("ZCDP", "ApproximateDP"):
        return epsilon_to_rho(target_eps, delta)
    raise ValueError(f"Unknown mechanism: {mechanism!r}")


def _budget_label(mechanism: str, value: float, delta: float) -> str:
    """One x-tick label per mechanism showing both its native param and the equivalent epsilon."""
    if mechanism == "PureDP":
        return f"{mechanism}\nepsilon={value:g}"
    if mechanism == "RenyiDP":
        return f"{mechanism}\nepsilon={value:g}"
    eps_eq = rho_to_epsilon(value, delta)
    return f"{mechanism}\nrho={value:.4g}\n(≡ epsilon≈{eps_eq:.2f})"


def _run(cfg, Q, truth, budgets: Dict[str, float]) -> List[dict]:
    rows = []
    for mech_name, b in budgets.items():
        sub_cfg = replace(cfg, mechanism=mech_name, total_budget=b)
        print(f"\n[{mech_name}]  per-mechanism budget = {b:.4g}")
        for t in range(cfg.trials):
            res = run_once(sub_cfg, Q=Q, trial_index=t, verbose=False)
            per_level = metrics_by_level(truth["x_truth"], res["x_hat"], res["levels"])
            leaf_lvl = int(res["levels"].max())
            leaf_tvd = per_level[leaf_lvl]["TVD_mean"]
            global_mape = float(np.mean([
                mape(res["y_truth"][i], res["y_hat"][i]) for i in range(res["y_truth"].shape[0])
            ]))
            rows.append({
                "mechanism": mech_name,
                "per_mechanism_budget": float(b),
                "trial": t,
                "leaf_TVD": leaf_tvd,
                "global_MAPE": global_mape,
                "mechanism_report": res["mechanism_report"].splitlines()[0],
            })
            print(f"   trial {t}: leaf_TVD={leaf_tvd:.4f}  global_MAPE={global_mape:.2f}%")
    return rows


def _bar(df: pd.DataFrame, metric: str, ylabel: str, title: str,
         out_dir: Path, ext: str, name: str, delta: float, footer: str = "") -> None:
    agg = df.groupby("mechanism", sort=False).agg(
        mean=(metric, "mean"),
        std=(metric, "std"),
        budget=("per_mechanism_budget", "first"),
    ).reset_index()
    agg["std"] = agg["std"].fillna(0)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(agg) + 2), 5.5))
    xs = np.arange(len(agg))
    ax.bar(xs, agg["mean"], yerr=agg["std"], capsize=4, color="tab:cyan", alpha=0.85)
    labels = [_budget_label(m, b, delta) for m, b in zip(agg["mechanism"], agg["budget"])]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_fig(fig, out_dir, name, ext, footer=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DP mechanisms at iso-(epsilon, delta).")
    add_common_args(parser)
    parser.add_argument("--workload", default="identity")
    parser.add_argument("--n_random_queries", type=int, default=20)
    parser.add_argument("--random_density", type=float, default=0.1)
    parser.add_argument("--mechanisms", nargs="+",
                        default=["PureDP", "ZCDP", "ApproximateDP", "RenyiDP"])
    parser.add_argument("--budgets_pure", type=float, default=None,
                        help="Override the iso-(eps, delta) budget for PureDP. Interpreted as epsilon.")
    parser.add_argument("--budgets_zcdp", type=float, default=None,
                        help="Override the iso-(eps, delta) budget for ZCDP. Interpreted as rho.")
    parser.add_argument("--budgets_approx", type=float, default=None,
                        help="Override the iso-(eps, delta) budget for ApproximateDP. Interpreted as rho.")
    parser.add_argument("--budgets_renyi", type=float, default=None,
                        help="Override the iso-(eps, delta) budget for RenyiDP. Interpreted as epsilon.")
    args = parser.parse_args()

    set_plot_style()
    cfg = cfg_from_args(args, "exp7_mechanisms")
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
    Q, _names = resolve_workload(
        args.workload, contingency_df, cfg.resolve_queries(),
        rng=rng, n_random=args.n_random_queries, density=args.random_density,
    )
    print(f"\nWorkload {args.workload!r}: Q.shape={Q.shape}, sensitivity={int(Q.sum(axis=0).max())}")

    # --budget is interpreted as target epsilon here. Without --budget, fall back to
    # the canonical epsilon — budgets["PureDP"] in the config, since PureDP's native
    # unit is epsilon. We can't just use cfg.total_budget because the per-mechanism
    # defaults table means it's in native units of the *default* mechanism (e.g. rho
    # for ZCDP), which is not what an iso comparison wants.
    target_eps = args.budget if args.budget is not None else float(_DEFAULTS["budgets"]["PureDP"])
    rho_equiv = epsilon_to_rho(target_eps, cfg.delta)
    print(f"\nIso-(epsilon={target_eps}, delta={cfg.delta}) budgeting:")
    print(f"  PureDP / RenyiDP : epsilon = {target_eps}")
    print(f"  ZCDP / ApproximateDP : rho = {rho_equiv:.6f}  (matches epsilon via inverse Bun-Steinke)")

    overrides = {
        "PureDP": args.budgets_pure,
        "ZCDP": args.budgets_zcdp,
        "ApproximateDP": args.budgets_approx,
        "RenyiDP": args.budgets_renyi,
    }
    budgets = {
        m: (overrides.get(m) if overrides.get(m) is not None
            else _iso_budget_for(m, target_eps, cfg.delta))
        for m in args.mechanisms
    }

    rows = _run(cfg, Q, truth, budgets)
    csv_path, json_path = dump_results(out_dir, "exp7_mechanisms", rows, cfg, extras={
        "workload": args.workload,
        "n_random_queries": args.n_random_queries,
        "random_density": args.random_density,
        "target_epsilon": target_eps,
        "iso_rho": rho_equiv,
        "budgets": budgets,
    })
    print(f"\nRaw results -> {csv_path}\nConfig     -> {json_path}")

    df = pd.DataFrame(rows)
    sensitivity = int(Q.sum(axis=0).max())
    footer = experiment_footer(cfg, extra=(
        f"Workload: {args.workload} | Q.shape={Q.shape}, Delta={sensitivity} | "
        f"Iso-(epsilon={target_eps:g}, delta={cfg.delta:g}); ZCDP/ApproxDP rho={rho_equiv:.4f}"
    ))
    _bar(df, "leaf_TVD", "Leaf-level TVD  (0 - 1)",
         f"Leaf TVD by mechanism — iso-(epsilon={target_eps:g}, delta={cfg.delta:g}), "
         f"workload={args.workload}, Delta={sensitivity}",
         out_dir, cfg.extension, "tvd_by_mechanism", cfg.delta, footer=footer)
    _bar(df, "global_MAPE", "Global query MAPE (%)",
         f"Global query MAPE by mechanism — iso-(epsilon={target_eps:g}, delta={cfg.delta:g}), "
         f"workload={args.workload}, Delta={sensitivity}",
         out_dir, cfg.extension, "mape_by_mechanism", cfg.delta, footer=footer)
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
