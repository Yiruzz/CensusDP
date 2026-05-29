"""Regenerate every plot from the CSVs already in tests/out/<expN>/.

Use this after changing labels / units / footers in the plot helpers so we don't have
to re-run the long experiments. Reads each `expN_*.csv` + `expN_*_config.json` and
rebuilds the exact same plots the original scripts would have produced now.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tests.common import (
    DATASETS,
    ExperimentConfig,
    experiment_footer,
    resolve_workload,
    set_plot_style,
)

ROOT = Path(__file__).resolve().parent / "out"


def _load(exp_dir: str, csv_name: str, cfg_name: str):
    d = ROOT / exp_dir
    csv_path = d / f"{csv_name}.csv"
    cfg_path = d / f"{cfg_name}.json"
    if not d.exists():
        raise FileNotFoundError(
            f"no run found at {d} — run `python -m tests.{exp_dir}` first."
        )
    if not csv_path.exists() or not cfg_path.exists():
        raise FileNotFoundError(
            f"{d} exists but {csv_path.name} or {cfg_path.name} is missing — "
            f"re-run `python -m tests.{exp_dir}`."
        )
    df = pd.read_csv(csv_path)
    with open(cfg_path) as f:
        meta = json.load(f)
    cfg_dict = meta["config"]
    valid = ExperimentConfig.__dataclass_fields__.keys()
    cfg = ExperimentConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
    extras = meta.get("extras", {})
    return df, cfg, extras, d


def replot_exp1():
    from tests.exp1_runtime import _plot_size, _plot_depth, _plot_stacked
    df, cfg, extras, d = _load("exp1_runtime", "exp1_runtime", "exp1_runtime_config")
    rows = df.to_dict("records")
    hierarchy = DATASETS[cfg.dataset]["hierarchy"]
    footer = experiment_footer(cfg, extra=(
        f"Query progression: {','.join(extras.get('query_progression', []))}  "
        f"| Depth sweep: {','.join(extras.get('depth_sweep', []))}"
    ))
    _plot_size(rows, d, cfg.extension, footer=footer)
    _plot_depth(rows, d, cfg.extension, hierarchy, footer=footer)
    _plot_stacked(rows, d, cfg.extension, footer=footer)
    print(f"[exp1] replotted -> {d}")


def replot_exp2():
    from tests.exp2_utility_per_level import _plot_metric_per_level
    df, cfg, extras, d = _load("exp2_utility_per_level", "exp2_per_level", "exp2_per_level_config")
    workload = extras.get("workload", "identity")
    q_shape = extras.get("Q_shape", [None, None])
    sens = extras.get("sensitivity", "?")
    footer = experiment_footer(cfg, extra=f"Workload: {workload} | Q.shape={tuple(q_shape)}, Δ={sens}")
    _plot_metric_per_level(df, "TVD", d, cfg.extension,
                           ylabel="Mean TVD  (0 – 1)",
                           title=f"TVD per tree level — {workload}",
                           footer=footer)
    _plot_metric_per_level(df, "MAE", d, cfg.extension,
                           ylabel="Mean Absolute Error",
                           title=f"MAE per tree level — {workload}",
                           footer=footer)
    _plot_metric_per_level(df, "L1", d, cfg.extension,
                           ylabel="Mean L1 distance",
                           title=f"L1 per tree level — {workload}",
                           footer=footer)
    print(f"[exp2] replotted -> {d}")


def replot_exp3():
    from tests.exp3_query_error import _plot_lines
    df, cfg, extras, d = _load("exp3_query_error", "exp3_query_error", "exp3_query_error_config")
    workload = extras.get("workload", "?")
    footer = experiment_footer(cfg, extra=f"Workload: {workload} | top_queries={extras.get('top_queries', '?')}")
    _plot_lines(df, d, cfg.extension, footer=footer)
    print(f"[exp3] replotted -> {d}")


def replot_exp4():
    from tests.exp4_budget_sweep import _plot
    df, cfg, extras, d = _load("exp4_budget_sweep", "exp4_budget_sweep", "exp4_budget_sweep_config")
    # Old CSVs (pre iso-(epsilon, delta) refactor) carried `budget` per mechanism in mixed
    # units. If we see that schema, rename to the new target_epsilon column so the new
    # _plot can render — values are not unit-converted, so the curves remain accurate as
    # long as the older runs were single-mechanism. Multi-mechanism old CSVs are not safe
    # to replot under iso-(epsilon, delta) semantics.
    if "target_epsilon" not in df.columns and "budget" in df.columns:
        df = df.rename(columns={"budget": "target_epsilon"})
    workload = extras.get("workload", "?")
    q_shape = extras.get("Q_shape", [None, None])
    sens = extras.get("sensitivity", "?")
    budgets = extras.get("target_epsilons", extras.get("budgets", []))
    footer = experiment_footer(cfg, extra=(
        f"Workload: {workload} | Q.shape={tuple(q_shape)}, Δ={sens} | "
        f"Mechanisms: {','.join(extras.get('mechanisms', []))} | "
        f"Target ε: {budgets} | δ={cfg.delta:g}"
    ))
    _plot(df, d, cfg.extension, cfg.delta, footer=footer)
    print(f"[exp4] replotted -> {d}")


def replot_exp5():
    from tests.exp5_query_matrix import _bar
    df, cfg, extras, d = _load("exp5_query_matrix", "exp5_query_matrix", "exp5_query_matrix_config")
    footer = experiment_footer(cfg, extra=f"Workloads: {','.join(extras.get('workloads', []))}")
    _bar(df, "leaf_TVD", "Leaf-level TVD  (0 – 1)",
         "Leaf TVD by workload", d, cfg.extension, "tvd_by_workload", footer=footer)
    _bar(df, "global_MAPE", "Global query MAPE (%)", "Global query MAPE by workload",
         d, cfg.extension, "mape_by_workload", footer=footer)
    print(f"[exp5] replotted -> {d}")


def replot_exp6():
    from tests.exp6_budget_allocation import _plot_tvd_per_level, _plot_mape_summary
    df, cfg, extras, d = _load("exp6_budget_allocation", "exp6_budget_allocation", "exp6_budget_allocation_config")
    workload = extras.get("workload", "?")
    strategies = extras.get("strategies", [])
    nodes_per_level = extras.get("nodes_per_level", [])
    footer = experiment_footer(cfg, extra=(
        f"Workload: {workload} | Strategies: {','.join(strategies)} | "
        f"Nodes/level: {nodes_per_level}"
    ))
    _plot_tvd_per_level(df, d, cfg.extension, cfg.total_budget, cfg.mechanism, footer=footer)
    _plot_mape_summary(df, d, cfg.extension, footer=footer)
    print(f"[exp6] replotted -> {d}")


def replot_exp7(exp_dir: str = "exp7_mechanisms"):
    from tests.exp7_mechanisms import _bar
    df, cfg, extras, d = _load(exp_dir, "exp7_mechanisms", "exp7_mechanisms_config")
    if "per_mechanism_budget" not in df.columns and "total_budget" in df.columns:
        df = df.rename(columns={"total_budget": "per_mechanism_budget"})
    workload = extras.get("workload", "?")
    target_eps = extras.get("target_epsilon", cfg.total_budget)
    rho_equiv = extras.get("iso_rho")
    sens_hint = ""
    if "sensitivity" in extras:
        sens_hint = f" Δ={extras['sensitivity']}"
    extra = f"Workload: {workload}{sens_hint} | Iso-(ε={target_eps:g}, δ={cfg.delta:g})"
    if rho_equiv is not None:
        extra += f"; ZCDP/ApproxDP ρ={rho_equiv:.4f}"
    footer = experiment_footer(cfg, extra=extra)
    _bar(df, "leaf_TVD", "Leaf-level TVD  (0 – 1)",
         f"Leaf TVD by mechanism — iso-(ε={target_eps:g}, δ={cfg.delta:g}), workload={workload}",
         d, cfg.extension, "tvd_by_mechanism", cfg.delta, footer=footer)
    _bar(df, "global_MAPE", "Global query MAPE (%)",
         f"Global query MAPE by mechanism — iso-(ε={target_eps:g}, δ={cfg.delta:g}), workload={workload}",
         d, cfg.extension, "mape_by_mechanism", cfg.delta, footer=footer)
    print(f"[exp7] replotted -> {d}")


def replot_exp9():
    from tests.exp9_matrix_size import _plot_runtime, _plot_stacked
    df, cfg, extras, d = _load("exp9_matrix_size", "exp9_matrix_size", "exp9_matrix_size_config")
    sweep = extras.get("n_queries_sweep", [])
    n_cells = extras.get("n_cells", "?")
    density = extras.get("random_density", "?")
    footer = experiment_footer(cfg, extra=(
        f"Workload: random (density={density}) | n_cells={n_cells} | n_queries sweep: {sweep}"
    ))
    _plot_runtime(df, d, cfg.extension, footer=footer)
    _plot_stacked(df, d, cfg.extension, footer=footer)
    print(f"[exp9] replotted -> {d}")


def main():
    set_plot_style()
    if not ROOT.exists():
        print(f"No outputs to replot: {ROOT} does not exist yet — run an experiment first "
              f"(e.g. `python -m tests.exp1_runtime --help`).")
        return
    for fn in (replot_exp1, replot_exp2, replot_exp3, replot_exp4, replot_exp5,
               replot_exp6, replot_exp7, replot_exp9):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"[skip] {fn.__name__}: {e}")
        except Exception as e:
            print(f"[skip] {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
