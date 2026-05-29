"""Experiment 8 — Categorical association of attributes: real vs private.

Census attributes are almost all categorical (housing type, materials, sex,
geography), so Pearson r — which treats column values as numeric — is misleading
here. The default metric is **Cramér's V**: a chi-square-based, symmetric measure
of association between two categorical variables, bounded in [0, 1]. It works on
nominal *and* ordinal codes, so geographic columns (REGION, PROVINCIA, …) can be
included meaningfully via `--include_geo`.

For each pair of columns this experiment computes V on the real microdata, then
runs the privatized TopDown pipeline, reconstructs noisy microdata from x_hat at
the leaves, and computes V on the noisy data. Pass `--metric pearson` to get the
classic Pearson r matrix instead (treats columns as numeric; useful only if your
attributes are genuinely numeric/ordinal).

Three heatmaps are produced (plus a side-by-side):
  - corr_real.{ext}       — association matrix from the original census microdata
  - corr_private.{ext}    — association matrix from the privatized microdata
  - corr_diff.{ext}       — element-wise (private − real), centered at zero with a
                             diverging colormap so distortion direction is obvious
  - corr_side_by_side.{ext}

Cramér's V here uses Bergsma–Wicher bias correction, which prevents the small-sample
upward bias on sparse contingency tables. For very large n (census-scale) the
correction is negligible but harmless.

Outputs (in --out, default tests/out/exp8_correlation/):
  - corr_real.{ext}, corr_private.{ext}, corr_diff.{ext}, corr_side_by_side.{ext}
  - corr_real.csv, corr_private.csv, corr_diff.csv
  - exp8_correlation_config.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from tests.common import (
    DATASETS,
    _PROJECT_ROOT,
    _release_shared_memory,
    _silenced,
    add_common_args,
    add_workload_args,
    build_mechanism,
    build_workload,
    cfg_from_args,
    experiment_footer,
    save_fig,
    set_plot_style,
)

from topdown import TopDown


def cramers_v(x: pd.Series, y: pd.Series, bias_correction: bool = True) -> float:
    """Cramér's V between two categorical series, in [0, 1].

    With `bias_correction=True` uses the Bergsma–Wicher correction (Bergsma 2013),
    which removes the small-sample upward bias on sparse tables. For census-scale n
    the correction is essentially zero, but it keeps things consistent.
    """
    table = pd.crosstab(x, y)
    if table.size == 0:
        return float("nan")
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.values.sum()
    if n == 0:
        return float("nan")
    phi2 = chi2 / n
    r, k = table.shape
    if not bias_correction:
        denom = min(r - 1, k - 1)
        return float(np.sqrt(phi2 / denom)) if denom > 0 else float("nan")
    phi2c = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(rc - 1, kc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else float("nan")


def cramers_v_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Symmetric matrix of pairwise Cramér's V over all columns of df."""
    cols = list(df.columns)
    n = len(cols)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        out[i, i] = 1.0  # by convention, perfect association with itself
        for j in range(i + 1, n):
            v = cramers_v(df[cols[i]], df[cols[j]])
            out[i, j] = out[j, i] = v
    return pd.DataFrame(out, index=cols, columns=cols)


def _build_and_run(cfg, include_geo: bool, Q=None, sensitivity: int = 1) -> Dict:
    """Run the full privatized pipeline once and return the noisy microdata + timings.

    Unlike `run_once` in common.py, this needs to call construct_microdata before
    shared memory is released, since the leaves' x_hat is consumed there to expand
    rows. So we manage the TopDown lifecycle inline. If Q is None, identity is used.
    """
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
    timings = {}
    with _silenced(stdout=True):
        t0 = time.time(); td.initialize();       timings["init"] = time.time() - t0
        t0 = time.time(); td.measurement_phase(); timings["measure"] = time.time() - t0
        t0 = time.time(); td.estimation_phase();  timings["estimate"] = time.time() - t0
        # Construct microdata from leaf x_hat. This is the slow postprocessing step
        # the other experiments deliberately skip; we need it here.
        t0 = time.time()
        noisy_microdata = td.data_handler.construct_microdata(td.tree)
        timings["construct"] = time.time() - t0

    # Columns to keep: queries always; hierarchy columns optionally.
    keep_cols = list(queries)
    if include_geo:
        keep_cols = list(hierarchy) + keep_cols
    real_df = td.data_handler.dataframe[keep_cols].copy()  # already loaded
    private_df = noisy_microdata[keep_cols].copy()
    _release_shared_memory(td)
    return {
        "real_df": real_df,
        "private_df": private_df,
        "timings": timings,
        "mechanism_report": td.privacy_mechanism.report_guarantee(),
        "sensitivity": sensitivity,
    }


def _corr(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "pearson":
        return df.corr(method="pearson")
    if metric == "cramer":
        return cramers_v_matrix(df)
    raise ValueError(f"Unknown metric: {metric!r}")


def _metric_label(metric: str) -> str:
    return "Cramér's V" if metric == "cramer" else "Pearson r"


def _metric_range(metric: str) -> Tuple[float, float, str, float]:
    """Return (vmin, vmax, cmap, center) for plot styling.

    Cramér's V is in [0, 1]: sequential colormap, no centering.
    Pearson r is in [-1, 1]: diverging colormap centered at 0.
    """
    if metric == "cramer":
        return 0.0, 1.0, "rocket_r", 0.5
    return -1.0, 1.0, "vlag", 0.0


def _plot_single(corr: pd.DataFrame, title: str, out_dir: Path, name: str, ext: str,
                 footer: str, metric: str) -> None:
    vmin, vmax, cmap, center = _metric_range(metric)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(corr) + 3), max(5, 0.6 * len(corr) + 3)))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=cmap, center=center, vmin=vmin, vmax=vmax,
        square=True, linewidths=0.5, cbar_kws={"label": _metric_label(metric)}, ax=ax,
    )
    ax.set_title(title)
    save_fig(fig, out_dir, name, ext, footer=footer)


def _plot_diff(diff: pd.DataFrame, out_dir: Path, name: str, ext: str, footer: str,
               metric: str) -> None:
    # Auto-scale to the real magnitude so even tiny distortions are visible. The colorbar
    # range is symmetric around 0 (diverging), and the cell annotations use an adaptive
    # decimal width derived from the magnitude.
    max_abs = float(np.abs(diff.values).max())
    if max_abs == 0.0:
        vmax = 1e-6
        fmt = ".1e"
    else:
        vmax = max_abs
        # Pick a fmt with enough decimals to actually show the values.
        if max_abs >= 0.1:
            fmt = ".3f"
        elif max_abs >= 0.01:
            fmt = ".4f"
        elif max_abs >= 0.001:
            fmt = ".5f"
        else:
            fmt = ".1e"
    label = "Δ V  (private − real)" if metric == "cramer" else "Δr  (private − real)"
    fig, ax = plt.subplots(figsize=(max(6, 0.85 * len(diff) + 3), max(5, 0.6 * len(diff) + 3)))
    sns.heatmap(
        diff, annot=True, fmt=fmt, annot_kws={"size": 8},
        cmap="RdBu_r", center=0.0, vmin=-vmax, vmax=vmax,
        square=True, linewidths=0.5,
        cbar_kws={"label": f"{label}  (max |Δ| = {max_abs:.2e})"}, ax=ax,
    )
    title = f"{_metric_label(metric)} difference: private − real"
    ax.set_title(title)
    save_fig(fig, out_dir, name, ext, footer=footer)


def _plot_side_by_side(real: pd.DataFrame, private: pd.DataFrame, out_dir: Path,
                       name: str, ext: str, footer: str, metric: str) -> None:
    vmin, vmax, cmap, center = _metric_range(metric)
    fig, axes = plt.subplots(1, 2, figsize=(max(12, 1.3 * len(real) + 6), max(5, 0.6 * len(real) + 3)))
    for ax, mat, title in zip(axes, [real, private], ["Real", "Private"]):
        sns.heatmap(
            mat, annot=True, fmt=".2f", cmap=cmap, center=center, vmin=vmin, vmax=vmax,
            square=True, linewidths=0.5, cbar_kws={"label": _metric_label(metric)}, ax=ax,
        )
        ax.set_title(title)
    fig.suptitle(f"{_metric_label(metric)}: real vs private", y=1.02, fontsize=13, weight="bold")
    save_fig(fig, out_dir, name, ext, footer=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real vs private association heatmaps.")
    add_common_args(parser)
    add_workload_args(parser)
    parser.add_argument("--metric", choices=["cramer", "pearson"], default="cramer",
                        help="Association measure. 'cramer' (default) = Cramér's V, "
                             "right for categorical attributes. 'pearson' = Pearson r, "
                             "treats codes as numeric (only useful for ordinal/numeric data).")
    parser.add_argument("--include_geo", action="store_true",
                        help="Include the hierarchical geographic columns in the matrix. "
                             "Makes sense with --metric cramer; misleading with --metric pearson.")
    args = parser.parse_args()

    set_plot_style()
    cfg = cfg_from_args(args, "exp8_correlation")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = cfg.resolve_queries()
    if len(queries) < 2 and not args.include_geo:
        raise SystemExit("Need at least 2 columns; pass more --queries or --include_geo.")

    if args.include_geo and args.metric == "pearson":
        print("[warn] --include_geo with --metric pearson is misleading: geographic codes "
              "are nominal and their Pearson r reflects ID-numbering coincidences, not "
              "real association. Recommend --metric cramer when including geography.")

    Q, _names, _cdf, sensitivity = build_workload(cfg, args, verbose=False)
    print(f"Running with queries={queries}, process_until={cfg.process_until}, "
          f"mechanism={cfg.mechanism}, budget={cfg.total_budget}, "
          f"metric={args.metric}, include_geo={args.include_geo}.")
    print(f"Workload {args.workload!r}: Q.shape={Q.shape}, sensitivity={sensitivity}")
    print("This run constructs microdata from x_hat, which is slower than the other experiments.")
    print(f"Building real + private {_metric_label(args.metric)} matrices ...\n")

    result = _build_and_run(cfg, include_geo=args.include_geo, Q=Q, sensitivity=sensitivity)
    real_df = result["real_df"]
    private_df = result["private_df"]
    timings = result["timings"]
    print(f"\nTimings: init={timings['init']:.1f}s  meas={timings['measure']:.1f}s  "
          f"est={timings['estimate']:.1f}s  construct={timings['construct']:.1f}s")
    print(f"Real rows: {len(real_df):,}   Private rows: {len(private_df):,}")
    print(f"Mechanism: {result['mechanism_report']}\n")

    print(f"Computing {_metric_label(args.metric)} on {len(real_df.columns)} columns ...")
    t0 = time.time()
    corr_real = _corr(real_df, args.metric)
    t_real = time.time() - t0
    t0 = time.time()
    corr_private = _corr(private_df, args.metric)
    t_priv = time.time() - t0
    print(f"  real matrix:    {t_real:.1f}s")
    print(f"  private matrix: {t_priv:.1f}s")
    corr_private = corr_private.reindex(index=corr_real.index, columns=corr_real.columns)
    corr_diff = corr_private - corr_real

    corr_real.to_csv(out_dir / "corr_real.csv")
    corr_private.to_csv(out_dir / "corr_private.csv")
    corr_diff.to_csv(out_dir / "corr_diff.csv")

    with open(out_dir / "exp8_correlation_config.json", "w") as f:
        json.dump({"config": asdict(cfg),
                   "extras": {
                       "metric": args.metric,
                       "include_geo": args.include_geo,
                       "workload": args.workload,
                       "n_random_queries": args.n_random_queries,
                       "random_density": args.random_density,
                       "sensitivity": sensitivity,
                       "Q_shape": list(Q.shape),
                       "columns": list(real_df.columns),
                       "n_real_rows": int(len(real_df)),
                       "n_private_rows": int(len(private_df)),
                       "mechanism_report": result["mechanism_report"],
                       "timings": timings,
                       "corr_timings": {"real": t_real, "private": t_priv},
                   }}, f, indent=2, default=str)

    print(f"\n{_metric_label(args.metric)} — real:\n", corr_real.round(3))
    print(f"\n{_metric_label(args.metric)} — private:\n", corr_private.round(3))
    print(f"\nΔ (private − real):\n", corr_diff.round(3))

    metric_label = _metric_label(args.metric)
    footer = experiment_footer(cfg, extra=(
        f"Metric: {metric_label} | include_geo: {args.include_geo} | "
        f"Workload: {args.workload} (Q.shape={tuple(Q.shape)}, Δ={sensitivity}) | "
        f"Real rows: {len(real_df):,} | Private rows: {len(private_df):,}"
    ))
    _plot_single(corr_real, f"{metric_label} — real microdata", out_dir, "corr_real",
                 cfg.extension, footer, args.metric)
    _plot_single(corr_private, f"{metric_label} — private microdata", out_dir, "corr_private",
                 cfg.extension, footer, args.metric)
    _plot_diff(corr_diff, out_dir, "corr_diff", cfg.extension, footer, args.metric)
    _plot_side_by_side(corr_real, corr_private, out_dir, "corr_side_by_side",
                       cfg.extension, footer, args.metric)
    print(f"\nPlots and matrices saved to {out_dir}")


if __name__ == "__main__":
    main()
