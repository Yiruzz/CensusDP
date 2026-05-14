#!/usr/bin/env python3
"""Benchmark TopDown.estimation_phase: wall time and peak RSS (main + workers).

Run baseline before any change:
    .venv/bin/python bench_estimation.py --label baseline --runs 3

Run after a change with a different label to compare:
    .venv/bin/python bench_estimation.py --label lp-writer --runs 3

Inspect all stored runs side by side:
    .venv/bin/python bench_estimation.py --compare

Results append to bench_results.json so historical runs survive across attempts.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    sys.stderr.write("bench requires psutil. Install with: .venv/bin/pip install psutil\n")
    sys.exit(1)

from topdown import TopDown
from constraints.contextual_constraints import SumEqualRealTotal
from constraints.logical_expressions.atomic import Equal, NotEqual, TrueExpression
from constraints.logical_expressions.compound import And, Implies


GEO_COLUMNS = ['REGION', 'PROVINCIA', 'COMUNA', 'DC', 'ZC_LOC']
DATA_PATH = 'data/csv-viviendas-censo-2017/Microdato_Censo2017-Viviendas.csv'


class RSSSampler(threading.Thread):
    """Poll RSS of the root process and all live descendants; keep the peak total."""

    def __init__(self, root_pid: int, interval: float = 0.05) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.root = psutil.Process(root_pid)
        self.peak_total = 0
        self.peak_main = 0
        self.peak_children = 0
        self.samples = 0
        self._stop_flag = threading.Event()

    def _sample(self) -> None:
        try:
            main_rss = self.root.memory_info().rss
        except psutil.NoSuchProcess:
            return
        children_rss = 0
        try:
            for child in self.root.children(recursive=True):
                try:
                    children_rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except psutil.NoSuchProcess:
            pass
        total = main_rss + children_rss
        if total > self.peak_total:
            self.peak_total = total
        if main_rss > self.peak_main:
            self.peak_main = main_rss
        if children_rss > self.peak_children:
            self.peak_children = children_rss

    def run(self) -> None:
        while not self._stop_flag.is_set():
            self._sample()
            self.samples += 1
            self._stop_flag.wait(self.interval)

    def stop(self) -> None:
        self._stop_flag.set()
        self.join(timeout=2.0)


def build_topdown(process_until: str, queries: list[str], user_constraints: bool) -> TopDown:
    process_until_idx = GEO_COLUMNS.index(process_until)
    geo = GEO_COLUMNS[:process_until_idx + 1]
    out_path = f"data/out/bench_viviendas_{process_until}_{'_'.join(queries)}.csv"
    solver_options = {'OutputFlag': 0, 'Threads': 1}

    td = TopDown(
        data_path=DATA_PATH,
        hierarchy=geo,
        queries=queries,
        out_path=out_path,
        optimizer='gurobi',
        solver_options=solver_options,
        optimizer_path=None,
    )

    # Same budget shape as main.py: exponential over a fixed 6 levels.
    n_levels = 6
    total_budget = 10
    aux = sum(2 ** i for i in range(n_levels))
    pp = [(total_budget / aux) * (2 ** i) for i in range(n_levels)]
    td.set_privacy_parameters(pp)
    td.set_mechanism('discrete_laplace')

    td.set_constraint_to_level(process_until_idx, SumEqualRealTotal(expression=TrueExpression()))
    if user_constraints:
        left = NotEqual('P02', 1)
        right = And(Equal('P03A', 98), Equal('P03B', 98))
        td.set_constraint_to_tree(Implies(left, right))
    return td


def run_one(process_until: str, queries: list[str], user_constraints: bool, sampler_interval: float) -> dict:
    td = build_topdown(process_until, queries, user_constraints)

    t0 = time.perf_counter()
    td.initialize()
    t_init = time.perf_counter() - t0

    t0 = time.perf_counter()
    td.measurement_phase()
    t_meas = time.perf_counter() - t0

    gc.collect()
    rss_before = psutil.Process().memory_info().rss

    sampler = RSSSampler(os.getpid(), interval=sampler_interval)
    sampler.start()
    t0 = time.perf_counter()
    td.estimation_phase()
    t_est = time.perf_counter() - t0
    sampler.stop()

    return {
        't_initialize_s': t_init,
        't_measurement_s': t_meas,
        't_estimation_s': t_est,
        'rss_before_estimation_bytes': rss_before,
        'rss_peak_total_bytes': sampler.peak_total,
        'rss_peak_main_bytes': sampler.peak_main,
        'rss_peak_children_bytes': sampler.peak_children,
        'rss_delta_from_before_bytes': sampler.peak_total - rss_before,
        'sampler_samples': sampler.samples,
    }


def git_meta() -> dict:
    def _run(*args):
        try:
            return subprocess.check_output(['git', *args], stderr=subprocess.DEVNULL).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    porcelain = _run('status', '--porcelain') or ''
    return {
        'sha': _run('rev-parse', 'HEAD'),
        'branch': _run('rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty': bool(porcelain.strip()),
    }


def fmt_mib(n: int | float) -> str:
    return f"{n / (1024 * 1024):.1f} MiB"


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else [data]


def append_results(path: Path, summary: dict) -> None:
    data = load_results(path)
    data.append(summary)
    path.write_text(json.dumps(data, indent=2))


def print_compare(path: Path) -> None:
    data = load_results(path)
    if not data:
        print(f"no results in {path}")
        return
    print(f"{'label':<28} {'runs':>5} {'t_est median (s)':>18} {'t_est min':>10} "
          f"{'peak RSS median':>18} {'peak RSS max':>15}  {'sha':<10}  config")
    print("-" * 130)
    for entry in data:
        runs = entry.get('runs', [])
        if not runs:
            continue
        ests = [r['t_estimation_s'] for r in runs]
        peaks = [r['rss_peak_total_bytes'] for r in runs]
        cfg = entry.get('config', {})
        cfg_str = f"{cfg.get('process_until')} {','.join(cfg.get('queries', []))}" \
                  f"{' +uc' if cfg.get('user_constraints') else ''}"
        sha = (entry.get('git', {}).get('sha') or '')[:7]
        dirty = '*' if entry.get('git', {}).get('dirty') else ''
        print(f"{entry.get('label', '?'):<28} {len(runs):>5} "
              f"{statistics.median(ests):>18.2f} {min(ests):>10.2f} "
              f"{fmt_mib(statistics.median(peaks)):>18} {fmt_mib(max(peaks)):>15}  "
              f"{sha+dirty:<10}  {cfg_str}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--label', help="Identifier for this run set, e.g. 'baseline-pyomo-nl'")
    p.add_argument('--process_until', default='COMUNA')
    p.add_argument('--queries', nargs='+', default=['P01', 'P02', 'P03A', 'P03B'])
    p.add_argument('--user_constraints', action='store_true')
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--out', default='bench_results.json')
    p.add_argument('--sampler_interval', type=float, default=0.05)
    p.add_argument('--compare', action='store_true', help="Print stored results and exit without running.")
    args = p.parse_args()

    out_path = Path(args.out)

    if args.compare:
        print_compare(out_path)
        return

    if not args.label:
        sys.stderr.write("--label is required when running a benchmark (omit only with --compare).\n")
        sys.exit(2)

    if not Path(DATA_PATH).exists():
        sys.stderr.write(f"data file not found: {DATA_PATH}\n")
        sys.exit(2)

    print(f"\n=== bench label: {args.label} ===")
    print(f"config: process_until={args.process_until} queries={args.queries} "
          f"user_constraints={args.user_constraints} runs={args.runs}\n")

    runs = []
    for i in range(args.runs):
        print(f"\n--- run {i + 1}/{args.runs} ---", flush=True)
        rec = run_one(args.process_until, args.queries, args.user_constraints, args.sampler_interval)
        rec['run_index'] = i
        runs.append(rec)
        print(f"  t_estimation = {rec['t_estimation_s']:.2f} s")
        print(f"  peak rss     = {fmt_mib(rec['rss_peak_total_bytes'])} "
              f"(main {fmt_mib(rec['rss_peak_main_bytes'])} + "
              f"children {fmt_mib(rec['rss_peak_children_bytes'])})")

    summary = {
        'label': args.label,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'git': git_meta(),
        'config': {
            'process_until': args.process_until,
            'queries': args.queries,
            'user_constraints': args.user_constraints,
            'runs': args.runs,
            'sampler_interval_s': args.sampler_interval,
        },
        'runs': runs,
    }
    append_results(out_path, summary)

    est_times = [r['t_estimation_s'] for r in runs]
    peaks = [r['rss_peak_total_bytes'] for r in runs]
    print(f"\n=== summary: {args.label} ===")
    print(f"estimation wall time (s) — median: {statistics.median(est_times):.2f}, "
          f"min: {min(est_times):.2f}, max: {max(est_times):.2f}")
    print(f"peak RSS (main + children) — median: {fmt_mib(statistics.median(peaks))}, "
          f"min: {fmt_mib(min(peaks))}, max: {fmt_mib(max(peaks))}")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
