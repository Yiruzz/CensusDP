"""Run the configurations of one or more experiments, round by round, resuming where it stopped.

    python -m benchmarks.orchestrate exp3_composition --reps 5 --timeout 12
    python -m benchmarks.orchestrate exp7 --datasets sinasc --sample --dry-run

Round r of every configuration finishes before round r + 1 starts, so the medians of the first
rounds can be analysed while the rest runs. Each run is two processes, one after the other: the
driver (which records time and memory) and benchmarks.metrics. The orchestrator only waits.

Every run, failed ones included, appends one line to data/out/<dataset>/runs.jsonl
(runs_sample.jsonl with --sample). Launching the same command again skips what is recorded;
--retry runs again what failed or timed out. Next to it stay <name>.json, <name>_metrics.json
(pairs and triples, for experiment 5) and <name>.log. The synthetic CSV is deleted unless
--keep-csv.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from benchmarks import common
from benchmarks.experiments import EXPERIMENTS

NAMED = ('full_joint', 'structure', 'rho', 'composition', 'rounding')


def label(config):
    """Unique name of a configuration, e.g. blocks_rho1_exponential_sweep or ..._columns3."""
    parts = ['fulljoint' if config.get('full_joint') else config['structure'],
             f"rho{config['rho']:g}", config['composition'], config['rounding']]
    return '_'.join(parts + [f'{key}{value}' for key, value in config.items() if key not in NAMED])


def flags(config):
    """The runner flags of a configuration."""
    result = []
    for key, value in config.items():
        flag = '--' + key.replace('_', '-')
        result += [flag] if value is True else [flag, str(value)]
    return result


def recorded(path):
    """Last status of every run in a runs.jsonl, by name. A torn last line is ignored."""
    status = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            status[row['name']] = row['status']
    return status


def call(command, log, timeout=None):
    """Run a command with its output appended to log. Returns its exit code, or 'timeout'."""
    with open(log, 'a', encoding='utf-8') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException as stop:  # timeout, Ctrl+C or kill: take the workers down too
            if hasattr(os, 'killpg'):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], capture_output=True)
            process.wait()
            if not isinstance(stop, subprocess.TimeoutExpired):
                raise
            code = 'timeout'
        finally:
            # A killed run leaves its spill behind (DataHandler.initialize_directories).
            for kind in ('spill', 'microdata'):
                shutil.rmtree(common.DATA / 'data_cache' / f'topdown_{kind}_{process.pid}',
                              ignore_errors=True)
    return code


def run(dataset, config, name, rep, args):
    """One repetition: the driver, then the metrics. Returns its line for runs.jsonl."""
    files = {suffix: Path(common.out_path(dataset, name + suffix))
             for suffix in ('.json', '_metrics.json', '.log', '.csv')}
    for path in files.values():
        path.unlink(missing_ok=True)  # nothing stale from an earlier attempt
    command = [sys.executable, '-m', f'benchmarks.{dataset}.driver', '--name', name,
               '--workers', str(args.workers)] + flags(config)
    if args.sample:
        command.append('--sample')
    code = call(command, files['.log'], args.timeout and args.timeout * 3600)

    row = {'name': name, 'rep': rep, 'config': config, 'status': 'error'}
    if files['.json'].exists():
        row.update(json.loads(files['.json'].read_text(encoding='utf-8')))
    if code != 0:
        row.update(status='timeout' if code == 'timeout' else 'error', returncode=code)
    else:
        code = call([sys.executable, '-m', 'benchmarks.metrics', dataset, name], files['.log'])
        if code == 0:
            metrics = json.loads(files['_metrics.json'].read_text(encoding='utf-8'))
            row.update(levels=metrics['levels'], workload=metrics['workload'],
                       metrics_seconds=metrics['seconds'])
        else:
            row.update(status='metrics_error', returncode=code)
    if not args.keep_csv:
        files['.csv'].unlink(missing_ok=True)
    row['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    return row


def main():
    parser = argparse.ArgumentParser(description='Run experiments round by round, resumably.')
    parser.add_argument('experiments', nargs='+', choices=sorted(EXPERIMENTS))
    parser.add_argument('--reps', type=int, default=5, help='rounds, one run of every configuration each')
    parser.add_argument('--datasets', nargs='+', help='only these datasets')
    parser.add_argument('--sample', action='store_true', help='use the small subsets')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--timeout', type=float, help='hours before a run is killed')
    parser.add_argument('--retry', action='store_true', help='run again what failed or timed out')
    parser.add_argument('--keep-csv', action='store_true', help='keep the synthetic microdata')
    parser.add_argument('--dry-run', action='store_true', help='list the pending runs and exit')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit('terminated'))

    configurations, seen = [], set()
    for experiment in args.experiments:
        for dataset, config in EXPERIMENTS[experiment]():
            if (dataset, label(config)) not in seen and dataset in (args.datasets or [dataset]):
                seen.add((dataset, label(config)))
                configurations.append((dataset, config))
    datasets = list(dict.fromkeys(dataset for dataset, _ in configurations))

    missing = [path for dataset in datasets
               for path in (common.parquet(dataset, args.sample),
                            common.parquet(dataset, args.sample).replace('.parquet', '_nodes.json'))
               if not os.path.exists(path)]
    if missing:
        sys.exit(f'Missing {missing}: run the prepare.py and columns.py of those datasets first.')

    suffix = '_sample' if args.sample else ''
    results = {dataset: Path(common.out_path(dataset, f'runs{suffix}.jsonl')) for dataset in datasets}
    status = {dataset: recorded(path) for dataset, path in results.items()}
    queue = [(dataset, config, f'{label(config)}_r{rep}{suffix}', rep)
             for rep in range(1, args.reps + 1) for dataset, config in configurations]
    pending = [item for item in queue if status[item[0]].get(item[2]) is None
               or (args.retry and status[item[0]][item[2]] != 'ok')]
    print(f'{len(configurations)} configurations x {args.reps} rounds = {len(queue)} runs, '
          f'{len(queue) - len(pending)} skipped, {len(pending)} to run', flush=True)
    if args.dry_run:
        for dataset, config, name, rep in pending:
            print(f'  {dataset:<16} {name}')
        return

    for index, (dataset, config, name, rep) in enumerate(pending, 1):
        started = time.perf_counter()
        row = run(dataset, config, name, rep, args)
        with open(results[dataset], 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        print(f'[{index}/{len(pending)}] {time.strftime("%H:%M")} {dataset} {name}: '
              f'{row["status"]} in {time.perf_counter() - started:.0f} s', flush=True)


if __name__ == '__main__':
    main()
