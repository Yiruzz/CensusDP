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
import threading
import time
from pathlib import Path

from benchmarks import common
from benchmarks.experiments import EXPERIMENTS

NAMED = ('full_joint', 'structure', 'rho', 'composition', 'rounding')
METRIC_FLAGS = ('triples',)
# Read by the orchestrator itself, never passed to the driver or the metrics.
ORCHESTRATOR_FLAGS = ('memory',)


def machine_mb():
    """MemTotal - MemAvailable in MB: the machine-wide instrument of benchmarks/ipums_1940/campaign.sh."""
    values = {}
    with open('/proc/meminfo', encoding='ascii') as handle:
        for line in handle:
            key, value = line.split(':', 1)
            values[key] = int(value.split()[0])
    return (values['MemTotal'] - values['MemAvailable']) / 1024


def session_pss_mb(session):
    """Sum of Pss + SwapPss in MB over every process of a session.

    PSS charges each shared page to the processes mapping it in equal parts, so the copy-on-write
    pages the forked workers share with the main process are counted once, where a sum of RSS
    would count them once per worker. SwapPss keeps the swapped pages, which RSS hides.
    """
    total = 0
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            with open(f'/proc/{entry}/stat', encoding='ascii', errors='replace') as handle:
                # After the parenthesised command name: state ppid pgrp session ...
                if int(handle.read().rsplit(')', 1)[1].split()[3]) != session:
                    continue
            with open(f'/proc/{entry}/smaps_rollup', encoding='ascii') as handle:
                for line in handle:
                    if line.startswith(('Pss:', 'SwapPss:')):
                        total += int(line.split()[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
            continue  # the process ended between the listing and the read
    return total / 1024


class MemorySampler(threading.Thread):
    """Peak memory of a running process tree, sampled from outside it.

    Records two instruments. The PSS of the run's own session is the one the structures table
    reports: it does not depend on what else the host holds. The machine-wide one is the DAS
    table's, and includes a host baseline of about 5 GB that moves by half a GB between runs, so
    it is kept as a cross-check (peak minus baseline agreed with the PSS within 0.3 GB on the test).

    The first sample is taken at once, then one every `first` seconds for the first `fast` seconds
    and one every `interval` after that: an ADULT run lasts one to four seconds, and a fixed second
    gave it a single sample, while reading smaps_rollup ten times a second on the 20 GB processes of
    IPUMS would cost the run itself. Those processes only grow after the first seconds, when the
    sampling has already slowed down. Linux only.
    """

    def __init__(self, session, interval=1.0, first=0.1, fast=10.0):
        super().__init__(daemon=True)
        self.session, self.interval, self.first, self.fast = session, interval, first, fast
        self.finished = threading.Event()
        self.baseline_machine = machine_mb()
        self.peak_machine = self.peak_pss = 0.0
        self.samples = 0

    def sample(self):
        self.peak_machine = max(self.peak_machine, machine_mb())
        self.peak_pss = max(self.peak_pss, session_pss_mb(self.session))
        self.samples += 1

    def run(self):
        started = time.monotonic()
        self.sample()
        while not self.finished.wait(self.first if time.monotonic() - started < self.fast
                                     else self.interval):
            self.sample()

    def result(self):
        self.finished.set()
        self.join()
        return {'peak_machine_gb': self.peak_machine / 1024,
                'baseline_machine_gb': self.baseline_machine / 1024,
                'peak_session_pss_gb': self.peak_pss / 1024,
                'samples': self.samples, 'interval_s': self.interval, 'first_interval_s': self.first,
                'fast_s': self.fast}


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


def call(command, log, timeout=None, memory=None):
    """Run a command with its output appended to log. Returns its exit code, or 'timeout'.

    With `memory` (a dict), the run's process tree is sampled while it runs and the peaks are
    written into it. The process is started in a session of its own, so its session id is its pid.
    """
    with open(log, 'a', encoding='utf-8') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                                   start_new_session=True)
        sampler = None
        if memory is not None:
            sampler = MemorySampler(process.pid)
            sampler.start()
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
            if sampler is not None:
                memory.update(sampler.result())
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
    driver = {key: value for key, value in config.items()
              if key not in METRIC_FLAGS and key not in ORCHESTRATOR_FLAGS}
    metrics = {key: value for key, value in config.items() if key in METRIC_FLAGS}
    command = [sys.executable, '-m', f'benchmarks.{dataset}.driver', '--name', name,
               '--workers', str(args.workers)] + flags(driver)
    if args.sample:
        command.append('--sample')
    memory = {} if config.get('memory') else None
    code = call(command, files['.log'], args.timeout and args.timeout * 3600, memory)

    row = {'name': name, 'rep': rep, 'config': config, 'status': 'error'}
    if files['.json'].exists():
        row.update(json.loads(files['.json'].read_text(encoding='utf-8')))
    if memory is not None:
        row['memory_sampled'] = memory
    if code != 0:
        row.update(status='timeout' if code == 'timeout' else 'error', returncode=code)
    elif memory is not None:
        row['status'] = 'ok'  # utility is not recomputed: exp6 already holds it
    else:
        code = call([sys.executable, '-m', 'benchmarks.metrics', dataset, name] + flags(metrics),
                    files['.log'])
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
    parser.add_argument('--workers', type=int, default=20)
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
