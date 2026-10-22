"""Run a flat baseline synthesizer (AIM or MST) on one benchmark dataset, geography as a column.

    <mbi venv>/python -m benchmarks.baselines.run chilean_census mst
    python -m benchmarks.metrics chilean_census flat_mst_rho1_d5
    python -m benchmarks.baselines.violations chilean_census flat_mst_rho1_d5

The baselines have no notion of a hierarchy. The path hierarchy[:depth] becomes one categorical
column GEO, with one value per node of that level (the node list is public, as TopDown's tree
is), and every coarser level is a sum over it, so the release is consistent by construction.
The default depth is the whole hierarchy: the leaves, the same release TopDown publishes.
The number of synthetic records is the true total, which bounded DP makes public.

Writes data/out/<dataset>/<name>.csv and <name>.json in runner.py's format, so
benchmarks/metrics.py evaluates it unchanged, from the national level down to `depth`. There
`bags` holds the cliques the baseline measured (GEO dropped), so "inside" means measured by it.
Run it with common.mbi_python(), the mbi checkout's venv: this module never imports TopDown.
benchmarks/orchestrate.py does that for the configurations of exp8.
"""

import argparse
import importlib
import itertools
import json
import os
import sys
import time
import traceback
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from benchmarks import common

GEO = 'GEO'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('mechanism', choices=('aim', 'mst'))
    parser.add_argument('--depth', type=int, choices=range(1, 6),
                        help='hierarchy levels folded into the GEO column (default: all)')
    parser.add_argument('--rho', type=float, default=1.0,
                        help='total rho-zCDP budget, replace-one neighbours')
    parser.add_argument('--sample', action='store_true', help='use the small subset')
    parser.add_argument('--columns', type=int, help='use only the first N columns')
    parser.add_argument('--max-model-size', type=float, default=80,
                        help='AIM: model size cap in MB; 80 is the value for official comparisons')
    parser.add_argument('--max-iters', type=int, default=1000, help='AIM: mirror descent iterations')
    parser.add_argument('--rounds', type=int, help='AIM: rounds (default 16 x attributes)')
    parser.add_argument('--seed', type=int,
                        help='random seed; drawn and recorded when omitted, so rounds differ')
    parser.add_argument('--name', help='output name; built from the flags when omitted')
    return parser.parse_args()


def default_name(args):
    parts = ['flat', args.mechanism, f'rho{args.rho:g}', f'd{args.depth}']
    if args.mechanism == 'aim' and args.max_model_size != 80:
        parts.append(f'mms{args.max_model_size:g}')
    if args.columns:
        parts.append(f'c{args.columns}')
    if args.sample:
        parts.append('sample')
    return '_'.join(parts)


def load(path, domains, hierarchy):
    """Every record encoded as domain indices, plus the GEO column and its node table."""
    con = duckdb.connect()
    source = f"read_parquet('{path}')"
    order = ', '.join(hierarchy)
    columns = ', '.join(domains)
    geo = f', DENSE_RANK() OVER (ORDER BY {order}) - 1 AS {GEO}' if hierarchy else ''
    raw = con.execute(f'SELECT {columns}{geo} FROM {source}').fetchnumpy()
    encoded = {}
    for column, values in domains.items():
        domain = np.asarray(values)
        value = np.asarray(raw.pop(column)).astype(domain.dtype)
        index = np.searchsorted(domain, value)
        if (index >= len(domain)).any() or (domain[np.minimum(index, len(domain) - 1)] != value).any():
            raise ValueError(f'{column} has values outside its declared domain')
        encoded[column] = index.astype(np.int32)
    nodes = None
    if hierarchy:
        encoded[GEO] = np.asarray(raw.pop(GEO)).astype(np.int32)
        nodes = con.execute(f'SELECT DISTINCT {order} FROM {source} ORDER BY ALL').df()
    return encoded, nodes


def decode(synthetic, domains, hierarchy, nodes):
    """The synthetic records back in the original values, hierarchy columns first."""
    frame = {}
    if hierarchy:
        geo = np.asarray(synthetic[GEO])
        for column in hierarchy:
            frame[column] = nodes[column].to_numpy()[geo]
    for column, values in domains.items():
        frame[column] = np.asarray(values)[np.asarray(synthetic[column])]
    return pd.DataFrame(frame)


def peak_memory_mb():
    """Peak resident memory of this process: peak working set on Windows, max RSS elsewhere."""
    if sys.platform == 'win32':
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**20
    import resource
    scale = 2**20 if sys.platform == 'darwin' else 2**10
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def main():
    args = parse_args()
    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(4), 'little')
    np.random.seed(args.seed)  # upstream draws from the global generator
    source = common.parquet(args.dataset, args.sample)
    saved = json.loads(Path(source.replace('.parquet', '_nodes.json')).read_text())
    if args.depth is None:
        args.depth = len(saved['hierarchy'])
    hierarchy = saved['hierarchy'][:args.depth]
    declared = importlib.import_module(f'benchmarks.{args.dataset}.domains')
    columns = list(declared.COLUMNS[:args.columns] if args.columns else declared.COLUMNS)
    domains = {column: declared.DOMAINS[column] for column in columns}
    name = args.name or default_name(args)
    record = {
        'dataset': args.dataset, 'name': name, 'sample': args.sample,
        'records': saved['records'], 'hierarchy': hierarchy,
        'nodes_per_level': saved['nodes'][:args.depth + 1], 'columns': columns,
        'mode': 'baseline', 'mechanism': args.mechanism, 'rho': args.rho,
        'neighbours': 'replace-one', 'seed': args.seed, 'status': 'error', 'seconds': {},
    }
    if args.mechanism == 'aim':
        record.update(max_model_size=args.max_model_size, max_iters=args.max_iters,
                      rounds=args.rounds)

    try:
        from benchmarks.baselines import mechanisms  # first: it configures JAX before mbi loads
        from mbi import Dataset, Domain

        started = time.perf_counter()
        encoded, nodes = load(source, domains, hierarchy)
        sizes = {column: len(values) for column, values in domains.items()}
        if hierarchy:
            sizes[GEO] = len(nodes)
        data = Dataset(encoded, Domain.fromdict(sizes))
        record['seconds']['load'] = time.perf_counter() - started
        record['domain_sizes'] = sizes

        started = time.perf_counter()
        if args.mechanism == 'aim':
            workload = [(cl, 1.0) for cl in itertools.combinations(data.domain.attrs, 2)]
            mechanism = mechanisms.AIM(args.rho, max_model_size=args.max_model_size,
                                       max_iters=args.max_iters, rounds=args.rounds)
            _, synthetic = mechanism.run(data, workload, num_synth_rows=saved['records'])
            measured = mechanism.measured
        else:
            synthetic, measured = mechanisms.mst(data, args.rho, num_synth_rows=saved['records'])
        record['seconds']['synthesis'] = time.perf_counter() - started
        record['measured'] = [list(cl) for cl in measured]
        record['bags'] = [[c for c in cl if c != GEO] for cl in measured
                          if len([c for c in cl if c != GEO]) > 1]

        started = time.perf_counter()
        decode(synthetic.to_dict(), domains, hierarchy, nodes).to_csv(
            common.out_path(args.dataset, f'{name}.csv'), sep=';', index=False)
        record['seconds']['write'] = time.perf_counter() - started
        record['memory_mb'] = peak_memory_mb()
        record['status'] = 'ok'
    except Exception:
        record['error'] = traceback.format_exc()
        raise
    finally:
        with open(common.out_path(args.dataset, f'{name}.json'), 'w', encoding='utf-8') as handle:
            json.dump(record, handle, indent=1)

    print(json.dumps({key: record.get(key) for key in ('name', 'status', 'seconds', 'memory_mb')}))


if __name__ == '__main__':
    main()
