"""Utility of a synthetic release, compared node by node at every level of the hierarchy.

Runs in its own process, after the run, so it never touches the time or memory the run records:

    python -m benchmarks.metrics sinasc blocks_rho1_exponential_d4 [--triples 100] [--discard]

For every marginal, the original and synthetic distributions are compared inside each node of
a level (TVD, and L1 over counts) and averaged over the nodes of that level. Pairs are split
into those inside some junction-tree bag and the rest; triples into those contained in a bag
and the rest. Pairs also carry their national symmetric uncertainty U in both files.
Writes data/out/<dataset>/<name>_metrics.json.
"""

import argparse
import importlib
import itertools
import json
import math
import os
import random
import time

import duckdb
import numpy as np

from benchmarks import common

DENSE_LIMIT = 5_000_000


def evaluate(original, synthetic_csv, domains, hierarchy, bags, triples=100, workload=()):
    """Compare a synthetic CSV written by TopDown against the original Parquet.

    Args:
        original: The Parquet the run read.
        synthetic_csv: The CSV the run wrote.
        domains: Declared domain of every query column.
        hierarchy: The hierarchy columns the run used.
        bags: Junction-tree bags; [all columns] for a full-joint run.
        triples: How many 3-way marginals to evaluate, drawn once with a fixed seed.
        workload: Extra marginals to report on their own (e.g. the DAS queries).

    Returns:
        dict: 'levels' (averages per level), 'pairs', 'triples' and 'workload' (per marginal).
    """
    columns = list(domains)
    sizes = {column: len(values) for column, values in domains.items()}
    leaves, parents = _tree(original, hierarchy)
    synthetic = _to_parquet(synthetic_csv, original, list(hierarchy) + columns)
    try:
        data = [_encode(path, domains, hierarchy, leaves) for path in (original, synthetic)]
    finally:
        os.remove(synthetic)
    bag_sets = [set(bag) for bag in bags]

    pairs = []
    for pair in itertools.combinations(columns, 2):
        tvd, l1, tables = _compare(data, pair, sizes, parents, national=True)
        shape = (sizes[pair[0]], sizes[pair[1]])
        pairs.append({'columns': list(pair), 'inside': any(set(pair) <= bag for bag in bag_sets),
                      'tvd': tvd, 'l1': l1,
                      'u_original': _uncertainty(tables[0].reshape(shape)),
                      'u_synthetic': _uncertainty(tables[1].reshape(shape))})

    chosen = list(itertools.combinations(columns, 3))
    if len(chosen) > triples:
        chosen = sorted(random.Random(0).sample(chosen, triples))
    triple_rows = []
    for triple in chosen:
        tvd, l1, _ = _compare(data, triple, sizes, parents)
        triple_rows.append({'columns': list(triple), 'tvd': tvd, 'l1': l1,
                            'contained': any(set(triple) <= bag for bag in bag_sets)})

    workload_rows = []
    for marginal in workload:
        tvd, l1, _ = _compare(data, tuple(marginal), sizes, parents)
        workload_rows.append({'columns': list(marginal), 'tvd': tvd, 'l1': l1})

    levels = []
    for level, column in enumerate(['national'] + list(hierarchy)):
        def mean(rows, keep=lambda row: True, key='tvd'):
            values = [row[key][level] for row in rows if keep(row)]
            return float(np.mean(values)) if values else None
        levels.append({
            'level': level, 'column': column, 'nodes': int(parents[level].max()) + 1,
            'tvd2': mean(pairs),
            'tvd2_inside': mean(pairs, lambda row: row['inside']),
            'tvd2_outside': mean(pairs, lambda row: not row['inside']),
            'tvd3': mean(triple_rows),
            'tvd3_contained': mean(triple_rows, lambda row: row['contained']),
            'tvd3_other': mean(triple_rows, lambda row: not row['contained']),
            'workload_l1': mean(workload_rows, key='l1'),
        })
    return {'levels': levels, 'pairs': pairs, 'triples': triple_rows, 'workload': workload_rows}


def _tree(original, hierarchy):
    """The leaves (distinct hierarchy paths) and, per level, the node index of every leaf."""
    keys = ', '.join(f'CAST({column} AS VARCHAR) AS {column}' for column in hierarchy)
    leaves = duckdb.connect().execute(
        f"SELECT DISTINCT {keys} FROM read_parquet('{original}') ORDER BY ALL").df()
    leaves['leaf'] = np.arange(len(leaves), dtype=np.int64)
    parents = [np.zeros(len(leaves), dtype=np.int64)]
    for level in range(1, len(hierarchy) + 1):
        parents.append(leaves.groupby(list(hierarchy[:level]), sort=False).ngroup().to_numpy())
    return leaves, parents


def _to_parquet(csv, original, columns):
    """The synthetic CSV as a Parquet typed like the original, reading '' back as ''."""
    con = duckdb.connect()
    described = con.execute(f"DESCRIBE SELECT {', '.join(columns)} "
                            f"FROM read_parquet('{original}')").fetchall()
    types = ', '.join(f"'{row[0]}': '{row[1]}'" for row in described)
    target = csv[:-len('.csv')] + '_evaluation.parquet'
    con.execute(f"COPY (SELECT {', '.join(columns)} FROM read_csv('{csv}', delim=';', "
                f"header=true, types={{{types}}}, allow_quoted_nulls=false)) "
                f"TO '{target}' (FORMAT PARQUET)")
    return target


def _encode(path, domains, hierarchy, leaves):
    """Leaf index and domain code of every record, in file order."""
    con = duckdb.connect()
    con.register('leaves', leaves)
    match = ' AND '.join(f'CAST(t.{column} AS VARCHAR) = l.{column}' for column in hierarchy)
    leaf = con.execute(f"SELECT l.leaf FROM read_parquet('{path}', file_row_number=true) t "
                       f"LEFT JOIN leaves l ON {match} ORDER BY t.file_row_number"
                       ).fetchnumpy()['leaf']
    if np.ma.is_masked(leaf):
        raise ValueError(f'{path} has records outside the hierarchy of the original data')
    data = {'leaf': np.asarray(leaf, dtype=np.int64), 'n_leaves': len(leaves)}
    for column, values in domains.items():
        domain = np.asarray(values)
        raw = np.asarray(con.execute(f"SELECT {column} FROM read_parquet('{path}')"
                                     ).fetchnumpy()[column])
        data[column] = np.searchsorted(domain, raw.astype(domain.dtype)).astype(np.int16)
    return data


def _compare(data, marginal, sizes, parents, national=False):
    """TVD and L1 of one marginal at every level, plus both national tables when asked."""
    cells = math.prod(sizes[column] for column in marginal)
    counts = [_leaf_counts(d, marginal, sizes, cells) for d in data]
    tvd, l1 = zip(*(_distance(counts, parent, cells) for parent in parents))
    tables = None
    if national:
        tables = [np.bincount(cell, weights=count, minlength=cells) for _, cell, count in counts]
    return list(tvd), list(l1), tables


def _leaf_counts(data, marginal, sizes, cells):
    """Sparse (leaf, cell, count) arrays of one marginal."""
    cell = np.zeros(len(data['leaf']), dtype=np.int64)
    for column in marginal:
        cell = cell * sizes[column] + data[column]
    key = data['leaf'] * cells + cell
    if data['n_leaves'] * cells <= DENSE_LIMIT:
        counts = np.bincount(key, minlength=data['n_leaves'] * cells)
        key = np.flatnonzero(counts)
        counts = counts[key]
    else:
        key, counts = np.unique(key, return_counts=True)
    return key // cells, key % cells, counts


def _distance(counts, parent, cells):
    """Mean over the nodes of one level of TVD and L1 between original and synthetic."""
    n_nodes = int(parent.max()) + 1
    (key_o, count_o, total_o), (key_s, count_s, total_s) = [
        (parent[leaf] * cells + cell, count,
         np.bincount(parent[leaf], weights=count, minlength=n_nodes))
        for leaf, cell, count in counts]
    safe_s = np.where(total_s > 0, total_s, 1)
    if n_nodes * cells <= DENSE_LIMIT:
        size = n_nodes * cells
        table_o = np.bincount(key_o, weights=count_o, minlength=size).reshape(n_nodes, cells)
        table_s = np.bincount(key_s, weights=count_s, minlength=size).reshape(n_nodes, cells)
        tvd = 0.5 * np.abs(table_o / total_o[:, None] - table_s / safe_s[:, None]).sum(axis=1)
        l1 = np.abs(table_o - table_s).sum(axis=1)
    else:
        keys, inverse = np.unique(np.concatenate([key_o, key_s]), return_inverse=True)
        table_o = np.bincount(inverse[:len(key_o)], weights=count_o, minlength=len(keys))
        table_s = np.bincount(inverse[len(key_o):], weights=count_s, minlength=len(keys))
        node = keys // cells
        difference = np.abs(table_o / total_o[node] - table_s / safe_s[node])
        tvd = 0.5 * np.bincount(node, weights=difference, minlength=n_nodes)
        l1 = np.bincount(node, weights=np.abs(table_o - table_s), minlength=n_nodes)
    tvd[total_s == 0] = 1.0
    return float(tvd.mean()), float(l1.mean())


def _uncertainty(table):
    """Symmetric uncertainty 2 I(A;B) / (H(A) + H(B)) of a 2-way count table."""
    p = table / table.sum()

    def entropy(q):
        q = q[q > 0]
        return float(-(q * np.log(q)).sum())

    h_a, h_b = entropy(p.sum(axis=1)), entropy(p.sum(axis=0))
    return 2 * (h_a + h_b - entropy(p.ravel())) / (h_a + h_b) if h_a + h_b > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Utility metrics of one finished run.')
    parser.add_argument('dataset')
    parser.add_argument('name', help='the run, as in data/out/<dataset>/<name>.json')
    parser.add_argument('--triples', type=int, default=100, help='3-way marginals to evaluate')
    parser.add_argument('--discard', action='store_true', help='delete the synthetic CSV afterwards')
    args = parser.parse_args()

    with open(common.out_path(args.dataset, f'{args.name}.json'), encoding='utf-8') as handle:
        run = json.load(handle)
    if run['status'] != 'ok':
        raise ValueError(f'{args.name} did not finish: {run.get("error", "")[-300:]}')
    declared = importlib.import_module(f'benchmarks.{args.dataset}.domains')
    marginals = importlib.import_module(f'benchmarks.{args.dataset}.marginals')
    domains = {column: declared.DOMAINS[column] for column in run['columns']}
    workload = [w for w in getattr(marginals, 'WORKLOAD', []) if set(w) <= set(run['columns'])]
    csv = common.out_path(args.dataset, f'{args.name}.csv')

    started = time.perf_counter()
    result = evaluate(common.parquet(args.dataset, run['sample']), csv, domains, run['hierarchy'],
                      run['bags'] or [run['columns']], args.triples, workload)
    result['seconds'] = time.perf_counter() - started
    with open(common.out_path(args.dataset, f'{args.name}_metrics.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(result, handle, indent=1)
    if args.discard:
        os.remove(csv)
    for level in result['levels']:
        print({key: round(value, 6) if isinstance(value, float) else value
               for key, value in level.items()})


if __name__ == '__main__':
    main()
