"""The utility of copying each parent's distribution, a reference for the error below the root.

    python -m benchmarks.parent_reference sinasc
    python -m benchmarks.parent_reference sinasc --released blocks_rho1_sqrt_sweep_triples100_keep1_r1

The reference gives every node the distribution of its parent, scaled to the node's population,
and is scored against the node's true marginal. It comes in two versions:

- By default the parent's distribution and the node's population are the true ones. It reads only
  the original data and spends no budget, but it is not private: it states how much a node differs
  from its parent, a property of the data, not of any method.
- With --released they are read from that run's synthetic CSV, so the copy is post-processing of
  the release, private at the same rho, and what any user of the release could compute. A release
  whose node TVD is above it gains nothing over what it already says about the parent. The CSV is
  kept only by the exp4_parent runs (benchmarks/experiments.py).

For a node with true counts n (population N), copy N' b, with b the parent's distribution and N'
the node's population in the source, only the cells the node occupies matter:
TVD = 1 - sum_c min(n_c / N, b_c) and L1 = N + N' - 2 sum_c min(n_c, N' b_c). Both are averaged over
the nodes of a level, as benchmarks/metrics.py does. The root has no parent, so its entries are None.

The pairs and triples are those of the scored run (the dataset's exp4 run by default, the
--released one otherwise), with its `inside` flags and triple weights, so the averages are over the
same marginals as the figures. Writes data/out/<dataset>/parent_reference.json, or
parent_reference_released.json, in the metrics' format: 'levels' with the same fields plus the L1
of every group, and per-marginal 'pairs' and 'triples'.
"""

import argparse
import importlib
import json
import math
import os
import time

import numpy as np

from benchmarks import common, metrics
from benchmarks.experiments import TRIPLES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--run', help='the exp4 run whose pairs and triples are scored '
                                      '(default: <DEFAULT>_rho1_sqrt_sweep_triples100_r1)')
    parser.add_argument('--released', metavar='NAME',
                        help='copy the parent of this run\'s release, from its synthetic CSV')
    return parser.parse_args()


def parent_copy(original, source, marginal, sizes, parents):
    """TVD and L1 of the parent copy for one marginal, per level (None at the root). The parent's
    distribution and the node's population come from `source`, the original data or a release."""
    cells = math.prod(sizes[column] for column in marginal)
    leaf, cell, count = metrics._leaf_counts(original, marginal, sizes, cells)
    source_leaf, source_cell, source_count = (
        (leaf, cell, count) if source is original else metrics._leaf_counts(source, marginal, sizes, cells))
    tvd, l1 = [None], [None]
    for level in range(1, len(parents)):
        n_nodes = int(parents[level].max()) + 1
        n_up = int(parents[level - 1].max()) + 1
        node, node_count = _aggregate(parents[level][leaf], cell, count, cells)
        up, up_count = _aggregate(parents[level - 1][source_leaf], source_cell, source_count, cells)
        # The parent of every node of this level, read off any leaf below it.
        parent_of = np.zeros(n_nodes, dtype=np.int64)
        parent_of[parents[level]] = parents[level - 1]
        total = np.bincount(node // cells, weights=node_count, minlength=n_nodes)
        copied = np.bincount(parents[level][source_leaf], weights=source_count, minlength=n_nodes)
        up_total = np.bincount(up // cells, weights=up_count, minlength=n_up)
        owner = node // cells
        # The parent's count in the same cell. In the original data every occupied cell of a node
        # is occupied in its parent; in a release it may not be, and then the copy has 0 there.
        wanted = parent_of[owner] * cells + node % cells
        position = np.minimum(np.searchsorted(up, wanted), len(up) - 1)
        found = up[position] == wanted
        parent_total = up_total[parent_of[owner]]
        share = np.where(found & (parent_total > 0), up_count[position] / np.maximum(parent_total, 1), 0.0)
        node_tvd = 1 - np.bincount(owner, weights=np.minimum(node_count / total[owner], share),
                                   minlength=n_nodes)
        overlap = np.bincount(owner, weights=np.minimum(node_count, copied[owner] * share),
                              minlength=n_nodes)
        tvd.append(float(node_tvd.mean()))
        l1.append(float((total + copied - 2 * overlap).mean()))
    return tvd, l1


def _aggregate(node, cell, count, cells):
    """Sorted unique keys node * cells + cell and their summed counts."""
    keys, inverse = np.unique(node * cells + cell, return_inverse=True)
    return keys, np.bincount(inverse, weights=count, minlength=len(keys))


def main():
    args = parse_args()
    marginals = importlib.import_module(f'benchmarks.{args.dataset}.marginals')
    name = args.released or args.run or f'{marginals.DEFAULT}_rho1_sqrt_sweep_triples{TRIPLES}_r1'
    # runs.jsonl, not <name>.json: only the former is brought back from the server.
    with open(common.out_path(args.dataset, 'runs.jsonl'), encoding='utf-8') as handle:
        run = next(record for record in map(json.loads, handle) if record['name'] == name)
    with open(common.out_path(args.dataset, f'{name}_metrics.json'), encoding='utf-8') as handle:
        scored = json.load(handle)
    declared = importlib.import_module(f'benchmarks.{args.dataset}.domains')
    domains = {column: declared.DOMAINS[column] for column in run['columns']}
    sizes = {column: len(values) for column, values in domains.items()}
    original = common.parquet(args.dataset, run['sample'])

    started = time.perf_counter()
    leaves, parents = metrics._tree(original, run['hierarchy'])
    data = metrics._encode(original, domains, run['hierarchy'], leaves)
    source = data
    if args.released:
        synthetic = metrics._to_parquet(common.out_path(args.dataset, f'{name}.csv'), original,
                                        list(run['hierarchy']) + list(domains))
        try:
            source = metrics._encode(synthetic, domains, run['hierarchy'], leaves)
        finally:
            os.remove(synthetic)
    pairs, triples = [], []
    for rows, out in ((scored['pairs'], pairs), (scored['triples'], triples)):
        for row in rows:
            tvd, l1 = parent_copy(data, source, row['columns'], sizes, parents)
            out.append({**{key: value for key, value in row.items() if key not in ('tvd', 'l1', 'u_synthetic')},
                        'tvd': tvd, 'l1': l1})

    levels = []
    for level, column in enumerate(['national'] + list(run['hierarchy'])):
        def mean(rows, keep=lambda row: True, key='tvd'):
            kept = [row for row in rows if keep(row)]
            if not kept or kept[0][key][level] is None:
                return None
            return float(np.average([row[key][level] for row in kept],
                                    weights=[row.get('weight', 1) for row in kept]))
        groups = {
            '2': (pairs, lambda row: True),
            '2_inside': (pairs, lambda row: row['inside']),
            '2_outside': (pairs, lambda row: not row['inside']),
            '3': (triples, lambda row: True),
            '3_contained': (triples, lambda row: row['contained']),
            '3_other': (triples, lambda row: not row['contained']),
            **{f'3_pairs{k}': (triples, lambda row, k=k: not row['contained']
                               and row['measured_pairs'] == k) for k in range(3)},
        }
        entry = {'level': level, 'column': column, 'nodes': int(parents[level].max()) + 1}
        for suffix, (rows, keep) in groups.items():
            entry[f'tvd{suffix}'] = mean(rows, keep)
            entry[f'l1_{suffix}'] = mean(rows, keep, key='l1')
        levels.append(entry)

    result = {'dataset': args.dataset, 'scored_like': name, 'released': bool(args.released),
              'levels': levels, 'pairs': pairs, 'triples': triples,
              'seconds': time.perf_counter() - started}
    output = 'parent_reference_released.json' if args.released else 'parent_reference.json'
    with open(common.out_path(args.dataset, output), 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=1)
    for entry in levels:
        print({key: round(value, 4) if isinstance(value, float) else value
               for key, value in entry.items() if key in ('column', 'nodes', 'tvd2', 'tvd3', 'l1_3')})


if __name__ == '__main__':
    main()
