"""Helpers shared by every benchmark package. Run everything from the repository root."""

import json
import math
from pathlib import Path

import duckdb

DATA = Path('data')

SOLVER_OPTIONS = {'OutputFlag': 0, 'Threads': 1, 'MIPGap': 5e-3, 'TimeLimit': 14400}

COMPOSITIONS = ('exponential', 'uniform', 'proportional', 'sqrt')


def parquet(dataset, sample=False):
    """data/<dataset>/<dataset>.parquet, or its _sample counterpart."""
    suffix = '_sample' if sample else ''
    return (DATA / dataset / f'{dataset}{suffix}.parquet').as_posix()


def out_path(dataset, name):
    """data/out/<dataset>/<name>, creating the folder (TopDown does not)."""
    folder = DATA / 'out' / dataset
    folder.mkdir(parents=True, exist_ok=True)
    return (folder / name).as_posix()


def write_full(dataset, source, columns):
    """Copy `columns` of the source Parquet into data/<dataset>/<dataset>.parquet."""
    selection = ', '.join(columns)
    duckdb.connect().execute(f"COPY (SELECT {selection} FROM read_parquet('{source}')) "
                             f"TO '{parquet(dataset)}' (FORMAT PARQUET)")


def write_sample(dataset, where):
    """Copy the rows of the full Parquet matching `where` into the sample Parquet."""
    source, target = parquet(dataset), parquet(dataset, sample=True)
    con = duckdb.connect()
    con.execute(f"COPY (SELECT * FROM read_parquet('{source}') WHERE {where}) "
                f"TO '{target}' (FORMAT PARQUET)")
    for path in (source, target):
        rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
        print(f'{path}: {rows:,} records')


def nodes_per_level(path, hierarchy):
    """Record count, and node count of every tree level (root first)."""
    con = duckdb.connect()
    source = f"read_parquet('{path}')"
    records = con.execute(f'SELECT COUNT(*) FROM {source}').fetchone()[0]
    nodes = [1] + [con.execute(f'SELECT COUNT(DISTINCT ({", ".join(hierarchy[:level])})) '
                               f'FROM {source}').fetchone()[0]
                   for level in range(1, len(hierarchy) + 1)]
    return records, nodes


def read_nodes(path, hierarchy):
    """The counts columns.py saved next to the Parquet, so a timed run never computes them."""
    saved = Path(path.replace('.parquet', '_nodes.json'))
    if not saved.exists():
        raise FileNotFoundError(f'{saved} not found: run the columns.py of this dataset first '
                                f'(with --sample for the subset).')
    stored = json.loads(saved.read_text())
    if stored['hierarchy'][:len(hierarchy)] != list(hierarchy):
        raise ValueError(f'{saved} was saved for the hierarchy {stored["hierarchy"]}')
    return stored['records'], stored['nodes'][:len(hierarchy) + 1]


def level_rhos(total, nodes, composition):
    """Split a total rho over the tree levels.

    exponential doubles towards the leaves, uniform gives every level the same, proportional
    follows the node count of each level and sqrt its square root. The last one minimizes the
    summed noise variance sum_i nodes_i / rho_i, the zCDP counterpart of the geometric
    allocation of Cormode et al. (ICDE 2012).
    """
    weights = {
        'exponential': [2 ** level for level in range(len(nodes))],
        'uniform': [1] * len(nodes),
        'proportional': list(nodes),
        'sqrt': [math.sqrt(n) for n in nodes],
    }[composition]
    return [total * weight / sum(weights) for weight in weights]


def inspect_columns(path, domains, hierarchy):
    """Print and save the node counts per level, and print each column's values outside its domain."""
    records, nodes = nodes_per_level(path, hierarchy)
    Path(path.replace('.parquet', '_nodes.json')).write_text(
        json.dumps({'hierarchy': list(hierarchy), 'records': records, 'nodes': nodes}))
    print(f'{path}: {records:,} records')
    for column, count in zip(hierarchy, nodes[1:]):
        print(f'  {column:<20} {count:>8,} nodes')
    con = duckdb.connect()
    for column, values in domains.items():
        counts = con.execute(f"SELECT {column}, COUNT(*) FROM read_parquet('{path}') "
                             f"GROUP BY 1").fetchall()
        declared = set(values)
        outside = sorted(((value, n) for value, n in counts if value not in declared), key=str)
        status = f'OUTSIDE {outside[:8]}' if outside else 'ok'
        if list(values) != sorted(values):
            status = 'NOT SORTED'
        print(f'  {column:<20} {len(counts):>5} of {len(declared):<5} values  {status}')
