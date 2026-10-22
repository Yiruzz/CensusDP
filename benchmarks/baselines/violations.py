"""Records of a synthetic release that break the dataset's declared edit rules.

    python -m benchmarks.baselines.violations chilean_census flat_mst_rho1_r1 [--original]

TopDown cannot produce such a record: every logical rule is folded into its cell space
(constraints/domain_restriction.py). A flat baseline only sees the rules if it enforces structural
zeros, which the upstream AIM and MST do not. Each rule is evaluated with the same reduce() that
builds TopDown's restriction, over the cells of its own scope, so both sides apply one definition.
--original evaluates the original file instead, which has to give zero (the check of the check).
Writes data/out/<dataset>/<name>_violations.json; runs in TopDown's venv.
"""

import argparse
import importlib
import json

import duckdb
import numpy as np
import pandas as pd

from benchmarks import common
from constraints.logical_expressions.base import LogicalExpression
from domain import ContingencyDomain


def count(path, original, domains, rules):
    """Per rule, the records breaking it; plus the records breaking at least one."""
    columns = sorted(set().union(*(rule.scope() for rule in rules)))
    con = duckdb.connect()
    if path == original:
        source = f"read_parquet('{original}')"
    else:
        described = con.execute(f"DESCRIBE SELECT {', '.join(columns)} "
                                f"FROM read_parquet('{original}')").fetchall()
        types = ', '.join(f"'{row[0]}': '{row[1]}'" for row in described)
        source = (f"read_csv('{path}', delim=';', header=true, types={{{types}}}, "
                  f"allow_quoted_nulls=false)")
    records = con.execute(f"SELECT {', '.join(columns)} FROM {source}").fetchnumpy()
    total = len(next(iter(records.values())))
    broken = np.zeros(total, dtype=bool)
    per_rule = []
    for rule in rules:
        scope = sorted(rule.scope())
        domain = ContingencyDomain(scope, {column: domains[column] for column in scope})
        allowed = rule.reduce(domain)
        # A DataFrame, not a dict: encode() takes the record count from len(data).
        cells = domain.encode(pd.DataFrame({column: np.asarray(records[column]).astype(
            domain.domains[column].dtype) for column in scope}))
        bad = ~allowed[cells]
        broken |= bad
        per_rule.append({'rule': repr(rule), 'scope': scope, 'records': int(bad.sum())})
    return {'records': total, 'violating': int(broken.sum()), 'share': float(broken.mean()),
            'rules': per_rule}


def main():
    parser = argparse.ArgumentParser(description='Edit-rule violations of one finished run.')
    parser.add_argument('dataset')
    parser.add_argument('name', help='the run, as in data/out/<dataset>/<name>.json')
    parser.add_argument('--original', action='store_true', help='evaluate the original file')
    args = parser.parse_args()

    with open(common.out_path(args.dataset, f'{args.name}.json'), encoding='utf-8') as handle:
        run = json.load(handle)
    declared = importlib.import_module(f'benchmarks.{args.dataset}.domains')
    constraints = importlib.import_module(f'benchmarks.{args.dataset}.constraints')
    domains = {column: declared.DOMAINS[column] for column in run['columns']}
    rules = [rule for rule in constraints.constraints(run['columns'])
             if isinstance(rule, LogicalExpression)]
    original = common.parquet(args.dataset, run['sample'])
    path = original if args.original else common.out_path(args.dataset, f'{args.name}.csv')
    result = count(path, original, domains, rules) if rules else {
        'records': None, 'violating': 0, 'share': 0.0, 'rules': []}
    suffix = '_violations_original' if args.original else '_violations'
    with open(common.out_path(args.dataset, f'{args.name}{suffix}.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps({key: result[key] for key in ('records', 'violating', 'share')}))


if __name__ == '__main__':
    main()
