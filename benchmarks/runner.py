"""Run TopDown on one benchmark dataset and record its cost.

Every driver.py calls main(). A run writes data/out/<dataset>/<name>.csv (the synthetic
microdata) and <name>.json (parameters, sizes, seconds per phase and peak memory). The process
does nothing but the run, so its time and memory are not mixed with any other measurement:
utility is computed afterwards, in its own process, by benchmarks/metrics.py.

    python -m benchmarks.sinasc.driver --rho 0.5 --composition sqrt
    python -m benchmarks.metrics sinasc blocks_rho0.5_sqrt_d4 --discard
"""

import argparse
import json
import sys
import time
import traceback

from constraints.contextual_constraints import SumEqualRealTotal
from constraints.logical_expressions.atomic import TrueExpression
from privacy import ZCDP
from topdown import TopDown

from benchmarks import common

try:
    import resource
except ImportError:  # Windows: peak memory is not recorded
    resource = None


def parse_args(hierarchy, marginals):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help='use the small subset')
    parser.add_argument('--depth', type=int, default=len(hierarchy),
                        choices=range(1, len(hierarchy) + 1), help='hierarchy levels to use')
    parser.add_argument('--columns', type=int, help='use only the first N columns')
    parser.add_argument('--full-joint', action='store_true', help='full-joint instead of marginals')
    parser.add_argument('--structure', default=marginals.DEFAULT,
                        choices=sorted(marginals.STRUCTURES))
    parser.add_argument('--rho', type=float, default=1.0, help='total rho-zCDP budget')
    parser.add_argument('--composition', default='exponential', choices=common.COMPOSITIONS)
    parser.add_argument('--rounding', default='auto', choices=('auto', 'sweep', 'mip'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--check', action='store_true',
                        help='verify that children sum to their parent (adds time)')
    parser.add_argument('--name', help='output name; built from the flags when omitted')
    return parser.parse_args()


def default_name(args):
    parts = ['fulljoint' if args.full_joint else args.structure, f'rho{args.rho:g}',
             args.composition, f'd{args.depth}']
    if args.columns:
        parts.append(f'c{args.columns}')
    if args.rounding != 'auto':
        parts.append(args.rounding)
    if args.sample:
        parts.append('sample')
    return '_'.join(parts)


def peak_memory_mb():
    """Peak resident memory of this process and of its largest finished child (Linux, macOS)."""
    if resource is None:
        return None
    scale = 1024 * 1024 if sys.platform == 'darwin' else 1024
    return {'main': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale,
            'largest_worker': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / scale}


def main(dataset, hierarchy, domains, constraints, marginals):
    args = parse_args(hierarchy, marginals)
    hierarchy = hierarchy[:args.depth]
    columns = list(domains.COLUMNS[:args.columns] if args.columns else domains.COLUMNS)
    source = common.parquet(dataset, args.sample)
    name = args.name or default_name(args)
    records, nodes = common.read_nodes(source, hierarchy)
    rhos = common.level_rhos(args.rho, nodes, args.composition)
    rounding = args.rounding if args.rounding != 'auto' else ('mip' if args.full_joint else 'sweep')
    record = {
        'dataset': dataset, 'name': name, 'sample': args.sample, 'records': records,
        'hierarchy': hierarchy, 'nodes_per_level': nodes, 'columns': columns,
        'mode': 'full_joint' if args.full_joint else 'marginal',
        'structure': None if args.full_joint else args.structure,
        'rho': args.rho, 'composition': args.composition, 'level_rhos': rhos,
        'rounding': rounding, 'workers': args.workers, 'check': args.check,
        'solver_options': common.SOLVER_OPTIONS, 'status': 'error', 'seconds': {},
    }

    try:
        algorithm = TopDown(data_path=source, hierarchy=hierarchy, query_columns=columns,
                            privacy_mechanism=ZCDP(rhos),
                            out_path=common.out_path(dataset, f'{name}.csv'),
                            solver_options=common.SOLVER_OPTIONS, num_workers=args.workers,
                            check_correctness=args.check,
                            domain={column: domains.DOMAINS[column] for column in columns})
        algorithm.data_handler.use_noise_cache = False
        if args.full_joint:
            algorithm.set_query_workload(None)
        else:
            bags = [[c for c in bag if c in columns] for bag in marginals.STRUCTURES[args.structure]]
            algorithm.set_marginals([bag for bag in bags if bag])
        algorithm.set_rounding_method(args.rounding)
        algorithm.set_constraint_to_level(0, SumEqualRealTotal(TrueExpression()))
        for rule in constraints.constraints(columns):
            algorithm.set_constraint_to_tree(rule)

        try:
            started = time.perf_counter()
            algorithm.initialize()
            record['seconds']['initialize'] = time.perf_counter() - started
            if args.full_joint:
                record.update(bags=None, width=int(algorithm.data_handler.n_cells))
            else:
                record.update(bags=[list(bag) for bag in algorithm.junction_tree.bags],
                              width=int(algorithm.data_handler.marginal_width))
            record['sensitivity'] = int(algorithm.query_sensitivity)
            started = time.perf_counter()
            algorithm.estimation_phase()
            record['seconds']['estimation'] = time.perf_counter() - started
        finally:
            algorithm.data_handler.cleanup_directories()
        record['memory_mb'] = peak_memory_mb()
        record['status'] = 'ok'
    except Exception:
        record['error'] = traceback.format_exc()
        raise
    finally:
        with open(common.out_path(dataset, f'{name}.json'), 'w', encoding='utf-8') as handle:
            json.dump(record, handle, indent=1)

    print(json.dumps({key: record.get(key) for key in
                      ('name', 'status', 'width', 'sensitivity', 'seconds', 'memory_mb')}))
