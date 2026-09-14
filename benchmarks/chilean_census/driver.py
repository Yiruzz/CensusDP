"""Run TopDown on the Chilean 2017 census, personas. Flags: benchmarks/runner.py."""

from benchmarks import runner
from . import constraints, domains, marginals

DATASET = 'chilean_census'
HIERARCHY = ['REGION', 'PROVINCIA', 'COMUNA', 'DC', 'ZC_LOC']

if __name__ == '__main__':
    runner.main(DATASET, HIERARCHY, domains, constraints, marginals)
