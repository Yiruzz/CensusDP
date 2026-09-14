"""Run TopDown on IPUMS 1940. Use --full-joint for the DAS comparison. Flags: benchmarks/runner.py."""

from benchmarks import runner
from . import constraints, domains, marginals

DATASET = 'ipums_1940'
HIERARCHY = ['STATEFIP', 'COUNTY', 'SUPDIST', 'ENUMDIST']

if __name__ == '__main__':
    runner.main(DATASET, HIERARCHY, domains, constraints, marginals)
