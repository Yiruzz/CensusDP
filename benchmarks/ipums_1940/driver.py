"""Run TopDown on IPUMS 1940.

The DAS comparison is --full-joint --workload das --epsilon 4 --composition uniform, which
reproduces their seven query groups, their uneven split between them and their pure-DP
mechanism. Flags: benchmarks/runner.py.
"""

from benchmarks import runner
from . import constraints, domains, marginals, workload

DATASET = 'ipums_1940'
HIERARCHY = ['STATEFIP', 'COUNTY', 'SUPDIST', 'ENUMDIST']

if __name__ == '__main__':
    runner.main(DATASET, HIERARCHY, domains, constraints, marginals, workload)
