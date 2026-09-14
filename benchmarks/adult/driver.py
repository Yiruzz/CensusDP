"""Run TopDown on Adult, over the fabricated age hierarchy. Flags: benchmarks/runner.py."""

from benchmarks import runner
from . import constraints, domains, marginals
from .hierarchy import HIERARCHY

DATASET = 'adult'

if __name__ == '__main__':
    runner.main(DATASET, HIERARCHY, domains, constraints, marginals)
