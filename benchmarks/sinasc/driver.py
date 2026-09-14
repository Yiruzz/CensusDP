"""Run TopDown on SINASC 2022, down to the municipality of residence. Flags: benchmarks/runner.py."""

from benchmarks import runner
from . import constraints, domains, marginals

DATASET = 'sinasc'
# Region, state, health region, municipality. The health macro-region cannot be stacked on the
# health region: the two DATASUS tables disagree on 90 municipalities.
HIERARCHY = ['REGIAO', 'UF', 'REGSAUD', 'CODMUNRES']

if __name__ == '__main__':
    runner.main(DATASET, HIERARCHY, domains, constraints, marginals)
