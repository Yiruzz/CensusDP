"""Run TopDown on the Spanish 2021 census: provinces and municipalities. Flags: benchmarks/runner.py."""

from benchmarks import runner
from . import constraints, domains, marginals

DATASET = 'spanish_census'
# CMUN pools municipalities under 10,000 inhabitants into size bands (INE disclosure control).
HIERARCHY = ['CPRO', 'CMUN']

if __name__ == '__main__':
    runner.main(DATASET, HIERARCHY, domains, constraints, marginals)
