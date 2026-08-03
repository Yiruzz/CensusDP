"""Full-joint TopDown run on the 2017 census VIVIENDAS file.

The full-joint pipeline stores one contingency vector per node, of length equal to the
product of all query-column cardinalities. It is exact but only feasible for a handful of
columns - here six columns give 155,520 cells. For many columns use the factored pipeline
instead (see census_examples/personas_marginals.py).

Run with:

    python -m census_examples.viviendas_joint
"""

from topdown import TopDown
from privacy import ZCDP
from constraints.contextual_constraints import SumEqualRealTotal
from constraints.logical_expressions.atomic import TrueExpression
from .census_constraints import viviendas_constraints
from .census_domains import viviendas_domain


DATA_PATH = ('data/csv-viviendas-censo-2017/microdato_censo2017-viviendas/'
             'Microdato_Censo2017-Viviendas.csv')
OUTPUT_PATH = 'data/out/viviendas_joint_microdata.csv'

# Geographic hierarchy: national root + one node per REGION.
HIERARCHY = ['REGION']


# Seven columns.
QUERIES = ['P01', 'P02', 'P03A', 'P03B', 'P03C', 'P04', 'P05']

# rho-zCDP budget, one value per tree level, exponentially increasing towards the leaves.
TOTAL_RHO = 5
N_LEVELS = len(HIERARCHY) + 1
_weight = sum(2 ** i for i in range(N_LEVELS))
PRIVACY_PARAMETERS = [(TOTAL_RHO / _weight) * (2 ** i) for i in range(N_LEVELS)]

SOLVER_OPTIONS = {'OutputFlag': 0, 'Threads': 1}
NUM_WORKERS = 4


def main():
    algorithm = TopDown(
        data_path=DATA_PATH,
        hierarchy=HIERARCHY,
        query_columns=QUERIES,
        privacy_mechanism=ZCDP(PRIVACY_PARAMETERS),
        out_path=OUTPUT_PATH,
        solver_options=SOLVER_OPTIONS,
        num_workers=NUM_WORKERS,
        check_correctness=True,
        # Declared from the questionnaire, not inferred from the data: inferring makes the
        # shape of the cell space data-dependent (not DP-safe) and drops valid-but-absent
        # values - including the "no aplica" sentinels the edit constraints below assert on.
        domain=viviendas_domain(QUERIES),
    )

    # Full-joint pipeline: no set_marginals() / set_marginal_selection() call.

    # Population/dwelling-total invariant, exact at the national root and per REGION.
    algorithm.set_constraint_to_level(0, SumEqualRealTotal(TrueExpression()))

    # Edit constraint: an unoccupied dwelling (P02 != 1) has no occupant characteristics, so
    # those fields carry the "no aplica" sentinel. Structural zeros that keep the released
    # microdata internally coherent. Only rules whose columns are all queried are kept.
    for constraint in viviendas_constraints(QUERIES):
        algorithm.set_constraint_to_tree(constraint)

    algorithm.run()


if __name__ == '__main__':
    main()
