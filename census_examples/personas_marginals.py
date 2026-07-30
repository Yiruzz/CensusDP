"""Marginals/Factored (junction-tree) TopDown run on the 2017 census PERSONAS file.

This driver processes ALL of the questionnaire's question columns (P07 .. P21A) at once.
With that many columns the full joint contingency vector is astronomically large and cannot
be materialized (the product of every column's cardinality), so this run uses the FACTORED
pipeline: every node stores one small marginal per junction-tree bag instead of the joint,
and consistency between overlapping bags replaces the joint. That is the only representation
that scales to the whole questionnaire.

The marginals are declared here (MARGINALS below) rather than selected from the data. On top
of them, every edit constraint's column scope is added as a mandatory bag automatically (see
TopDown._build_junction_tree), so the skip-logic rules are always enforceable regardless of
what is declared. Declaring rather than selecting keeps the junction tree deterministic and
its bags small: automatic selection over these 27 columns, several of which are
high-cardinality identifier codes (comuna, country, year, occupation), can triangulate into a
single bag of tens of millions of cells.

Run with:

    python -m census_examples.personas_marginals
"""

from topdown import TopDown
from privacy import ZCDP
from constraints.contextual_constraints import SumEqualRealTotal
from constraints.logical_expressions.atomic import TrueExpression
from .census_constraints import personas_constraints


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------

DATA_PATH = ('data/csv-personas-censo-2017/microdato_censo2017-personas/'
             'Microdato_Censo2017-Personas.csv')

# Geographic hierarchy. The root is the national aggregate; the one column below adds a node
# per REGION. Keeping the hierarchy shallow keeps the 17.5M-person reconstruction feasible;
# add PROVINCIA / COMUNA here to go deeper (much heavier).
HIERARCHY = ['REGION']

# Every QUESTION column of the personas questionnaire. The remaining CSV columns are either
# geographic (REGION, PROVINCIA, COMUNA, DC, AREA, ZC_LOC, ID_ZONA_LOC, NVIV, NHOGAR,
# PERSONAN) or derived recodings (*_GRUPO, ESCOLARIDAD, *_15R), not raw questions.
QUERIES = [
    'P07',           # Parentesco con el jefe de hogar
    'P08',           # Sexo
    'P09',           # Edad
    'P10',           # Residencia habitual
    'P10COMUNA', 'P10PAIS',
    'P11',           # Residencia hace 5 anios
    'P11COMUNA', 'P11PAIS',
    'P12',           # Lugar de nacimiento
    'P12COMUNA', 'P12PAIS', 'P12A_LLEGADA', 'P12A_TRAMO',
    'P13',           # Asistencia a la educacion
    'P14', 'P15', 'P15A',
    'P16',           # Pertenencia a pueblo originario
    'P16A', 'P16A_OTRO',
    'P17',           # Trabajo la semana pasada
    'P18',           # Ocupacion (VARCHAR)
    'P19',           # Hijos nacidos vivos
    'P20', 'P21M', 'P21A',
]

OUTPUT_PATH = 'data/out/personas_marginals_microdata.csv'

# Declared marginals: the correlations to preserve, beyond what the constraints already force.
# Each group is measured jointly as one bag; the edit-constraint scopes are unioned in
# automatically. Kept to core cross-tabs a census publishes (demographics x education x work x
# ethnicity x migration) and away from the high-cardinality codes, so the bags stay small.
MARGINALS = [
    ['P07', 'P08', 'P09'],   # parentesco x sexo x edad
    ['P08', 'P09', 'P13'],   # sexo x edad x asistencia a educacion
    ['P13', 'P14', 'P15'],   # nivel educativo (detalle)
    ['P08', 'P09', 'P16'],   # sexo x edad x pueblo originario
    ['P08', 'P09', 'P17'],   # sexo x edad x trabajo la semana pasada
    ['P09', 'P12'],          # edad x lugar de nacimiento
    ['P10', 'P11'],          # residencia actual x hace 5 anios (migracion)
]


# --------------------------------------------------------------------------------------------
# Privacy budget: rho-zCDP, exponential per-level allocation (more budget towards the leaves)
# --------------------------------------------------------------------------------------------

TOTAL_RHO = 10.0
N_LEVELS = len(HIERARCHY) + 1                    # root + one per hierarchy column
_weight = sum(2 ** i for i in range(N_LEVELS))
PRIVACY_PARAMETERS = [(TOTAL_RHO / _weight) * (2 ** i) for i in range(N_LEVELS)]


# --------------------------------------------------------------------------------------------
# Solver. Threads=1 per worker so the workers do not oversubscribe the machine. MIPGap gives
# the rounding MIP a small optimality tolerance so it does not chase the last fraction on a
# wide, high-cardinality bag.
# --------------------------------------------------------------------------------------------

SOLVER_OPTIONS = {'OutputFlag': 0, 'Threads': 1, 'MIPGap': 1e-4}

# 8-core machine: 8 single-threaded workers saturate the cores; the main process is mostly
# idle while they solve.
NUM_WORKERS = 8


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
        optimizer_backend='write_lp',
    )

    # Factored pipeline with DECLARED marginals. The edit-constraint scopes are added as
    # mandatory bags automatically, so the skip-logic rules below are always enforced.
    algorithm.set_marginals(MARGINALS)
    print(f'Factored pipeline with {len(MARGINALS)} declared marginals (constraint scopes '
          f'are added automatically)')

    # Population-total invariant: publish the exact real total at the national root and at
    # every REGION. Contextual - the true count is read per node at tree-build time.
    # The level argument indexes HIERARCHY (0 = REGION); set_constraint_to_level applies it to
    # every tree level up to that index + 1, so index 0 already covers root (0) AND region (1).
    algorithm.set_constraint_to_level(0, SumEqualRealTotal(TrueExpression()))

    # Edit constraints (questionnaire skip logic): structural zeros that keep the released
    # microdata internally coherent. Only the rules whose columns are all queried are kept.
    edit_constraints = personas_constraints(QUERIES)
    for constraint in edit_constraints:
        algorithm.set_constraint_to_tree(constraint)

    print(f'\n{len(edit_constraints)} edit constraints applicable to the {len(QUERIES)} '
          f'question columns:')
    for constraint in edit_constraints:
        print(f'  scope {sorted(constraint.scope())}: {constraint}')

    algorithm.run()


if __name__ == '__main__':
    main()
