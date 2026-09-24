import argparse

from topdown import TopDown

# Import privacy mechanism classes
from privacy import PureDP, ZCDP, ApproximateDP, RenyiDP

# Import constraint building classes
from constraints.contextual_constraints import SumEqualRealTotal
from constraints.logical_expressions.atomic import Equal, NotEqual, TrueExpression
from constraints.logical_expressions.compound import And, Implies

# Import query building classes
from queries import QueryWorkload, col

# Declared per-column domains, read off the 2017 questionnaire (see census_examples).
from census_examples.census_domains import viviendas_domain



def main(process_until: str, queries: list[str], user_constraints: bool,
         num_workers: int, check_correctness: bool, optimizer_backend: str):
    '''Main function to set variables and run the TopDown algorithm.'''

    ###################################
    # Hierachy and queries definition #
    ###################################

    # NOTE: The hierarchical columns must be in the dataframe used as input data.
    #       The algorithm will assume that the data has a functional hierarchy that
    #       where each level is a proper aggregation of the lower levels.
    #       A root node will be created automatically, representing the aggregation of all the data
    #       in the dataframe (therefore the second level of the tree will be the first hierarchical column).
    
    # In this case we use a geographic hierarchy.
    # In this case the tree will have 6 levels: The root representing the national level, and then the 
    # following levels in descending hierarchical order:
    GEO_COLUMNS = ['REGION', 'PROVINCIA', 'COMUNA', 'DC', 'ZC_LOC']

    # We will process the data until a specific level of the tree for this test case.
    PROCESS_UNTIL = process_until
    PROCESS_UNTIL_idx = GEO_COLUMNS.index(PROCESS_UNTIL)
    GEO_COLUMNS_TO_USE = GEO_COLUMNS[:PROCESS_UNTIL_idx + 1]

    # Define the columns to use that will be queried in each node of the tree.
    QUERIES = queries

    ##############################
    # Input and output data path # 
    ##############################

    # TODO: Accept other data formats that are not csv.

    # In this case we are using chilean 2017 Census data that can be found at:
    # https://www.ine.gob.cl/estadisticas/sociales/censos-de-poblacion-y-vivienda/censo-de-poblacion-y-vivienda

    DATA_PATH_VIVIENDAS= 'data/csv-viviendas-censo-2017/microdato_censo2017-viviendas/Microdato_Censo2017-Viviendas.csv'

    OUTPUT_PATH = 'data/out/'
    OUTPUT_FILE = 'viviendas_noisy_microdata_' + PROCESS_UNTIL + '_' + '_'.join(QUERIES) + '.csv'

    #######################
    # Solver configuration #
    #######################
    
    # Solver: Gurobi
    SOLVER_OPTIONS = {'OutputFlag': 0, 'Threads': 1}  # Suppress Gurobi output

    ######################################################
    # Differential privacy budget and mechanism settings #
    ######################################################

    # The privacy budget needs to be defined for each level of the tree, using a list.
    # The tree has len(hierarchy) + 1 levels (root + one per hierarchical column).
    total_privacy_budget = 10
    n_levels = len(GEO_COLUMNS_TO_USE) + 1

    # We will consider and exponential allocation of the privacy budget across the levels of the tree.
    aux = sum(2**i for i in range(n_levels))
    # Privacy parameters for the noise generation. First value for root, last for leaves.
    PRIVACY_PARAMETERS = [(total_privacy_budget/aux)*(2**i) for i in range(n_levels)]

    # Privacy variant. Choose ONE:
    #   - PureDP(epsilons)              ε per tree level (Laplace mechanism)
    #   - ZCDP(rhos)                    ρ per tree level (discrete Gaussian, ρ-zCDP linear composition)
    #   - ApproximateDP(rhos, delta=δ)  ρ per tree level + global δ (zCDP under the hood, reports (ε, δ)-DP)
    #   - RenyiDP(epsilons, delta=δ)    ε per tree level + global δ, optimised discrete Gaussian calibration,
    #                                   it considers the total budget to calibrate the noise of each level
    PRIVACY_MECHANISM = ZCDP(PRIVACY_PARAMETERS)

    # With the TopDown class instantiated, we can set all the parameters
    topdown = TopDown(
        data_path=DATA_PATH_VIVIENDAS,
        hierarchy=GEO_COLUMNS_TO_USE,
        query_columns=QUERIES,
        privacy_mechanism=PRIVACY_MECHANISM,
        out_path=OUTPUT_PATH+OUTPUT_FILE,
        solver_options=SOLVER_OPTIONS,
        num_workers=num_workers,
        check_correctness=check_correctness,
        optimizer_backend=optimizer_backend,
        # Declared rather than inferred: inference is data-dependent (not DP-safe). Filtered
        # to whatever --queries asked for, so any column subset works.
        domain=viviendas_domain(QUERIES),
    )

    ###########################################
    # Cell space: full-joint vs. factored     #
    ###########################################

    # This example runs the FULL-JOINT pipeline: each node stores ONE contingency vector over
    # the joint of QUERIES, whose length is the product of the column cardinalities. It is
    # exact but only feasible for a handful of columns. Leaving the query workload as the
    # identity (None) is what selects it.
    topdown.set_query_workload(None)

    # To run the FACTORED (junction-tree) pipeline instead - the one that scales to many
    # columns - call ONE of the following BEFORE topdown.run() and do NOT call
    # set_query_workload. Each node then stores one small marginal per junction-tree bag, and
    # consistency between overlapping bags replaces the joint. Every constraint scope (below)
    # is added as a mandatory bag automatically, so the edit constraints stay enforceable.
    # A full factored example lives in census_examples/personas_marginals.py.
    #
    #   # (a) Declare the marginals to keep jointly (deterministic):
    #   topdown.set_marginals([['P01', 'P02'], ['P03A', 'P03B']])
    #
    #   # (b) Or let the algorithm pick them from the data privately. The selection share is
    #   #     taken OUT OF the total budget (not added on top), so the guarantee is unchanged:
    #   topdown.set_marginal_selection(budget_fraction=0.2)

    ####################
    # Edit Constraints #
    ####################

    # NOTE: Consistency constraints are automatically added by the algorithm to ensure that
    #       the tree structure is maintained. Therefore, only additional constraints need to be added here.
    #       Edit constraints usually depend on the specific dataset being used and restrictions on the publication
    #       of the data, such as legal or policy requirements.
    #
    #       In the constraints package there are a series of classes that should be used to build the constraints.
    #       See the documentation for more details.

    # CONSTRAINT BUILDING GUIDE:
    #
    # The constraints framework provides a DSL for expressing complex logical conditions:
    #
    # 1. ATOMIC EXPRESSIONS (conditions on single columns):
    #    - Equal(column, value)        : column == value
    #    - NotEqual(column, value)     : column != value
    #    - GreaterThan(column, value)  : column > value
    #    - LessThan(column, value)     : column < value
    #    - TrueExpression()            : always true (useful for aggregations)
    #
    # 2. LOGICAL COMBINATORS:
    #    - And(expr1, expr2, ...)      : all expressions must be true
    #    - Or(expr1, expr2, ...)       : at least one expression must be true
    #    - Not(expr)                   : negation of the expression
    #    - Implies(left, right)        : if left is true, then right must be true
    #
    # 3. CONSTRAINT TYPES:
    #    - SumEqual(expression, value) : sum of matching records equals value
    #    - SumEqualRealTotal(expression) : sum equals the true total from data
    #
    # 4. APPLICATION METHODS:
    #    - topdown.set_constraint_to_level(level, constraint) : apply to specific tree level
    #    - topdown.set_constraint_to_tree(constraint)         : apply to all tree levels
    #
    # EXAMPLES:

    # In this case, we will add two constraints as examples.
    # We want that for the level of 'COMUNA' that the true total of viviendas is published.
    # Since we don't know the specific number of households per COMUNA in advance, 
    # we use a contextual constraint that will get the real total from the data at runtime (dynamically).
    real_total_constraint = SumEqualRealTotal(expression=TrueExpression())
    topdown.set_constraint_to_level(PROCESS_UNTIL_idx, real_total_constraint)

    # TODO: Maybe a refactor to the constraint building process to make it more user-friendly.
    #       See queries workload definition for inspiration and use overloading of boolean and comparison operators
    #       in the expressions to make it more intuitive to build the constraints.

    if user_constraints:
        # The Census data specifies that if a household was empty when the census was taken,
        # then the occupant questions can't be answered. A "no aplica" sentinel is used
        # instead (98 for most fields, 0 for the household/person counts). So:
        #   'P02' != 1 -> ('P03A'=98) & ('P03B'=98) & ('P03C'=98) &
        #                 ('P04'=98) & ('P05'=98) & ('CANT_HOG'=0) & ('CANT_PER'=0)
        #
        # Build the LARGEST form the current QUERIES allow: include only the consequent
        # columns that are actually being queried, so the same code works for any --queries
        # (referencing a column that is not queried would have no cell to constrain). The
        # antecedent P02 must be queried too, otherwise the rule cannot be expressed.
        SENTINELS = {'P03A': 98, 'P03B': 98, 'P03C': 98, 'P04': 98,
                     'P05': 98, 'CANT_HOG': 0, 'CANT_PER': 0}
        consequents = [Equal(column, value) for column, value in SENTINELS.items()
                       if column in QUERIES]

        if 'P02' in QUERIES and consequents:
            # Apply to all levels of the tree.
            topdown.set_constraint_to_tree(Implies(NotEqual('P02', 1), And(*consequents)))
            print(f"\nViviendas edit constraint on: {[c.variable_id for c in consequents]}")
        else:
            print("\nViviendas edit constraint skipped: needs 'P02' and at least one of "
                  f"{sorted(SENTINELS)} in --queries.")

    #######################
    # Additional settings #
    #######################

    # TODO: Implement a logging system to keep track of the algorithm's execution and partial results.
    #       Then implement the option to load the logs and partial results to continue the execution of the algorithm from a specific point.
    #       This is useful for long executions and to avoid losing progress in case of crashes or interruptions.

    # Distance metric to use (manhattan, euclidean, cosine) if None no distance will be computed.
    # The distance metric is used to compare the original contingency vector with the noisy one.
    # Only used for testing and analysis purposes.
    DISTANCE_METRIC = None
    if DISTANCE_METRIC: topdown.set_distance_metric(DISTANCE_METRIC)

    # Finally, we can run the TopDown algorithm
    topdown.run()

    # Privacy garantee
    #print(topdown.privacy_mechanism.report_guarantee())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--process_until",
        help="Column of the geographic hierarchy where the tree processing stops",
        required=True
    )

    parser.add_argument(
        "--queries",
        nargs="+",
        help="Dataset columns used to create the contingency vector",
        required=True
    )

    parser.add_argument(
        "--user_constraints",
        action="store_true",
        help="Whether to consider user constraints during execution",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for parallel execution (default: 2)",
    )

    parser.add_argument(
        "--check_correctness",
        action="store_true",
        help="Whether to check correctness during execution",
    )

    parser.add_argument(
        "--optimizer",
        choices=["write_lp", "pyoptinterface"],
        default="write_lp",
        help="Optimizer backend to use: 'write_lp' (default) or 'pyoptinterface' "
             "(requires the optional pyoptinterface package)",
    )

    args = parser.parse_args()

    main(args.process_until, args.queries, args.user_constraints, args.num_workers, args.check_correctness, args.optimizer)