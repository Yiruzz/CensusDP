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
    # QUERIES = ['P08', 'P09'] # Sex and Age
    QUERIES = queries

    ##############################
    # Input and output data path # 
    ##############################

    # TODO: Accept other data formats that are not csv.

    # In this case we are using chilean 2017 Census data that can be found at:
    # https://www.ine.gob.cl/estadisticas/sociales/censos-de-poblacion-y-vivienda/censo-de-poblacion-y-vivienda

    # DATA_PATH_PERSONAS = 'data/csv-personas-censo-2017/microdato_censo2017-personas/Microdato_Censo2017-Personas.csv'
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
    )

    # Set the queries to be answered at each node of the tree.
    topdown.set_query_workload(None)

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
        # then the question can't be answeredd. The value 98 is used to indicate that the question
        # does not apply to that household. Therefore, we need to set the following constraint.
        # if 'P02' != 1 -> ('P03A' = 98) & ('P03B' = 98) & ('P03C' = 98) & 
        #                  ('P04' = 98) & ('P05' = 98) & ('CANT_HOG' = 0) & ('CANT_PER' = 0)
        left_side = NotEqual('P02', 1)
        #right_side = And(Equal('P03A', 98), Equal('P03B', 98), Equal('P03C', 98), Equal('P04', 98), Equal('P05', 98), Equal('CANT_HOG', 0), Equal('CANT_PER', 0))
        right_side = And(Equal('P03A', 98), Equal('P03B', 98))
        VIVIENDAS_CONSTRAINT = Implies(left_side, right_side)

        # We will apply this constraint to all levels of the tree.
        topdown.set_constraint_to_tree(VIVIENDAS_CONSTRAINT)

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
        default="pyoptinterface",
        help="Optimizer backend to use: 'pyoptinterface' (default) or 'write_lp'",
    )

    args = parser.parse_args()

    main(args.process_until, args.queries, args.user_constraints, args.num_workers, args.check_correctness, args.optimizer)