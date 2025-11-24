from topdown import TopDown

# Import constraint building classes
from constraints.contextual_constraints import SumEqualRealTotal
from constraints.logical_expressions.atomic import Equal, NotEqual, TrueExpression
from constraints.logical_expressions.compound import And, Implies  

def main():
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
    PROCESS_UNTIL = 'COMUNA'
    GEO_COLUMNS_TO_USE = GEO_COLUMNS[:GEO_COLUMNS.index(PROCESS_UNTIL) + 1]

    # Define the columns to use that will be queried in each node of the tree.
    # QUERIES = ['P08', 'P09'] # Sex and Age
    QUERIES = ['P02', 'P03A', 'P03B'] # Viviendas queries

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

    # With the TopDown class instantiated, we can set all the parameters
    topdown = TopDown(data_path=DATA_PATH_VIVIENDAS, hierarchy=GEO_COLUMNS_TO_USE, queries=QUERIES, out_path=OUTPUT_PATH+OUTPUT_FILE)

    #######################################################
    # Differential privacy budget and mechanism settings #
    #######################################################

    # The privact budget needs to be defined for each level of the tree, using a list.
    # In this case, we will use a tree with 6 levels, so we split the total budget in 6 parts.
    # Privacy parameter to use for the whole algorithm.
    total_privacy_budget = 10
    n_levels = 6

    # We will consider and exponential allocation of the privacy budget across the levels of the tree.
    aux = 0
    for i in range(n_levels):
        aux += (2**i)
    # Privacy parameters for the noise generation. First value for root, last for leaves.
    PRIVACY_PARAMETERS = [(total_privacy_budget/aux)*(2**i) for i in range(n_levels)]
    topdown.set_privacy_parameters(PRIVACY_PARAMETERS)

    # Noise mechanism to use (discrete_laplace or discrete_gaussian).
    MECHANISM = 'discrete_laplace'
    topdown.set_mechanism(MECHANISM)

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

    # TODO: Do a documentation of how to build constraints with the package.

    # In this case, we will add two constraints as examples.
    # We want that for the level of 'COMUNA' that the true total of viviendas is published.
    # Since we don't know the specific number of households per COMUNA in advance, 
    # we use a contextual constraint that will get the real total from the data at runtime (dynamically).
    real_total_constraint = SumEqualRealTotal(expression=TrueExpression())
    # 3 is the level of 'COMUNA' in the tree
    topdown.set_constraint_to_level(3, real_total_constraint)

    # TODO: Study macros for a better user interface

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

    # Path of data that already has been processed until higher level of the tree.
    # This is used to avoid to process the data again if it has already been processed by the algorithm.
    # If None the algorithm will start from the root node.
    DATA_PATH_PROCESSED = None
    if DATA_PATH_PROCESSED: topdown.read_processed_data(DATA_PATH_PROCESSED, sep=';')

    # Distance metric to use (manhattan, euclidean, cosine) if None no distance will be computed.
    # The distance metric is used to compare the original contingency vector with the noisy one.
    # Only used for testing and analysis purposes.
    DISTANCE_METRIC = None
    if DISTANCE_METRIC: topdown.set_distance_metric(DISTANCE_METRIC)



    # Finally, we can run the TopDown algorithm
    topdown.run()
    
    # This method can be used to check the correctness of the results.
    # Also used for testing purposes.
    topdown.check_correctness()

if __name__ == "__main__":
    main()