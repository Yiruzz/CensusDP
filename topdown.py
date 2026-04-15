import pandas as pd
import numpy as np
from hierarchical_tree import HierarchicalTree
from hierarchical_tree import HierarchicalNode
from data_handler import DataHandler
from optimizer import OptimizationModel
from constraints.constraint import Constraint

from noisy import sample_dgauss_fast, sample_dgauss_optimized,  sample_dlaplace_fast, sample_dlaplace_optimized

from typing import Callable, Dict, List
import time

class TopDown():
    '''Represents the TopDown algorithm for generating differentially private microdata.
    
    The algorithm works by constructing a hierarchical tree structure, then adding noise to the data
    considering differential privacy principles. It propagates the noise to each node in the tree and
    finally it solves optimization problems to ensure consistency across the tree and adherence to 
    specified constraints by the user.
    '''
    def __init__(self, data_path: str, hierarchy: List[str], last_hierarchical_column: str, queries: List[str],
                 tree_path: str, microdata_path: str, optimizer: str, solver_options: dict) -> None:
        '''
        Initialize the TopDown algorithm.

        Args:
            data_path (str): Path to the input data file.
            hierarchy (List[str]): List of columns representing the hierarchy levels.
            last_hierarchical_column (str): The last column in the hierarchy that will be considered.
            queries (List[str]): List of columns to be queried and aggregated.
            tree_path (str): Path to save the three and maybe the contingency vectors.
            microdata_path (str): Path to save the processed microdata.
            optimizer (str): The optimization solver to use ('gurobi', 'ipopt', 'glpk', etc.).
            solver_options (dict): Dictionary of options to pass to the solver.

        Attributes:
            data_handler (DataHandler): Instance of DataHandler for managing data operations.

            hierarchical_columns (List[str]): List of columns representing the hierarchy levels.
            index_last_hierarchical_column (int): Index that indicate which is the positions of the last hierarchical column to consider.
            query_columns (List[str]): List of columns to be queried and aggregated.

            privacy_parameters (List[float]): List of privacy parameters for each level of the tree.
            mechanism (str): The noise mechanism to use ('discrete_laplace' or 'discrete_gaussian').

            tree (HierarchicalTree): Instance of HierarchicalTree representing the hierarchical structure.
            optimizer (OptimizationModel): Instance of OptimizationModel for solving optimization problems.
            constraints (Dict[int, List[Constraint]]): Dictionary mapping tree levels to their constraints

        '''
        self.data_handler: DataHandler = DataHandler(input_path=data_path,
                                                     hierarchical_columns=hierarchy,
                                                     query_columns=queries,
                                                     output_tree=tree_path,
                                                     output_path=microdata_path)
        
        self.hierarchical_columns: List[str] = hierarchy
        self.index_last_hierarchical_column: int = self.hierarchical_columns.index(last_hierarchical_column)
        self.query_columns: List[str] = queries

        self.privacy_parameters: List[float] = []
        self.mechanism: Callable = lambda x: x  # Default to identity function

        self.tree: HierarchicalTree = HierarchicalTree(nodes=[], node_ranges_by_level=[])
        self.optimizer: OptimizationModel = OptimizationModel(optimizer, solver_options)
        self.constraints: Dict[int, List[Constraint]] = {}
    
    def initialize(self) -> None:
        '''Initialize the TopDown algorithm.
        This involves loading an existing tree or building it from the data, then generating
        the contingency vectors for each node in the tree and setting their constraints.
        '''

        print(f'\nInitializing TopDown algorithm...')

        # Read the data 
        t1 = time.time()
        print(f' Reading data...', end=' ')
        self.data_handler.read_data(sep=';')
        print(f'{time.time() - t1:.2f} seconds.')

        # Check if use a created tree with its sorted data (assumed)
        if self.data_handler.tree_file is None:
            # Sort read data
            t1 = time.time()
            print(f' Sorting data by hierarchical columns...', end=' ')
            self.data_handler.sort_data_by_hierarchy()
            print(f'{time.time() - t1:.2f} seconds.')

            # Write sorted data to use in a next execution (but update the current dataframe)
            t1 = time.time()
            print(f' Writing data by hierarchical columns...', end=' ')
            folder = "/".join(self.data_handler.data_path.split('/')[:-1])
            filename = self.data_handler.data_path.split('/')[-1]
            out_path = f"{folder}/{filename.split('.')[0]}_sorted.csv"
            self.data_handler.write_data(self.data_handler.dataframe, out_path)
            print(f'{time.time() - t1:.2f} seconds.')

            #NOTE: Create the tree is very fast, so not is necessary save it, but it is useful for the user to see and debugg then.
            # Create tree
            t1 = time.time()
            print(f' Creating hierarchical tree structure...', end=' ')
            df = self.data_handler.create_tree()
            print(f'{time.time() - t1:.2f} seconds.')

            # Save tree
            t1 = time.time()
            print(f' Saving hierarchical tree structure...', end=' ')
            tree_file = "tree.csv"
            self.data_handler.write_data(df, f"{self.data_handler.tree_folder}/{tree_file}")
            self.data_handler.tree_file = tree_file
            print(f'{time.time() - t1:.2f} seconds.')

        # Reduce the data
        self.data_handler.hierarchical_columns = self.hierarchical_columns[:self.index_last_hierarchical_column+1]
        self.data_handler.reduce_data()
        
        # Load tree
        t1 = time.time()
        print(f' Loading hierarchical tree structure...', end=' ')
        self.tree = self.data_handler.load_tree()
        print(f'{time.time() - t1:.2f} seconds.')
        
        # Create contingency dataframe to create constrains after
        self.data_handler.generate_contingency_dataframe()

        # Check if there are oldest contigency vectors
        # NOTE: The user have the responsability to check if the file passed has the contigency vectors for all nodes and
        # them correspondent to the queries.
        raw_file, raw_flag = self.data_handler.raw_contingency_vectors_file
        noisy_file, noisy_flag = self.data_handler.noisy_contingency_vectors_file

        file_to_load = None
        if raw_file is not None and not raw_flag:
            file_to_load = raw_file
        elif noisy_file is not None and not noisy_flag:
            file_to_load = noisy_file

        if file_to_load is not None:
            t1 = time.time()
            print(" Loading contingency vectors for each node in the tree...", end=" ")
            self.data_handler.load_contingency_vectors(self.tree, file_to_load)
            print(f"{time.time() - t1:.2f} seconds.")
        
        else:
            # Generate contigency vectors for each node in the tree
            t1 = time.time()
            print(f' Creating contingency vectors for each node in the tree...', end=' ')
            self.data_handler.create_contingency_vectors(self.tree)
            print(f'{time.time() - t1:.2f} seconds.')


        # Set constraints for each node in the tree
        t1 = time.time()
        print(f' Setting constraints for each node in the tree...', end=' ')
        self.data_handler.add_constraints(self.tree, self.constraints)
        print(f'{time.time() - t1:.2f} seconds.')

        return None

    def measurement_phase(self) -> None:
        '''Perform the measurement phase of the TopDown algorithm.
        
        This method adds noise to the data at each node in the hierarchical tree according to the specified
        privacy parameters and mechanism.
        '''
        t1 = time.time()
        print(f'\nRunning measurement phase...', end=' ')
        for node in self.tree.nodes:
            self.add_noise(node, self.privacy_parameters[node.level])
        print(f'{time.time() - t1:.2f} seconds.')
        return None

    def estimation_phase(self) -> None:
        '''Perform the estimation phase of the TopDown algorithm.
        
        This method solves optimization problems at each node in the hierarchical tree to ensure
        consistency and adherence to constraints after noise has been added.
        '''
        t1 = time.time()
        print(f'\nRunning estimation phase...', end=' ')

        # Root estimation (level 0)
        # Does not require consistency adjustments
        root = self.tree.nodes[0]
        x_tilde: np.ndarray = self.optimizer.non_negative_real_estimation(
            contingency_vector=root.contingency_vector,
            id_node=root.id,
            constraints=root.constraints
        )
        root.contingency_vector = self.optimizer.rounding_estimation(
            x_tilde=x_tilde,
            id_node=root.id,
            constraints=root.constraints
        )

        # Now process the rest of the tree level by level
        for node in self.tree.nodes:
            # If the node is a leaf, no need to solve optimization
            # NOTE: With a break we assume that all leaves are at the same level
            if node.is_leaf():
                break
                
            # Solve the optimization problem for the children of the current node
            childs_contingency_vectors = []
            for i in range(node.children_range[0], node.children_range[1]+1):
                child = self.tree.nodes[i]
                childs_contingency_vectors.append(child.contingency_vector)

            joint_contingency_vector = np.concatenate(childs_contingency_vectors)

            # Transform individual constraints for joint vector
            joint_constraints: List[Callable] = []
            # All vectors have the same length
            vectors_length = len(childs_contingency_vectors[0])
            start = 0
            for i in range(node.children_range[0], node.children_range[1]+1):
                child = self.tree.nodes[i]
                end = start + vectors_length
                for constraint in child.constraints:
                    # NOTE: We use default arguments to avoid late binding issues in lambdas
                    # This can lead to all constraints using the last values saved of start and end
                    # Build a sub-dict with keys 0..(e-s-1) so the constraint's indices still match
                    joint_constraints.append(lambda joint_array, s=start, e=end, c=constraint: c({i - s: joint_array[i] for i in range(s, e)}))
                start = end

            # Consistency constraint: sum of children = parent
            for index in range(vectors_length):
                # Parent's contingency vector value at 'index' must equal sum of children's values at 'index'
                # Precompute the indices to sum to avoid slice notation incompatible with Pyomo vars
                indices_to_sum = list(range(index, len(joint_contingency_vector), vectors_length))
                joint_constraints.append(lambda joint_array, idxs=indices_to_sum, value=node.contingency_vector[index]:
                                            sum(joint_array[j] for j in idxs) == value)
                    
            # Solve for children nodes (joint contingency vector)
            x_tilde = self.optimizer.non_negative_real_estimation(
                contingency_vector=joint_contingency_vector,
                id_node=node.id,
                constraints=joint_constraints
            )
            joint_solution: np.ndarray = self.optimizer.rounding_estimation(
                x_tilde=x_tilde,
                id_node=node.id,
                constraints=joint_constraints
            )

            # Save the solution back to each child node
            start = 0
            for i in range(node.children_range[0], node.children_range[1]+1):
                child = self.tree.nodes[i]
                end = start + vectors_length
                child.contingency_vector = joint_solution[start:end]
                start = end

        print(f'{time.time() - t1:.2f} seconds.')
        return None
    
    def add_noise(self, node: HierarchicalNode, privacy_budget: float) -> None:
        '''Add noise to the node's contingency vector using the specified mechanism.
        Generate the same number of values as the entries in the contingency vector, 
        then sum the values and update the vector.
        
        Args:
            node (HierarchicalNode): The node with a contingency vector.
            privacy_budget (float): The privacy budget (epsilon) for noise addition.
        '''

        n = len(node.contingency_vector)
        noise = self.mechanism(privacy_budget, n)
        node.contingency_vector += noise
        return None

    def construct_microdata(self) -> pd.DataFrame:
        '''Construct the differentially private microdata from the hierarchical tree.
        
        This method generates the final microdata by traversing the hierarchical tree and
        aggregating the data from each node.

        Returns:
            pd.DataFrame: The constructed differentially private microdata.
        '''
        print(f'\nConstructing microdata from hierarchical tree...', end=' ')
        t1 = time.time()
        noisy_df = self.data_handler.construct_microdata(self.tree)
        print(f'{time.time() - t1:.2f} seconds.')

        print(f'Writing noisy data to {self.data_handler.microdata_path}...', end=' ')
        t1 = time.time()
        self.data_handler.write_data(noisy_df)
        print(f'{time.time() - t1:.2f} seconds.\n')
        return noisy_df

    def set_constraint_to_tree(self, constraint: Constraint) -> None:
        '''Add a constraint to all nodes in the hierarchical tree.
        
        The constraint will be applied when the tree is built.

        Args:
            constraint (Constraint): The Constraint to add.
        '''
        for level in range(len(self.hierarchical_columns)):
            if level not in self.constraints:
                self.constraints[level] = []
            self.constraints[level].append(constraint)

    def set_constraint_to_level(self, level: int, constraint: Constraint) -> None:
        '''Add a constraint to a specific level in the hierarchical tree.
        
        The constraint will be applied when the tree is built.

        Args:
            level (int): The level in the tree to which the constraint should be added.
            constraint (Constraint): The Constraint to add.
        '''
        for level_iter in range(level+1):
            if level_iter not in self.constraints:
                self.constraints[level_iter] = []
            self.constraints[level_iter].append(constraint)

    # TODO: Implement method to set constraint to specific node
    # def set_constraint_to_node(self, node_id: int, constraint: Constraint) -> None:
    #     '''Set a constraint to a specific node in the hierarchical tree.
        
    #     It is the user's responsibility to ensure that the optimization problem remains feasible
    #     after adding the constraint. If used improperly, it may lead to infeasible optimization problems.

    #     For example, adding a constraint to all children of a node may contradict the consistency constraint
    #     that the sum of the children's contingency vectors equals the parent's contingency vector.

    #     Args:
    #         node_id (int): The ID of the node to which the constraint should be added.
    #         constraint (Constraint): The Constraint to add.
    #     '''
    #     node = self.tree.nodes[node_id]
    #     if node is not None:
    #         node.constraints.append(constraint)

    def set_privacy_parameters(self, privacy_parameters: List[float]) -> None:
        '''Set the privacy parameters for each level of the hierarchical tree.
        
        Args:
            privacy_parameters (List[float]): List of privacy parameters (epsilon) for each level.
        '''
        self.privacy_parameters = privacy_parameters

    def discrete_gaussian(self, mode: str) -> Callable:
        '''Create a discrete Gaussian sampling mechanism using the selected implementation mode.

        Args:
            mode (str): Implementation mode ('fast' or 'optimized' (default))

        Returns:
            Callable: A function that takes privacy parameter used to derive the scale and number of samples to generate
                      and returns samples from the discrete Gaussian mechanism.
        '''
        match mode:
            case 'fast':
                return lambda epsilon, n_samples: sample_dgauss_fast(1/epsilon, n_samples)
            case _:
                return lambda epsilon, n_samples: sample_dgauss_optimized(1/epsilon, n_samples)
    
    def discrete_laplace(self, mode: str) -> Callable:
        '''Create a discrete Laplace sampling mechanism using the selected implementation mode.

        Args:
            mode (str): Implementation mode ('fast' or 'optimized' (default))

        Returns:
            Callable: A function that takes privacy parameter used to derive the scale and number of samples to generate
                      and returns samples from the discrete Laplace mechanism.
        '''
        match mode:
            case 'fast':
                return lambda epsilon, n_samples: sample_dlaplace_fast(1/epsilon, n_samples)
            case _:
                return lambda epsilon, n_samples: sample_dlaplace_optimized(1/epsilon, n_samples)
        
    def set_mechanism(self, mechanism: str, mode: str) -> None:
        '''Set the noise mechanism and mode to use for adding noise to the data.
        
        Args:
            mechanism (str): The noise mechanism to use ('discrete_laplace' or 'discrete_gaussian').
            mode (str): The mode to use ('Optimized' or 'Fast').
        '''
        match mechanism:
            case 'discrete_laplace':
                self.mechanism = self.discrete_laplace(mode)

            case 'discrete_gaussian':
                self.mechanism = self.discrete_gaussian(mode)
            case _:
                 raise ValueError("Mechanism must be either 'discrete_laplace' or 'discrete_gaussian'.")

    def load_tree(self, filename: str) -> None:
        '''Save the file to the tree to load in the initialization phase.
        Thus if the file doesn't save in data handler attribute, the tree must be created.
        
        Args:
            filename (str): Path to the tree file.
        '''
        self.data_handler.tree_file = filename
        return None

    def load_contigency_vectors(self, filename: str, noisy: bool = False) -> None:
        '''Save the file to the contingency vectors to load in the initialization phase.
        Thus if the file doesn't save in data handler attribute, the contingency vectors must be created (but not necessarily save).

        Args:
            filename (str) = Path to the contingency vectors file.
            noisy (bool) = Specify if the vectors in the file have noisy or not. 
        '''
        if noisy: 
            self.data_handler.noisy_contingency_vectors_file = (filename, False)
        else:
            self.data_handler.raw_contingency_vectors_file = (filename, False)
        return None
    
    def save_contingency_vectors(self, filename: str) -> pd.DataFrame:
        '''Retrieve the contingency vectors from the tree structure,
        writes them to a file in the configured tree folder, and returns them
        as a DataFrame.

        Args:
            filename (str): Name of the output file where the contingency vectors will be stored.

        Returns:
            pd.DataFrame: DataFrame containing the computed contingency vectors.
        '''
        print(f'\nSaving progress...')
        print(f' Getting contingency vectors from hierarchical tree...', end=' ')
        t1 = time.time()
        vectors_df = self.data_handler.get_contingency_vectors(self.tree)
        print(f'{time.time() - t1:.2f} seconds.')

        print(f' Writing contingency vectors data to {self.data_handler.tree_folder}/{filename}...', end=' ')
        t1 = time.time()
        self.data_handler.write_data(vectors_df, f"{self.data_handler.tree_folder}/{filename}")
        print(f'{time.time() - t1:.2f} seconds.')
        return vectors_df
    
    def save_raw_contingency_vectors(self, filename: str) -> None: 
        '''Store the filename and set a flag indicating that the raw contingency vectors need to be saved.
        '''
        self.data_handler.raw_contingency_vectors_file = (filename, True)
        return None

    def save_noisy_contingency_vectors(self, filename: str) -> None:
        '''Store the filename for the noisy contingency vectors and set a flag indicating that they need to be saved.
        '''
        noisy_file = f"{filename.split('.')[0]}_noisy.csv"
        self.data_handler.noisy_contingency_vectors_file = (noisy_file, True)
        return None 
    
    def run(self) -> pd.DataFrame:
        '''Run the TopDown algorithm end-to-end.
        
        This method executes the full TopDown algorithm, including initialization,
        measurement phase, estimation phase, and microdata construction.

        Returns:
            pd.DataFrame: The constructed differentially private microdata.
        '''
        raw_file, raw_flag = self.data_handler.raw_contingency_vectors_file
        noisy_file, noisy_flag = self.data_handler.noisy_contingency_vectors_file

        self.initialize()
        if raw_flag: self.save_contingency_vectors(raw_file)

        if noisy_file is None or noisy_flag: self.measurement_phase()
        if noisy_flag: self.save_contingency_vectors(noisy_file)

        self.estimation_phase()

        noisy_data = self.construct_microdata()
        return noisy_data
    
    def check_correctness(self) -> None:
        '''Checks the correctness of the tree structure considering that its childs sums up to the parent node.
        '''
        print(f'Checking correctness of the tree...')
        time1 = time.time()
        for node in self.tree.nodes:
            if not node.children_range is None:
                node_sum = np.sum(node.contingency_vector)
                children_sum = 0 

                start, end = node.children_range
                for index in range(start, end+1):
                    children_sum += np.sum(self.tree.nodes[index].contingency_vector)

                if node_sum != children_sum:            
                    print(f'\nError: The sum of the contingency vectors of the children nodes is not equal to the parent node\'s contingency vector.')
                    print(f'Parent node contingency vector: {node.contingency_vector}')
                    raise ValueError('Tree correctness check failed.')

        time2 = time.time()
        print(f'Finished checking correctness in {time2-time1} seconds.\n')
        return None
