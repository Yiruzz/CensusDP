import pandas as pd
import numpy as np
from hierarchical_tree import HierarchicalTree
from data_handler import DataHandler
from optimizer import OptimizationModel
from constraints.constraint import Constraint

from discretegauss import sample_dlaplace, sample_dgauss

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
                 microdata_path: str, tree_path: str, pree_tree: bool,
                 contingency_vectors_folder_path: str, pree_contigency_vectors: bool,
                 optimizer: str, solver_options: dict) -> None:
        '''
        Initialize the TopDown algorithm.

        Args:
            data_path (str): Path to the input data file.
            hierarchy (List[str]): List of columns representing the hierarchy levels.
            last_hierarchical_column (str): The last column in the hierarchy that will be considered.
            queries (List[str]): List of columns to be queried and aggregated.
            microdata_path (str): Path to save the processed microdata.
            tree_path (str): Path to save the hierarchical tree. 
            pree_tree (bool): Flag indicating whether a pre-built tree is provided.
            contingency_vectors_folder_path (str): Path to the folder containing contingency vectors.
            pree_contigency_vectors (bool): Flag indicating whether pre-built contingency vectors are provided.
            optimizer (str): The optimization solver to use ('gurobi', 'ipopt', 'glpk', etc.).
            solver_options (dict): Dictionary of options to pass to the solver.

        Attributes:
            data_handler (DataHandler): Instance of DataHandler for managing data operations.

            hierarchical_columns (List[str]): List of columns representing the hierarchy levels.
            query_columns (List[str]): List of columns to be queried and aggregated.

            privacy_parameters (List[float]): List of privacy parameters for each level of the tree.
            mechanism (str): The noise mechanism to use ('discrete_laplace' or 'discrete_gaussian').

            pre_tree (bool): Flag indicating whether a pre-built tree is provided.
            pre_contigency_vectors (bool): Flag indicating whether pre-built contingency vectors are provided.

            tree (HierarchicalTree): Instance of HierarchicalTree representing the hierarchical structure.
            optimizer (OptimizationModel): Instance of OptimizationModel for solving optimization problems.
            constraints (Dict[int, List[Constraint]]): Dictionary mapping tree levels to their constraints
        '''

        self.data_handler: DataHandler = DataHandler(input_path=data_path,
                                                     hierarchical_columns=hierarchy,
                                                     query_columns=queries,
                                                     tree_path=tree_path,
                                                     contingency_vectors_folder_path=contingency_vectors_folder_path,
                                                     output_path=microdata_path)
        
        self.hierarchical_columns: List[str] = hierarchy
        self.index_last_hierarchical_column: int = self.hierarchical_columns.index(last_hierarchical_column)
        self.query_columns: List[str] = queries

        self.privacy_parameters: List[float] = []
        self.mechanism: Callable = lambda x: x  # Default to identity function

        self.pre_tree: bool = pree_tree
        self.pre_contigency_vectors: bool = pree_contigency_vectors

        self.tree: HierarchicalTree = HierarchicalTree(nodes=[], node_ranges_by_level=[])
        self.optimizer: OptimizationModel = OptimizationModel(optimizer, solver_options)
        self.constraints: Dict[int, List[Constraint]] = {}
    
    def initialize(self) -> None:
        '''Initialize the TopDown algorithm.
        This involves loading an existing tree or building it from the data, then generating
        the contingency vectors for each node in the tree and setting their constraints.
        '''

        print(f'Initializing TopDown algorithm...')

        # Read the data
        t1 = time.time()
        print(f'Reading data...', end=' ')
        self.data_handler.read_data(sep=';')
        print(f'{time.time() - t1:.2f} seconds.')

        # Create the hierarchical tree structure based on the hierarchical columns if no tree path is provided
        if not self.pre_tree:
            # If no tree path is provided, that meaning that the data passed is not sorted by the hierarchical columns
            t1 = time.time()
            print(f'Sorting data by hierarchical columns...', end=' ')
            self.data_handler.sort_data_by_hierarchy()
            print(f'{time.time() - t1:.2f} seconds.')
            
            original_file = self.data_handler.data_path
            sorted_file = self.data_handler.data_path.replace('.csv', '_sorted.csv')

            t1 = time.time()
            print(f'Write sorted data by hierarchical columns...', end=' ')
            self.data_handler.write_data(self.data_handler.dataframe, out_path=sorted_file)
            print(f'{time.time() - t1:.2f} seconds.')

            self.data_handler.data_path = sorted_file
            #self.data_handler.compress_file(original_file, should_delete_original=True)

            t1 = time.time()
            print(f'Creating hierarchical tree structure...', end=' ')
            self.data_handler.create_tree()
            print(f'{time.time() - t1:.2f} seconds.')

        # Reduce the data to the necessary columns 
        self.data_handler.hierarchical_columns = self.hierarchical_columns[:self.index_last_hierarchical_column+1]
        self.data_handler.reduce_data()

        # Load the tree until the hierarchical level specified
        print(f'Loading existing tree...', end=' ')
        t1 = time.time()
        self.tree = self.data_handler.load_tree()
        print(f'{time.time() - t1:.2f} seconds.')
    
        print(f'\nHierarchical tree structure loaded:')
        print(self.tree)

        self.data_handler.generate_contingency_dataframe()

        t1 = time.time()
        print(f'Creating contingency vectors for each node in the tree...', end=' ')
        self.data_handler.create_contingency_vectors(self.tree)
        print(f'{time.time() - t1:.2f} seconds.')

        # Set constraints for each node in the tree
        t1 = time.time()
        print(f'Setting constraints for each node in the tree...', end=' ')
        self.data_handler.add_constraints(self.tree, self.constraints)
        print(f'{time.time() - t1:.2f} seconds.')

        return None

    def measurement_phase(self) -> None:
        '''Perform the measurement phase of the TopDown algorithm.
        
        This method adds noise to the data at each node in the hierarchical tree according to the specified
        privacy parameters and mechanism.
        '''
        t1 = time.time()
        print(f'Running measurement phase...', end=' ')
        for node in self.tree.nodes:
            self.add_noise(node.contingency_vector, self.privacy_parameters[node.level])
        print(f'{time.time() - t1:.2f} seconds.')
        return None

    def estimation_phase(self) -> None:
        '''Perform the estimation phase of the TopDown algorithm.
        
        This method solves optimization problems at each node in the hierarchical tree to ensure
        consistency and adherence to constraints after noise has been added.
        '''
        t1 = time.time()
        print(f'Running estimation phase...', end=' ')

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
    
    def add_noise(self, contingency_vector: np.ndarray, privacy_budget: float,) -> np.ndarray:
        '''Add noise to the contingency vector using the specified mechanism.
        
        Args:
            contingency_vector (np.ndarray): The original contingency vector.
            privacy_budget (float): The privacy budget (epsilon) for noise addition.
        
        Returns:
            np.ndarray: The noisy contingency vector.
        '''

        for i in range(len(contingency_vector)):
            contingency_vector[i] += self.mechanism(privacy_budget)
        
        return contingency_vector

    def construct_microdata(self) -> pd.DataFrame:
        '''Construct the differentially private microdata from the hierarchical tree.
        
        This method generates the final microdata by traversing the hierarchical tree and
        aggregating the data from each node.

        Returns:
            pd.DataFrame: The constructed differentially private microdata.
        '''
        print(f'Constructing microdata from hierarchical tree...', end=' ')
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

    
    def discrete_gaussian(self, rho: float) -> int:
        '''Applies discrete Gaussian noise to the contingency vector.
        
        Args:
            rho (float): The privacy parameter.
        
        Returns:
            int: The noise value to be added.
        '''
        return sample_dgauss(rho)
    
    def discrete_laplace(self, epsilon: float) -> int:
        '''Applies Laplace noise to the contingency vector.
        
        Args:
            epsilon (float): The privacy parameter.
        
        Returns:
            int: The noise value to be added.
        '''
        return sample_dlaplace(1/epsilon)
    
    def set_mechanism(self, mechanism: str) -> None:
        '''Set the noise mechanism to use for adding noise to the data.
        
        Args:
            mechanism (str): The noise mechanism to use ('discrete_laplace' or 'discrete_gaussian').
        '''
        match mechanism:
            case 'discrete_laplace':
                self.mechanism = self.discrete_laplace
            case 'discrete_gaussian':
                self.mechanism = self.discrete_gaussian
            case _:
                 raise ValueError("Mechanism must be either 'discrete_laplace' or 'discrete_gaussian'.")

    def run(self) -> pd.DataFrame:
        '''Run the TopDown algorithm end-to-end.
        
        This method executes the full TopDown algorithm, including initialization,
        measurement phase, estimation phase, and microdata construction.

        Returns:
            pd.DataFrame: The constructed differentially private microdata.
        '''
        self.initialize()
        self.measurement_phase()
        self.estimation_phase()
        noisy_data = self.construct_microdata()
        return noisy_data
    
    def check_correctness(self) -> None:
        '''Checks the correctness of the tree structure considering that its childs sums up to the parent node.
        '''
        if self.tree.root is not None:
            print(f'Checking correctness of the tree...')
            time1 = time.time()
            self._check_correctness_node(self.tree.root)
            time2 = time.time()
            print(f'Finished checking correctness in {time2-time1} seconds.\n')


    def _check_correctness_node(self, node) -> None:
        '''Checks that the sum of the values of the current node are equal to the sum of the values of its children.

        Args:
            node (HierarchicalNode): The node to check.
        '''
        if node.children:
            # Check if the sum of the contingency vectors of the children nodes is equal to the parent node's contingency vector
            node_sum = np.sum(node.contingency_vector)
            children_sum = 0
            for child in node.children:
                children_sum += np.sum(child.contingency_vector)

            if node_sum != children_sum:            
                print(f'\nError: The sum of the contingency vectors of the children nodes is not equal to the parent node\'s contingency vector.')
                print(f'Parent node contingency vector: {node.contingency_vector}')
                raise ValueError('Tree correctness check failed.')
            else:
                for child in node.children:
                    self._check_correctness_node(child)