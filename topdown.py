import pandas as pd
import numpy as np
from hierarchical_tree import HierarchicalTree
from data_handler import DataHandler
from optimizer import OptimizationModel
from constraints.constraint import Constraint

from discretegauss import sample_dlaplace, sample_dgauss

from typing import Callable, Dict, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from itertools import count
import time

process_solver = None

def init_process(solver_name: str, solver_options: Dict[str, Any]) -> None:
    ''' Initialize the solver instance for the process using the user-provided parameters.
    
    The solver instance is stored in a global variable for access by tasks executed in this process.

    Args:
        solver_name (str): Name of the solver to use (e.g., 'gurobi').
        solver_options (Dict[str, Any]): Dictionary of options to configure the solver.
    '''
    global process_solver
    process_solver = OptimizationModel(solver_name, solver_options)
    return None

def solve(node_id: int, joint_contingency_vector: np.ndarray[int], joint_constraints: List[Callable]) -> Tuple[int, np.ndarray[int]]:
    ''' Task executed in a separate process to solve a node in parallel.

    The solver instance is stored in a global variable for the process, 
    so it does not need to be passed as an argument each time.

    Args:
        node_id (int): Unique identifier for this hierarchical node.
        joint_contingency_vector (np.ndarray[int]): Contingency vector constructed from the node's children.
        joint_constraints (List[Callable]): List of constraints associated with the joint contingency vector.

    Returns:
        Tuple[int, np.ndarray[int]]: A tuple containing the node ID and the resulting solution of the problem.
                                     This allows mapping the solution back to the corresponding node.
    '''
    x_tilde = process_solver.non_negative_real_estimation(
        contingency_vector=joint_contingency_vector,
        id_node=node_id,
        constraints=joint_constraints
    )
    joint_solution: np.ndarray = process_solver.rounding_estimation(
        x_tilde=x_tilde,
        id_node=node_id,
        constraints=joint_constraints
    )
    return node_id, joint_solution

class TopDown():
    '''Represents the TopDown algorithm for generating differentially private microdata.
    
    The algorithm works by constructing a hierarchical tree structure, then adding noise to the data
    considering differential privacy principles. It propagates the noise to each node in the tree and
    finally it solves optimization problems to ensure consistency across the tree and adherence to 
    specified constraints by the user.
    '''
    def __init__(self, data_path: str, hierarchy: List[str], queries: List[str], out_path: str = 'noisy_data.csv', optimizer='gurobi', solver_options={}) -> None:
        '''
        Initialize the TopDown algorithm.

        Args:
            data_path (str): Path to the input data file.
            hierarchy (List[str]): List of columns representing the hierarchy levels.
            queries (List[str]): List of columns to be queried and aggregated.
            out_path (str): Path to save the processed data. Defaults to 'noisy_data.csv'.
            optimizer (str): The optimization solver to use ('gurobi', 'ipopt', 'glpk', etc.). Defaults to 'gurobi'.
            solver_options (dict): Dictionary of options to pass to the solver. If None, defaults to empty dict.

        Attributes:
            data_handler (DataHandler): Instance of DataHandler for managing data operations.

            hierarchical_columns (List[str]): List of columns representing the hierarchy levels.
            query_columns (List[str]): List of columns to be queried and aggregated.

            privacy_parameters (List[float]): List of privacy parameters for each level of the tree.
            mechanism (str): The noise mechanism to use ('discrete_laplace' or 'discrete_gaussian').

            constraints (Dict[int, List[Constraint]]): Dictionary mapping tree levels to their constraints

            tree (HierarchicalTree): Instance of HierarchicalTree representing the hierarchical structure.
            optimizer (Tuple[str, Dict[str, Any]]): Tuple containing the solver name and its configuration options used to create an OptimizationModel instance.
            max_workers (int): Number of processes used to parallelize the main stages.
        '''
        self.data_handler: DataHandler = DataHandler(file_path=data_path, output_path=out_path)
        self.hierarchical_columns: List[str] = hierarchy
        self.query_columns: List[str] = queries

        self.data_handler.hierarchical_columns = hierarchy
        self.data_handler.query_columns = queries

        self.privacy_parameters: List[float] = []
        self.mechanism: Callable = lambda x: x  # Default to identity function

        self.constraints: Dict[int, List[Constraint]] = {}

        self.tree: HierarchicalTree = HierarchicalTree(constraints=[])
        self.optimizer: Tuple[str, Dict[str, Any]] = (optimizer, solver_options)
        self.max_workers: int = 4
        
        #self.constraints: List[List[Callable]] = []
        # self.processed_data: pd.DataFrame = None
        # self.distance_metric: Optional[str] = None
    
    def initialize(self) -> None:
        '''Initialize the TopDown algorithm.
        
        This method sets up the necessary components for the algorithm to run, such as data handling and tree structure.
        '''
        print(f'Initializing TopDown algorithm...')
        t1 = time.time()
        print(f'Reading data from {self.data_handler.file_path}...', end=' ')
        self.data_handler.read_data(self.hierarchical_columns + self.query_columns, sep=';')
        print(f'{time.time() - t1:.2f} seconds.')

        t1 = time.time()
        print(f'Generating contingency dataframe...', end=' ')
        self.data_handler.generate_contingency_dataframe(self.query_columns)
        print(f'{time.time() - t1:.2f} seconds.')

        t1 = time.time()
        print(f'Building hierarchical tree...', end=' ')
        self.tree = self.data_handler.build_hierarchical_tree(count(start=0), self.constraints)
        print(f'{time.time() - t1:.2f} seconds.\n')

        return None

    def measurement_phase(self) -> None:
        '''Perform the measurement phase of the TopDown algorithm.
        
        This method adds noise to the data at each node in the hierarchical tree according to the specified
        privacy parameters and mechanism.
        '''

        # TODO: Parallelize this stage; we can divide the list of nodes into roughly equal parts for each process.
        t1 = time.time()
        print(f'Running measurement phase...\n')
        for level, nodes in self.tree.iterate_by_levels():
            t2 = time.time()
            print(f'Processing level {level} with {len(nodes)} nodes...', end=' ')
            privacy_budget = self.privacy_parameters[level]
            for node in nodes:
                self.add_noise(node.contingency_vector, privacy_budget)
            print(f'{time.time() - t2:.2f} seconds.')
        print(f'Measurement phase completed in {time.time() - t1:.2f} seconds.\n')
        
        return None

    def root_estimation_phase(self) -> None:
        '''Perform the estimation phase of the TopDown algorithm for root node.

        '''
        optimizer = OptimizationModel(*self.optimizer)
        x_tilde: np.ndarray = optimizer.non_negative_real_estimation(
            contingency_vector=self.tree.root.contingency_vector,
            id_node=self.tree.root.node_id,
            constraints=self.tree.root.constraints
        )
        self.tree.root.contingency_vector = optimizer.rounding_estimation(
            x_tilde=x_tilde,
            id_node=self.tree.root.node_id,
            constraints=self.tree.root.constraints
        )
        return None
    
    def subtree_estimation_phase(self) -> None:
        ''' 
        Allows solving optimization models in parallel once their contingency vectors are updated.
        
        This means that it is not necessary to wait for an entire level to finish before moving to the next,
        because the executor processes tasks in the order they are submitted, but each task may take a different amount of time.
        There is no explicit waiting for a level to complete.
        '''

        # Create arguments from the root node to pass updated vectors to regional nodes.
        joint_contingency_vector = self.tree.root.combine_child_vectors()
        root_arguments = (self.tree.root.node_id, joint_contingency_vector, self.tree.root.combine_child_constraints(joint_contingency_vector))

        # Create a process pool with workers, each with its own environment (for Gurobi)
        # This avoids the cost of creating a new environment for each thread; we can reuse the environments
        # because no information about the solved models is saved—only the license and other parameters.
        # TODO: Investigate how 'cplex' or 'glpk' handle parallelism, whether they use an environment or something similar, 
        # because this might require modifying the init_process function.
        with ProcessPoolExecutor(max_workers=self.max_workers,
                                 initializer=init_process, initargs=self.optimizer) as executor:
            # Submit the root node task
            futures = {executor.submit(solve, *root_arguments): root_arguments}

            # While there are tasks running
            while futures:
                # Take the first task that completes
                done, _ = wait(futures, return_when=FIRST_COMPLETED)

                # Remove completed futures and retrieve their results
                for fut in done:
                    futures.pop(fut) 
                    node_id, joint_solution = fut.result()
                    node = self.tree._nodes[node_id]

                    # Update the child nodes associated with this node
                    node.update_child_vectors(joint_solution)

                    # Process the child nodes of the current node
                    # All children of this node are either internal nodes or leaves.
                    # If a child is not a leaf, submit a task to solve its sub-tree.
                    # If a child is a leaf, we stop submitting further tasks for this branch.
                    # TODO: Consider submitting child tasks gradually instead of all at once,
                    # when there is sufficient RAM, because having many futures in memory can be expensive.
                    for child in node.children:
                        if not child.is_leaf():
                            joint_contingency_vector = child.combine_child_vectors()
                            child_arguments = (child.node_id, joint_contingency_vector, child.combine_child_constraints(joint_contingency_vector))
                            fut = executor.submit(solve, *child_arguments)
                            futures[fut] = child_arguments
                        else: break
        return None

    def estimation_phase(self) -> None:
        '''Perform the estimation phase of the TopDown algorithm.
        
        This method solves optimization problems at each node in the hierarchical tree to ensure
        consistency and adherence to constraints after noise has been added.
        '''
        t1 = time.time()
        print(f'Running estimation phase...')

        t2 = time.time()
        print(f'\nProcessing root node (level 0)... ', end=' ')
        self.root_estimation_phase()
        print(f'{time.time() - t2:.2f} seconds.')

        self.subtree_estimation_phase()
        print(f'Estimation phase completed in {time.time() - t1:.2f} seconds.\n')
        
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

        print(f'Writing noisy data to {self.data_handler.output_path}...', end=' ')
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
        if not self.hierarchical_columns: 
            raise ValueError("Hierarchical columns must be set before adding constraints to the tree.")
    
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
    #     node = self.tree.find_node_by_id(node_id)
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