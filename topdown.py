import pandas as pd
import numpy as np
from hierarchical_tree import HierarchicalTree
from data_handler import DataHandler
from optimizer import OptimizationModel
from constraints.constraint import Constraint

import noisy

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory, get_context
from typing import Callable, Dict, List, Tuple
import time

def attach_memory(name, shape, dtype):
    global shm, arr
    shm = shared_memory.SharedMemory(name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

def add_noise(start: int, end: int, sampler_name: str, samples: int, privacy_budget: List[Tuple[int, float]], sensibility: int = 1) -> Tuple[int, int, float]:
    sampler = getattr(noisy, sampler_name)

    idx_privacy_budget = 0
    next_change = (privacy_budget[idx_privacy_budget + 1][0] if len(privacy_budget) > 1 else float("inf"))
    privacy_budget_value = privacy_budget[idx_privacy_budget][1]

    t1 = time.time()
    for idx in range(start, end):
        if idx == next_change:
            idx_privacy_budget += 1
            privacy_budget_value = privacy_budget[idx_privacy_budget][1]

            if idx_privacy_budget + 1 < len(privacy_budget):
                next_change = privacy_budget[idx_privacy_budget + 1][0]
            else:
                next_change = float("inf")
        arr[idx] += sampler(sensibility / privacy_budget_value, samples)

    shm.close()
    return start, end, (time.time()-t1) 

class TopDown():
    '''Represents the TopDown algorithm for generating differentially private microdata.
    
    The algorithm works by constructing a hierarchical tree structure, then adding noise to the data
    considering differential privacy principles. It propagates the noise to each node in the tree and
    finally it solves optimization problems to ensure consistency across the tree and adherence to 
    specified constraints by the user.
    '''
    def __init__(self, data_path: str, hierarchy: List[str], queries: List[str], out_path: str = 'noisy_data.csv', optimizer='gurobi', solver_options={}, optimizer_path=None) -> None:
        '''
        Initialize the TopDown algorithm.

        Args:
            data_path (str): Path to the input data file.
            hierarchy (List[str]): List of columns representing the hierarchy levels.
            queries (List[str]): List of columns to be queried and aggregated.
            out_path (str): Path to save the processed data. Defaults to 'noisy_data.csv'.
            optimizer (str): The optimization solver to use ('gurobi', 'ipopt', 'glpk', etc.). Defaults to 'gurobi'.
            solver_options (dict): Dictionary of options to pass to the solver. If None, defaults to empty dict.
            optimizer_path (str): Path to the optimizer executable. If None, defaults to None.

        Attributes:
            data_handler (DataHandler): Instance of DataHandler for managing data operations.

            hierarchical_columns (List[str]): List of columns representing the hierarchy levels.
            query_columns (List[str]): List of columns to be queried and aggregated.

            privacy_parameters (List[float]): List of privacy parameters for each level of the tree.
            mechanism (str): The noise mechanism to use ('discrete_laplace' or 'discrete_gaussian').

            tree (HierarchicalTree): Instance of HierarchicalTree representing the hierarchical structure.
            optimizer (OptimizationModel): Instance of OptimizationModel for solving optimization problems.

            constraints (Dict[int, List[Constraint]]): Dictionary mapping tree levels to their constraints



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
        self.optimizer: OptimizationModel = OptimizationModel(optimizer, solver_options, optimizer_path)

        self.workers: int = 4
        
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
        self.tree = self.data_handler.build_hierarchical_tree(self.constraints)
        print(f'{time.time() - t1:.2f} seconds.\n')

    def measurement_phase(self) -> None:
        '''Perform the measurement phase of the TopDown algorithm.
        
        This method adds noise to the data at each node in the hierarchical tree according to the specified
        privacy parameters and mechanism.
        '''
        t1 = time.time()
        print(f'Running measurement phase...\n')

        # TODO: change form to construct chunks, for example, considering
        # a constanst value of nodes like 1000. This form if occurs a exception,
        # can repeat the chunk.
        sampler_name = self.mechanism.__name__
        samples = self.data_handler.contingency_df_length
        chunk_size = self.tree._node_count // self.workers
        rest = self.tree._node_count % self.workers

        chunks = []
        start = 0
        for i in range(self.workers):
            extra = 1 if i < rest else 0
            end = start + chunk_size + extra
            chunks.append((start, end))
            start = end

        privacy_parameters_chunk = []
        max_level = 1 + len(self.hierarchical_columns)

        for start, end in chunks:
            privacy_parameters = []
            idx = start
            level = self.tree.nodes[idx].level

            while idx < end and level < max_level:
                privacy_parameters.append((idx, self.privacy_parameters[level]))

                level += 1
                if level >= max_level:
                    break

                idx = self.tree._levels[level]
            privacy_parameters_chunk.append(privacy_parameters)

        # Create a pool with process
        with ProcessPoolExecutor(max_workers=self.workers,  mp_context=get_context("spawn"),
                                 initializer=attach_memory,
                                 initargs=(self.tree._contingency_vectors_shm.name,
                                           (self.tree._node_count, self.data_handler.contingency_df_length),
                                           self.data_handler.dtype)) as executor:
            
            # Send tasks
            futures = []
            for i in range(len(chunks)):
                futures.append(executor.submit(add_noise, *chunks[i], sampler_name, samples, privacy_parameters_chunk[i]))
            
            # Retrive results
            for f in futures:
                start_chunk, end_chunk, time_chunk = f.result()
                print(f"Chunk ({start_chunk}, {end_chunk}( finished in {time_chunk:.2f} seconds")

        print(f'Measurement phase completed in {time.time() - t1:.2f} seconds.\n')

    def estimation_phase(self) -> None:
        '''Perform the estimation phase of the TopDown algorithm.
        
        This method solves optimization problems at each node in the hierarchical tree to ensure
        consistency and adherence to constraints after noise has been added.
        '''
        t1 = time.time()
        print(f'Running estimation phase...')

        # Root estimation (level 0)
        # Does not require consistency adjustments
        t2 = time.time()
        print(f'\nProcessing root node (level 0)... ', end=' ')
        idx = 0
        root = self.tree.nodes[idx]
        x_tilde: np.ndarray = self.optimizer.non_negative_real_estimation(
            contingency_vector=self.tree._contingency_vectors[idx],
            id_node=root.id,
            constraints=root.constraints
        )
        self.tree._contingency_vectors[idx] = self.optimizer.rounding_estimation(
            x_tilde=x_tilde,
            id_node=root.id,
            constraints=root.constraints
        )
        print(f'{time.time() - t2:.2f} seconds.')

        # Now process the rest of the tree level by level
        for level, nodes in self.tree.iterate_by_levels():
            t2 = time.time()
            if len(nodes[0].children) != 0: print(f'Processing level {level+1}...', end=' ')
            for node in nodes:
                # If the node is a leaf, no need to solve optimization
                # NOTE: With a break we assume that all leaves are at the same level
                if len(node.children) == 0:
                    break
                
                # Solve the optimization problem for the children of the current node
                joint_contingency_vector = self.tree.combine_vectors(node)

                # Transform individual constraints for joint vector
                joint_constraints = self.tree.combine_child_constraints(node, joint_contingency_vector)
                    
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
                self.tree.update_vectors(node, joint_solution)

            if len(nodes[0].children) != 0: print(f'{time.time() - t2:.2f} seconds.')
        print(f'Estimation phase completed in {time.time() - t1:.2f} seconds.\n')
        
        return None

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
        self.data_handler.write_data(data=noisy_df, cols=self.hierarchical_columns+self.query_columns)
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
    
    def set_mechanism(self, mechanism: str) -> None:
        '''Set the noise mechanism to use for adding noise to the data.
        
        Args:
            mechanism (str): The noise mechanism to use ('discrete_laplace' or 'discrete_gaussian').
        '''
        match mechanism:
            case 'discrete_laplace':
                self.mechanism = noisy.sample_dlaplace_optimized
            case 'discrete_gaussian':
                self.mechanism = noisy.sample_dgauss_optimized
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
        root = self.tree.nodes[0]
        if root is not None:
            print(f'Checking correctness of the tree...')
            time1 = time.time()
            self._check_correctness_node(root)
            time2 = time.time()
            print(f'Finished checking correctness in {time2-time1} seconds.\n')


    def _check_correctness_node(self, node) -> None:
        '''Checks that the sum of the values of the current node are equal to the sum of the values of its children.

        Args:
            node (HierarchicalNode): The node to check.
        '''
        if node.children:
            # Check if the sum of the contingency vectors of the children nodes is equal to the parent node's contingency vector
            node_sum = sum(self.tree._contingency_vectors[node.id])
            children_sum = 0
            for child in node.children:
                children_sum += np.sum(self.tree._contingency_vectors[child.id])

            if node_sum != children_sum:      
                print(node_sum, children_sum)      
                print(f'\nError: The sum of the contingency vectors of the children nodes is not equal to the parent node\'s contingency vector.')
                print(f'Parent node contingency vector: {self.tree._contingency_vectors[node.id]}')
                raise ValueError('Tree correctness check failed.')
            else:
                for child in node.children:
                    self._check_correctness_node(child)