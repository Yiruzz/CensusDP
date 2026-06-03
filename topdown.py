import pandas as pd
import numpy as np
from hierarchical_tree import HierarchicalTree
from data_handler import DataHandler
from optimizer import OptimizationModel
from constraints.constraint import Constraint
from queries import QueryWorkload
from privacy import PrivacyMechanism

from parallel_utils import init_process, solve

from collections import deque
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import Callable, Dict, List, Tuple, Optional, Union
import time

class TopDown():
    '''Represents the TopDown algorithm for generating differentially private microdata.
    
    The algorithm works by constructing a hierarchical tree structure, then adding noise to the data
    considering differential privacy principles. It propagates the noise to each node in the tree and
    finally it solves optimization problems to ensure consistency across the tree and adherence to 
    specified constraints by the user.
    '''
    def __init__(self, data_path: str, hierarchy: List[str], query_columns: List[str],
                 privacy_mechanism: PrivacyMechanism,
                 out_path: str = 'noisy_data.csv', optimizer='gurobi', solver_options={}, optimizer_path=None) -> None:
        '''
        Initialize the TopDown algorithm.

        Args:
            data_path (str): Path to the input data file.
            hierarchy (List[str]): List of columns representing the hierarchy levels.
            query_columns (List[str]): List of columns to be queried and aggregated.
            privacy_mechanism (PrivacyMechanism): DP variant carrying per-level parameters.
                Length of mechanism.level_params must equal len(hierarchy) + 1 (root + per-column levels).
            out_path (str): Path to save the processed data. Defaults to 'noisy_data.csv'.
            optimizer (str): The optimization solver to use ('gurobi', 'ipopt', 'glpk', etc.). Defaults to 'gurobi'.
            solver_options (dict): Dictionary of options to pass to the solver. If None, defaults to empty dict.
            optimizer_path (str): Path to the optimizer executable. If None, defaults to None.

        Attributes:
            data_handler (DataHandler): Instance of DataHandler for managing data operations.

            hierarchical_columns (List[str]): List of columns representing the hierarchy levels.
            query_columns (List[str]): List of columns to be queried and aggregated.

            privacy_mechanism (PrivacyMechanism): The DP variant + per-level parameters.

            tree (HierarchicalTree): Instance of HierarchicalTree representing the hierarchical structure.
            optimizer (OptimizationModel): Instance of OptimizationModel for solving optimization problems.

            constraints (Dict[int, List[Constraint]]): Dictionary mapping tree levels to their constraints
        '''
        n_levels = len(hierarchy) + 1
        if len(privacy_mechanism.level_params) != n_levels:
            raise ValueError(
                f"privacy_mechanism has {len(privacy_mechanism.level_params)} per-level params; "
                f"expected {n_levels} (= len(hierarchy) + 1)."
            )

        self.data_handler: DataHandler = DataHandler(file_path=data_path, output_path=out_path)
        self.hierarchical_columns: List[str] = hierarchy
        self.query_columns: List[str] = query_columns

        self.data_handler.hierarchical_columns = hierarchy
        self.data_handler.query_columns = query_columns

        self.privacy_mechanism: PrivacyMechanism = privacy_mechanism

        self.Q: Union[QueryWorkload, np.ndarray, None] = None  # set via set_query_workload(); resolved in initialize()
        self.query_sensitivity: int = 1  # L1 sensitivity of Q; computed in initialize() once Q is materialized

        self.constraints: Dict[int, List[Constraint]] = {}

        self.tree: HierarchicalTree = HierarchicalTree()
        self.optimizer = (optimizer, solver_options, optimizer_path)

        self.workers: int = 4

    
    def initialize(self) -> None:
        '''Initialize the TopDown algorithm.

        This method reads data, generates the contingency dataframe, computes Q,
        and builds the hierarchical tree structure (without computing contingency vectors).
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

        # t1 = time.time()
        # print(f'Computing query matrix Q...', end=' ')
        # if isinstance(self.Q, QueryWorkload):
        #     self.Q = self.Q.build(self.data_handler.contingency_df)
        # elif not isinstance(self.Q, np.ndarray):
        #     self.Q = np.eye(len(self.data_handler.contingency_df))
        # assert np.all((self.Q == 0) | (self.Q == 1)), \
        #     "Q must be binary (entries in {0,1}) for the column-sum sensitivity reasoning."
        # self.query_sensitivity = int(self.Q.sum(axis=0).max())
        # print(f'\n  Query matrix: n_queries={self.Q.shape[0]}, sensitivity={self.query_sensitivity}')
        # print(f'  Privacy mechanism: {self.privacy_mechanism.report_guarantee()}')
        # print(f'{time.time() - t1:.2f} seconds.')

        t1 = time.time()
        print(f'Building hierarchical tree structure...', end=' ')
        self.tree = self.data_handler.build_hierarchical_tree()
        print(f'{time.time() - t1:.2f} seconds.\n')

        # Initialize output file with headers
        self.data_handler._initialize_output_file()

        print(self.tree)

    def estimation_phase(self) -> None:
        '''Perform the estimation phase of the TopDown algorithm.

        Traverses the tree using BFS. Materializes root and its children immediately,
        then for each node in queue: materializes its children, constructs microdata for leaves,
        and frees its own vector.
        '''
        print(f'Materializing contingency vectors and constructing microdata...', end=' ')
        t1 = time.time()

        # Materialize root and its children immediately
        root = self.tree.root
        root.contingency_vector = self.data_handler.create_contingency_vector(root.hierarchical_path)
        self.privacy_mechanism.add_noise(root.contingency_vector, root.level, self.query_sensitivity)

        queue = deque()
        for child in root.children:
            child.contingency_vector = self.data_handler.create_contingency_vector(child.hierarchical_path)
            self.privacy_mechanism.add_noise(child.contingency_vector, child.level, self.query_sensitivity)
            queue.append(child)

        # Free root's memory immediately after materializing children
        root.contingency_vector = None

        # Process remaining nodes
        while queue:
            node = queue.popleft()

            # Materialize all children's vectors
            for child in node.children:
                child.contingency_vector = self.data_handler.create_contingency_vector(child.hierarchical_path)
                self.privacy_mechanism.add_noise(child.contingency_vector, child.level, self.query_sensitivity)
                queue.append(child)

            # If node is a leaf, construct and write its microdata
            if node.is_leaf(): self.data_handler.write_microdata_for_leaf(node)

            # Free memory: delete current node's vector
            node.contingency_vector = None

        print(f'{time.time() - t1:.2f} seconds.\n')

    def root_estimation_phase(self, index: int = 0) -> None:
        '''Perform the estimation phase of the TopDown algorithm for root node.

        Reads the noisy measurement y from the first n_queries slots of the row, then
        overwrites the first n_cells slots with x_hat. The row stays the same physical size.
        '''
        optimizer = OptimizationModel(*self.optimizer)
        root = self.tree.root
        n_queries = self.tree.n_queries
        n_cells = self.tree.n_cells
        x_tilde = optimizer.non_negative_real_estimation(
            noisy_measurements=self.tree._contingency_vectors[index, :n_queries],
            node_id=root,
            constraints=root.constraints,
            query_matrix=self.Q
        )
        self.tree._contingency_vectors[index, :n_cells] = optimizer.rounding_estimation(
            x_tilde=x_tilde,
            node_id=root,
            constraints=root.constraints
        )
        return None

    def subtree_estimation_phase(self) -> None:
        '''Allows solving optimization models in parallel once their contingency vectors are updated.

        This means that it is not necessary to wait for an entire level to finish before moving to the next,
        because the executor processes tasks in the order they are submitted, but each task may take a different amount of time.
        '''

        root = self.tree.root
        child_constraints = {child: child.constraints for child in root.children}

        root_arguments = (
            root,
            child_constraints,
        )

        max_workers = self.workers
        buffer = 2
        max_outstanding = max_workers + buffer

        # Retrieve solver configuration and shared memory information.
        # Each process initializes its own solver instance and attaches
        # to the shared memory space on first execution.
        solver_name = self.optimizer[0]
        solver_options = self.optimizer[1]
        optimizer_path = self.optimizer[2]
        shared_memory_name = self.tree._contingency_vectors_shm.name
        shared_array_shape = self.tree._contingency_vectors.shape
        shared_array_dtype = str(self.tree._contingency_vectors.dtype)

        args = (
            solver_name,
            solver_options,
            optimizer_path,
            shared_memory_name,
            shared_array_shape,
            shared_array_dtype,
            self.tree.n_queries,
            self.tree.n_cells,
            self.Q,
        )

        nodes_per_level = [abs(self.tree._levels[i] - self.tree._levels[i + 1])for i in range(len(self.tree._levels) - 1)]
        time_per_level = np.zeros(len(self.hierarchical_columns))

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn"),
                                 initializer=init_process, initargs=args) as executor:
            next_ranges = deque()
            curr_range = None

            # Submit root
            futures = {
                executor.submit(solve, *root_arguments): root_arguments
            }

            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)

                # Process completed tasks
                for fut in done:
                    futures.pop(fut)
                    node_id, elapsed_time = fut.result()
                    node = self.tree.nodes[node_id]

                    time_per_level[node.level] += elapsed_time
                    nodes_per_level[node.level] -= 1

                    if nodes_per_level[node.level] == 0: print(f"Level {node.level} processed in {time_per_level[node.level]:.2f} seconds.")

                    # Add children range
                    next_ranges.append(iter(node.children))

                    # Initialize current range if needed
                    if curr_range is None and next_ranges:
                        curr_range = next_ranges.popleft()

                # Fill available slots
                while len(futures) < max_outstanding and curr_range is not None:
                    try:
                        node = next(curr_range)
                    except StopIteration:
                        if next_ranges:
                            curr_range = next_ranges.popleft()
                        else:
                            curr_range = None
                        continue

                    if node.is_leaf(): continue

                    child_arguments = (
                        node,
                        {child: child.constraints for child in node.children},
                    )
                    fut = executor.submit(solve, *child_arguments)
                    futures[fut] = child_arguments

        return None

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

    def set_query_workload(self, query_matrix: Union[QueryWorkload, np.ndarray]) -> None:
        '''Set the workload query matrix Q.

        Q is applied during tree construction: each node stores Q @ x instead of x.
        If not called, initialize() constructs np.eye(n_cells) as the default.

        Args:
            query_matrix: Either a QueryWorkload (DSL object, built lazily at initialize() time)
                          or a pre-built numpy ndarray of shape (n_queries, n_cells).
        '''
        self.Q = query_matrix

    def run(self) -> pd.DataFrame:
        '''Run the TopDown algorithm end-to-end.

        This method executes the full TopDown algorithm, including initialization,
        measurement phase, estimation phase, and microdata construction.

        Returns:
            pd.DataFrame: The constructed differentially private microdata.
        '''
        self.initialize()
        self.estimation_phase()
        self.tree.print_all_nodes()
    
        #self.estimation_phase()
  
    def check_correctness(self) -> None:
        '''Checks the correctness of the tree structure considering that its childs sums up to the parent node.
        '''
        root = self.tree.root
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
            node_sum = sum(self.tree._contingency_vectors[node])
            children_sum = 0
            for child in node.children:
                children_sum += np.sum(self.tree._contingency_vectors[child])

            if node_sum != children_sum:
                print(node_sum, children_sum)
                print(f'\nError: The sum of the contingency vectors of the children nodes is not equal to the parent node\'s contingency vector.')
                print(f'Parent node contingency vector: {self.tree._contingency_vectors[node]}')
                raise ValueError('Tree correctness check failed.')
            else:
                for child in node.children:
                    self._check_correctness_node(child)