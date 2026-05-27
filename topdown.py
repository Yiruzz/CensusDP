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

        self.tree: HierarchicalTree = HierarchicalTree(constraints=[])
        self.optimizer = (optimizer, solver_options, optimizer_path)

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
        if isinstance(self.Q, QueryWorkload):
            self.Q = self.Q.build(self.data_handler.contingency_df)
        elif not isinstance(self.Q, np.ndarray):
            # No workload set — use identity. NOTE: np.eye(n_cells) is dense; avoid for large domains.
            self.Q = np.eye(len(self.data_handler.contingency_df))
        # NOTE: Privacy guarantees rely on Q being binary so that the L1 sensitivity (max column sum) is well defined
        #       and coincides with the squared L2 sensitivity. If Q is not binary, the privacy guarantees may not hold.
        assert np.all((self.Q == 0) | (self.Q == 1)), \
            "Q must be binary (entries in {0,1}) for the column-sum sensitivity reasoning."
        self.query_sensitivity = int(self.Q.sum(axis=0).max())
        print(f'\n  Query matrix: n_queries={self.Q.shape[0]}, sensitivity={self.query_sensitivity}')
        print(f'  Privacy mechanism: {self.privacy_mechanism.report_guarantee()}')
        self.tree = self.data_handler.build_hierarchical_tree(self.constraints, self.Q)
        print(f'{time.time() - t1:.2f} seconds.\n')

    def add_noise_to_chunk(self, start: int, end: int) -> Tuple[int, int, float]:
        '''Apply noise to a chunk of contingency vectors.

        Args:
            start (int): Starting index of the chunk.
            end (int): Ending index of the chunk (exclusive).

        Returns:
            Tuple[int, int, float]: The start index, end index, and execution time for processing the chunk.
        '''
        t1 = time.time()
        # Only the first n_queries slots of each row hold the live measurement y; the
        # trailing slots (when vector_length > n_queries) are reserved padding for x_hat
        # and must not be touched in this phase.
        n_queries = self.tree.n_queries
        for idx in range(start, end):
            self.add_noise(self.tree._contingency_vectors[idx, :n_queries], self.tree.nodes[idx].level)

        t2 = time.time()-t1
        return start, end, t2
        
    def add_noise(self, contingency_vector: np.ndarray, level: int) -> None:
        '''Add noise to the contingency vector using the configured privacy mechanism, in place.

        Args:
            contingency_vector (np.ndarray): The vector to be noised (modified in place).
            level (int): Tree level — selects the per-level privacy parameter inside the mechanism.
        '''
        self.privacy_mechanism.add_noise(contingency_vector, level, self.query_sensitivity)

    def measurement_phase(self) -> None:
        '''Perform the measurement phase of the TopDown algorithm.

        Each node's contingency_vector already holds y = Q @ x from tree construction.
        This phase adds discrete noise calibrated to the sensitivity of Q in place: y <- y + noise.
        '''
        t1 = time.time()
        print(f'Running measurement phase...\n')

        # Create chunks to distribute among workers.
        # Since the data is independent, workers do not require synchronization.
        num_chunks = self.workers
        base_chunk_size = self.tree._node_count // num_chunks
        remaining_nodes = self.tree._node_count % num_chunks

        chunks = []
        start = 0

        for i in range(num_chunks):
            extra = 1 if i < remaining_nodes else 0
            end = start + base_chunk_size + extra
            chunks.append((start, end))
            start = end

        # Create a thread pool.
        # Each worker receives a chunk.
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for chunk in chunks:
                print(f"Processing nodes {chunk[0]} until {chunk[1]}...")
                futures.append(executor.submit(self.add_noise_to_chunk, *chunk))

            print("\n", end="")

            # The vectors are modified in-place,
            # so only execution time is returned for logging purposes.
            for future in as_completed(futures):
                start, end, finished_time = future.result()
                print(f"Finshed chunk nodes {start} until {end} in {finished_time:.2f} seconds")
            
        print(f"\nMeasurement phase completed in {time.time() - t1:.2f} seconds.\n")

    def root_estimation_phase(self, index: int = 0) -> None:
        '''Perform the estimation phase of the TopDown algorithm for root node.

        Reads the noisy measurement y from the first n_queries slots of the row, then
        overwrites the first n_cells slots with x_hat. The row stays the same physical size.
        '''
        optimizer = OptimizationModel(*self.optimizer)
        root = self.tree.nodes[index]
        n_queries = self.tree.n_queries
        n_cells = self.tree.n_cells
        x_tilde = optimizer.non_negative_real_estimation(
            noisy_measurements=self.tree._contingency_vectors[index, :n_queries],
            node_id=root.id,
            constraints=root.constraints,
            query_matrix=self.Q
        )
        self.tree._contingency_vectors[index, :n_cells] = optimizer.rounding_estimation(
            x_tilde=x_tilde,
            node_id=root.id,
            constraints=root.constraints
        )
        return None

    def subtree_estimation_phase(self) -> None:
        '''Allows solving optimization models in parallel once their contingency vectors are updated.
        
        This means that it is not necessary to wait for an entire level to finish before moving to the next,
        because the executor processes tasks in the order they are submitted, but each task may take a different amount of time.
        '''
    
        root = self.tree.nodes[0]
        child_constraints = {child.id: child.constraints for child in root.children}
    
        root_arguments = (
            root.id,
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
                        node.id,
                        {child.id: child.constraints for child in node.children},
                    )
                    fut = executor.submit(solve, *child_arguments)
                    futures[fut] = child_arguments

        return None

    def estimation_phase(self) -> None:
        '''Perform the estimation phase of the TopDown algorithm.

        This method solves optimization problems at each node in the hierarchical tree to ensure
        consistency and adherence to constraints after noise has been added.

        Reads node.contingency_vector as the noisy measurement y (shape (n_queries,)) for any
        node whose parent has already been estimated, and overwrites it with x_hat
        (shape (n_cells,)) once that node is itself estimated. Top-down level order
        guarantees parents are converted to x_hat before their children are read.
        '''
        t1 = time.time()
        print(f'Running estimation phase...\n')
        self.root_estimation_phase()
        self.subtree_estimation_phase()
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