import heapq
import numpy as np
import scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context

from hierarchical_tree import HierarchicalTree
from hierarchical_node import HierarchicalNode
from data_handler import DataHandler
from optimizer import OptimizationModel
from constraints.constraint import Constraint
from parallel_utils import init_process, estimate_and_update_children, _combine_child_constraints, _check_node_correctness, _real_and_round_estimation, _split_child_vectors
from queries import QueryWorkload
from privacy import PrivacyMechanism

from typing import Dict, List, Optional, Union
import time

class TopDown():
    '''Represents the TopDown algorithm for generating differentially private microdata.
    
    The algorithm works by constructing a hierarchical tree structure, then adding noise to the data
    considering differential privacy principles. It propagates the noise to each node in the tree and
    finally it solves optimization problems to ensure consistency across the tree and adherence to 
    specified constraints by the user.
    '''
    def __init__(self, data_path: str, hierarchy: List[str], query_columns: List[str],
                 privacy_mechanism: PrivacyMechanism, num_workers: int, out_path: str = 'noisy_data.csv',
                 solver_name: str = 'gurobi', solver_options: dict = {}, optimizer_path: Optional[str] = None,
                 domain: Optional[Dict[str, List]] = None, check_correctness: bool = False) -> None:
        '''
        Initialize the TopDown algorithm.

        Args:
            data_path (str): Path to the input data file.
            hierarchy (List[str]): List of columns representing the hierarchy levels.
            query_columns (List[str]): List of columns to be queried and aggregated.
            privacy_mechanism (PrivacyMechanism): DP variant carrying per-level parameters.
                Length of mechanism.level_params must equal len(hierarchy) + 1 (root + per-column levels).
            out_path (str): Path to save the processed data. Defaults to 'noisy_data.csv'.
            solver_name (str): The optimization solver to use ('gurobi', 'ipopt', 'glpk', etc.). Defaults to 'gurobi'.
            solver_options (dict): Dictionary of options to pass to the solver. If None, defaults to empty dict.
            optimizer_path (str): Path to the optimizer executable. If None, defaults to None.
            domain (Optional[Dict[str, List]]): Per-column set of all possible values for the
                query columns, defining the contingency cell space. Should be data-independent
                for a sound DP guarantee. When None (or a column omitted), the domain is inferred
                from the observed data with a warning. Passed through to DataHandler.
            num_workers (int): Number of parallel workers for the estimation phase. Defaults to 2.
            check_correctness (bool): Whether to check correctness during execution. Defaults to False.

        Attributes:
            data_handler (DataHandler): Instance of DataHandler for managing data operations.

            hierarchical_columns (List[str]): List of columns representing the hierarchy levels.
            query_columns (List[str]): List of columns to be queried and aggregated.

            privacy_mechanism (PrivacyMechanism): The DP variant + per-level parameters.

            tree (HierarchicalTree): Instance of HierarchicalTree representing the hierarchical structure.
            optimizer (OptimizationModel): Instance of OptimizationModel for solving optimization problems.

            constraints (Dict[int, List[Constraint]]): Dictionary mapping tree levels to their constraints.

            workers (int): Number of parallel workers for estimation phase.
        '''
        n_levels = len(hierarchy) + 1
        if len(privacy_mechanism.level_params) != n_levels:
            raise ValueError(
                f"privacy_mechanism has {len(privacy_mechanism.level_params)} per-level params; "
                f"expected {n_levels} (= len(hierarchy) + 1)."
            )

        self.hierarchical_columns: List[str] = hierarchy
        self.query_columns: List[str] = query_columns

        self.data_handler: DataHandler = DataHandler(file_path=data_path, output_path=out_path, domain=domain)
        self.data_handler.hierarchical_columns = hierarchy
        self.data_handler.query_columns = query_columns

        self.privacy_mechanism: PrivacyMechanism = privacy_mechanism

        self.Q: Union[QueryWorkload, np.ndarray, None] = None  # set via set_query_workload(); resolved in initialize()
        self.query_sensitivity: int = 1  # L1 sensitivity of Q; computed in initialize() once Q is materialized

        self.constraints: Dict[int, List[Constraint]] = {i: [] for i in range(len(hierarchy) + 1)}

        self.tree: HierarchicalTree = HierarchicalTree()

        self.optimizer: OptimizationModel = OptimizationModel(
            solver_name=solver_name,
            solver_options=solver_options,
            optimizer_path=optimizer_path
        )
        
        self.solver_options = solver_options

        self.workers = num_workers
        self.check_correctness = check_correctness

    def initialize(self) -> None:
        '''Initialize the TopDown algorithm.

        This method reads data, generates the contingency dataframe, computes Q,
        and builds the hierarchical tree structure (without computing contingency vectors).
        '''
        print(f'Initializing TopDown algorithm...')

        t1 = time.time()
        print(f'Converting CSV to Parquet if needed...', end=' ')
        self.data_handler.convert_csv_to_parquet()
        print(f'{time.time() - t1:.2f} seconds.')

        self.data_handler.create_data_view()

        t1 = time.time()
        print(f'Building contingency domain...', end=' ')
        self.data_handler.build_contingency_domain()
        print(f'{time.time() - t1:.2f} seconds.')

        t1 = time.time()
        print(f'Building query workload...', end=' ')
        if isinstance(self.Q, QueryWorkload):
            self.Q = self.Q.build(self.data_handler.contingency_domain)
        elif not (isinstance(self.Q, np.ndarray) or sp.issparse(self.Q)):
            # No workload set - use the sparse identity (each cell answered directly).
            n_cells = self.data_handler.contingency_domain.n_cells
            self.Q = sp.identity(n_cells, format='csr', dtype=float)
        # NOTE: Privacy guarantees rely on Q being binary so that the L1 sensitivity (max column sum) is well defined
        #       and coincides with the squared L2 sensitivity. If Q is not binary, the privacy guarantees may not hold.
        if sp.issparse(self.Q):
            assert np.all((self.Q.data == 0) | (self.Q.data == 1)), \
                "Q must be binary (entries in {0,1}) for the column-sum sensitivity reasoning."
            self.query_sensitivity = int(np.asarray(self.Q.sum(axis=0)).max())
        else:
            assert np.all((self.Q == 0) | (self.Q == 1)), \
                "Q must be binary (entries in {0,1}) for the column-sum sensitivity reasoning."
            self.query_sensitivity = int(self.Q.sum(axis=0).max())
        print(f'\n  Query matrix: n_queries={self.Q.shape[0]}, sensitivity={self.query_sensitivity}')
        print(f'  Privacy mechanism: {self.privacy_mechanism.report_guarantee()}')
        print(f'{time.time() - t1:.2f} seconds.\n')

        t1 = time.time()
        print(f'Building hierarchical tree structure...', end=' ')
        self.tree = self.data_handler.build_hierarchical_tree()
        print(f'{time.time() - t1:.2f} seconds.\n')

        # Initialize directories to temporarily save vectors and microdata
        self.data_handler.initialize_directories()

        print(self.tree, "\n")

    def estimation_phase(self) -> None:
        '''Run the estimation phase of the TopDown algorithm.

        Processes all nodes in the hierarchical tree, solving joint optimization
        problems that enforce consistency between parent and child contingency vectors.
        After processing, merges all partial microdata files into the final output.
        '''
        print(f'Running estimation phase...')
        t1 = time.time()
        if self._estimation_phase_root():
            self._estimation_phase_subtree()
        print(f'{time.time() - t1:.2f} seconds.\n')

        print(f'Merging microdata files...', end=' ')
        t_merge = time.time()
        self.data_handler.merge_microdata_files()
        self.data_handler.cleanup_directories()
        print(f'{time.time() - t_merge:.2f} seconds.\n')

    def _estimation_phase_root(self) -> bool:
        '''Process the root and its direct children entirely in memory.

        Materializes contingency vectors for the root and its children, adds noise
        in parallel, solves the root individually, then solves the joint optimization
        problem over all children to enforce parent-child consistency.

        Returns:
            bool: True if root's children are not leaves (no microdata written), False otherwise.
        '''
        root = self.tree.root
        root.contingency_vector, root.constraints = self.data_handler.materialize_node_data(root.filter_dict, self.constraints[root.level], self.Q)
        self.privacy_mechanism.add_noise(root.contingency_vector, root.level, self.query_sensitivity)

        # Solve first problem, only root
        self._estimate_node_individually(root)

        # Solve with children
        return not self._estimate_and_update_children_in_memory(root)

    def _estimation_phase_subtree(self) -> None:
        '''Perform the estimation phase of the TopDown algorithm.

        Uses breadth-first traversal with lazy materialization to minimize memory usage.
        '''

        root = self.tree.root
        # Process remaining nodes
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=get_context("spawn"),
                                initializer=init_process, initargs=(self.solver_options, self.constraints,
                                                                    self.data_handler.spill_dir, self.data_handler.microdata_dir,
                                                                    self.data_handler.file_path,
                                                                    self.data_handler.contingency_domain.domains,
                                                                    self.hierarchical_columns, self.query_columns,
                                                                    self.privacy_mechanism, self.Q, self.query_sensitivity,
                                                                    self.check_correctness)) as executor:

            def _submit(node):
                node_path = self.data_handler.spill_path(node.filter_dict)
                children_filter_dicts = [child.filter_dict for child in node.children]
                children_level = node.children[0].level
                is_leaf = node.children[0].is_leaf()

                return executor.submit(estimate_and_update_children, node.id, node_path,
                                     children_filter_dicts, children_level, is_leaf)

            def _fill_window():
                while pending and len(futures) < (self.workers)*2:
                    _, _, node = heapq.heappop(pending)
                    futures[_submit(node)] = node

            # Create a priority queue ordered by number of children: nodes with more
            # children are prioritized so the executor stays busy with nodes that will take more time.
            # id(node) is a tiebreaker to avoid comparing HierarchicalNode objects.
            pending = []
            for node in root.children:
                heapq.heappush(pending, (-len(node.children), id(node), node))

            # future -> node
            # Fill the executor queue
            futures = {}
            _fill_window()

            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)

                for fut in done:
                    fut.result()
                    node = futures.pop(fut)

                    if not node.children[0].is_leaf():
                        for child in node.children:
                            heapq.heappush(pending, (-len(child.children), id(child), child))

                # Update executor queue
                _fill_window()

    def _estimate_node_individually(self, node: HierarchicalNode) -> None:
        '''Solve optimization for a node's own contingency vector.

        Args:
            node (HierarchicalNode): The node to process.
        '''

        node.contingency_vector = _real_and_round_estimation(
            self.optimizer, node.contingency_vector, node.id, node.constraints, self.Q
        )

    def _estimate_and_update_children_in_memory(self, node: HierarchicalNode) -> float:
        '''Solve the joint optimization for a node's children and update their vectors.

        Collects children contingency vectors and constraints, concatenates them into
        a joint problem that also enforces consistency with the parent, solves it, then
        either updates the spilled child vectors (non-leaf) or writes microdata (leaf).

        Args:
            node (HierarchicalNode): The parent node whose children are processed.

        Returns:
            float: Time spent writing microdata files (0.0 if children are not leaves).
        '''
        is_leaf = node.children[0].is_leaf()
        children_vectors = []
        children_constraints = []
        children_filter_dicts = []

        for child in node.children:
            child.contingency_vector, child.constraints = self.data_handler.materialize_node_data(child.filter_dict, self.constraints[child.level], self.Q)
            
            children_vectors.append(child.contingency_vector)
            children_constraints.append(child.constraints)
            children_filter_dicts.append(child.filter_dict)

        joint_contingency_vector = np.concatenate(children_vectors)
        children_vectors = None

        # Apply noise to joint vector in chunks using thread pool
        chunks = []
        for chunk_start in range(0, len(joint_contingency_vector), self.data_handler.noise_chunk_size):
            chunk_end = min(chunk_start + self.data_handler.noise_chunk_size, len(joint_contingency_vector))
            chunks.append((joint_contingency_vector[chunk_start:chunk_end], node.children[0].level, self.query_sensitivity))

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            list(executor.map(lambda args: self.privacy_mechanism.add_noise(*args), chunks))

        joint_constraints = _combine_child_constraints(len(node.children), node.contingency_vector, children_constraints)
        joint_solution = _real_and_round_estimation(self.optimizer, joint_contingency_vector, node.id, joint_constraints, self.Q)

        if self.check_correctness: _check_node_correctness(node.contingency_vector, joint_solution)
        joint_contingency_vector = None
        joint_constraints = None

        microdata_time = 0
        if not is_leaf:
            self.data_handler.update_child_vectors(joint_solution, self.data_handler.contingency_df_length, children_filter_dicts)
        else:
            t_microdata = time.time()
            child_vectors = _split_child_vectors(joint_solution, len(node.children), self.data_handler.contingency_df_length)
            self.data_handler.write_microdata(node.id, child_vectors, children_filter_dicts)
            microdata_time = time.time() - t_microdata

        return microdata_time

    def set_constraint_to_tree(self, constraint: Constraint) -> None:
        '''Add a constraint to all nodes in the hierarchical tree.

        The constraint will be applied when the tree is built.

        Args:
            constraint (Constraint): The Constraint to add.
        '''

        self.set_constraint_to_level(len(self.hierarchical_columns) - 1, constraint)

    def set_constraint_to_level(self, level: int, constraint: Constraint) -> None:
        '''Add a constraint to a specific level in the hierarchical tree.

        The constraint will be applied when the tree is built.

        Args:
            level (int): The index in hierarchical_columns (0-based). Constraint applies to all levels from root up to and including this level.
            constraint (Constraint): The Constraint to add.
        '''
        for level_iter in range(level + 2):
            self.constraints[level_iter].append(constraint)

    def set_query_workload(self, query_matrix: Union[QueryWorkload, np.ndarray]) -> None:
        '''Set the workload query matrix Q.

        Q is applied during tree construction: each node stores Q @ x instead of x.
        If not called, initialize() constructs np.eye(n_cells) as the default.

        Args:
            query_matrix: Either a QueryWorkload (DSL object, built lazily at initialize() time)
                          or a pre-built numpy ndarray of shape (n_queries, n_cells).
        '''
        self.Q = query_matrix

    def run(self) -> None:
        '''Run the TopDown algorithm end-to-end.

        This method executes the full TopDown algorithm, including initialization,
        estimation phase, and microdata construction.
        '''
        try:
            self.initialize()
            self.estimation_phase()
        finally:
            self.data_handler.cleanup_directories()
