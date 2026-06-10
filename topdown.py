import numpy as np
import scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context

from hierarchical_tree import HierarchicalTree
from hierarchical_node import HierarchicalNode
from data_handler import DataHandler
from optimizer import OptimizationModel
from constraints.constraint import Constraint
from parallel_utils import init_process, estimate_and_update_children
from queries import QueryWorkload
from privacy import PrivacyMechanism

from collections import deque
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
        # Also output to put microdata
        self.data_handler.initialize_directories()
        self.data_handler.initialize_output_file()

        print(self.tree, "\n")

    def estimation_phase(self) -> None:
        '''Perform the estimation phase of the TopDown algorithm.

        Uses breadth-first traversal with lazy materialization to minimize memory usage.
        '''
        print(f'Running estimation phase (BFS)...')
        t1 = time.time()

        # Materialize root
        root = self.tree.root
        root.contingency_vector, root.constraints = self.data_handler.materialize_node_data(root.filter_dict, self.constraints[root.level], self.Q)
        self.privacy_mechanism.add_noise(root.contingency_vector, root.level, self.query_sensitivity)

        # First phase: resolve root's own contingency vector
        self._estimate_node_individually(root)

        # Save the contingency vector to disk.
        root_path = self.data_handler.spill_path(root.filter_dict)
        self.data_handler.spill_vector(root_path, root.contingency_vector)
        root.contingency_vector = None

        # Extract privacy mechanism parameters for passing to workers
        privacy_name = self.privacy_mechanism.name
        level_params = self.privacy_mechanism.level_params
        delta = getattr(self.privacy_mechanism, 'delta', None)
        alphas = getattr(self.privacy_mechanism, 'alphas', None)

        # Process remaining nodes
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=get_context("spawn"),
                                initializer=init_process, initargs=(self.solver_options,
                                                                    self.data_handler.spill_dir,
                                                                    self.data_handler.microdata_dir,
                                                                    self.Q,
                                                                    self.data_handler.file_path,
                                                                    self.hierarchical_columns,
                                                                    self.query_columns,
                                                                    self.data_handler.contingency_domain.domains,
                                                                    self.constraints,
                                                                    privacy_name,
                                                                    level_params,
                                                                    delta,
                                                                    alphas,
                                                                    self.query_sensitivity,
                                                                    self.check_correctness)) as executor:

            def _submit(node):
                node_path = self.data_handler.spill_path(node.filter_dict)
                children_filter_dicts = [child.filter_dict for child in node.children]
                children_level = node.children[0].level
                is_leaf = node.children[0].is_leaf()

                return executor.submit(estimate_and_update_children, node.id, node_path,
                                     children_filter_dicts, children_level, is_leaf)
            
            total_microdata_time = 0.0
            futures = {_submit(root): root}
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)

                for fut in done:
                    worker_microdata_time = fut.result()
                    total_microdata_time += worker_microdata_time
                    node = futures.pop(fut)

                    if not node.children[0].is_leaf():
                        for child in node.children:
                            futures[_submit(child)] = child

            # Merge microdata files from workers
            t_merge = time.time()
            print(f'Merging microdata files...', end=' ')
            self.data_handler.merge_microdata_files()
            merge_time = time.time() - t_merge
            print(f'{merge_time:.2f}s')
            print(f'  Worker microdata time: {total_microdata_time:.2f}s')
            print(f'  Total microdata (workers + merge): {total_microdata_time + merge_time:.2f}s')

        self.data_handler.cleanup_directories()
        print(f'{time.time() - t1:.2f} seconds.\n')

    def _estimate_node_individually(self, node: HierarchicalNode) -> None:
        '''Solve optimization for a node's own contingency vector.

        Args:
            node (HierarchicalNode): The node to process.
        '''

        t1 = time.time()
        x_tilde = self.optimizer.non_negative_real_estimation(
            noisy_measurements=node.contingency_vector,
            node_id=node.id,
            constraints=node.constraints,
            query_matrix=self.Q
        )
        real_time = time.time() - t1

        t1 = time.time()
        node.contingency_vector = self.optimizer.rounding_estimation(
            x_tilde=x_tilde,
            node_id=node.id,
            constraints=node.constraints
        )
        rounding_time = time.time() - t1

        print(f'  [Node {node.id}] - real {real_time:.1f}s - rounding {rounding_time:.1f}s')

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
