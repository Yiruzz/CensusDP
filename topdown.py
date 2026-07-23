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
from parallel_utils.estimation_phase import (init_process, estimate_and_update_children,
                                             _combine_child_constraints, _check_node_correctness)
from queries import QueryWorkload
from privacy import PrivacyMechanism

from typing import Dict, List, Optional, Union, Tuple
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
                 solver_options: dict = {}, domain: Optional[Dict[str, List]] = None, check_correctness: bool = False) -> None:
        """Initialize the TopDown algorithm.

        Args:
            data_path (str): Path to the input data file.
            hierarchy (List[str]): List of columns representing the hierarchy levels.
            query_columns (List[str]): List of columns to be queried and aggregated.
            privacy_mechanism (PrivacyMechanism): DP variant carrying per-level parameters.
                Length of mechanism.level_params must equal len(hierarchy) + 1 (root + per-level).
            num_workers (int): Number of parallel workers for the estimation phase.
            out_path (str): Path to save the noisy output data. Defaults to 'noisy_data.csv'.
            solver_options (dict): Dictionary of Gurobi parameters passed to the optimizer environment.
                Defaults to empty dict.
            domain (Optional[Dict[str, List]]): Per-column set of all possible values for the
                query columns, defining the contingency cell space. Should be data-independent
                for a sound DP guarantee. When None (or a column omitted), the domain is inferred
                from the observed data with a warning. Passed through to DataHandler.
            check_correctness (bool): Whether to run correctness checks during execution. Defaults to False.

        Attributes:
            data_handler (DataHandler): Manages data loading, preprocessing, and output.
            hierarchical_columns (List[str]): Columns representing the hierarchy levels.
            query_columns (List[str]): Columns to be queried and aggregated.
            privacy_mechanism (PrivacyMechanism): DP variant and its per-level parameters.
            tree (HierarchicalTree): Hierarchical structure of the data.
            optimizer (Tuple[type, str, Dict]): Params to pass to the solver (result dtype, temporary files directory
                                                and solver options dict).
            constraints (Dict[int, List[Constraint]]): Constraints registered per tree level.
            workers (int): Number of parallel workers for the estimation phase.
        """
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

        self.optimizer: Tuple[type, Optional[str], Dict] = (self.data_handler.dtype,
                                                            self.data_handler.lp_problems_dir,
                                                            solver_options)

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
        self.optimizer = (self.optimizer[0],
                          self.data_handler.lp_problems_dir,
                          self.optimizer[2])

        print(self.tree, "\n")

        # Pre-generate noise vectors for all nodes
        t1 = time.time()
        print(f'Pre-generating noise vectors if needed...', end=' ')
        if not self.data_handler.noisy_vectors_exist(self.tree._node_count, self.privacy_mechanism.param_spec,
                                                     self.query_sensitivity):
            print("")
            self.data_handler.generate_noise_vectors(self.workers, self.tree._node_count,
                                                     self.tree.iter_nodes_with_levels(),
                                                     self.privacy_mechanism.name, self.privacy_mechanism.level_params, self.query_sensitivity)
        print(f'{time.time() - t1:.2f} seconds.\n')

    def estimation_phase(self) -> None:
        '''Run the estimation phase of the TopDown algorithm.

        Processes the root and its children in memory, then solves the rest of the tree
        with a process pool, and finally merges the partial microdata files into the output.
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

        Materializes and noises the root, solves it individually, then solves the joint
        problem over its children (in the main process). The root vector stays in memory
        as the children's parent — it is never spilled.

        Returns:
            bool: True if the root's children are not leaves (the subtree still needs
                processing), False if they are leaves (microdata already written).
        '''
        root = self.tree.root
        root.contingency_vector, root.constraints = self.data_handler.materialize_node_data(root.filter_dict, self.constraints[root.level], self.Q)
        try:
            self.privacy_mechanism.add_noise_from_precomputed(self.data_handler.noise_zarr_group[self.data_handler.noisy_array_name], root.contingency_vector, root.id)
        except:
            self.privacy_mechanism.add_noise(root.contingency_vector, root.level, self.query_sensitivity)

        # First phase: resolve root's own contingency vector
        self._estimate_node_individually(root)

        # Second phase: solve jointly with the children
        is_leaf = self._estimate_and_update_children_in_memory(root)
        root.contingency_vector = None
        return not is_leaf

    def _estimation_phase_subtree(self) -> None:
        '''Solve the rest of the tree (below the root's children) with a process pool.

        Uses breadth-first traversal with lazy materialization to minimize memory usage.
        Nodes are scheduled through a priority queue ordered by number of children, so
        nodes with more work are dispatched first and the executor stays busy.
        '''
        root = self.tree.root
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=get_context("spawn"),
                                initializer=init_process, initargs=(self.optimizer, self.constraints,
                                                                    self.data_handler.spill_dir,
                                                                    self.data_handler.microdata_dir,
                                                                    self.data_handler.file_path,
                                                                    self.data_handler.contingency_domain.domains,
                                                                    self.hierarchical_columns, self.query_columns,
                                                                    self.privacy_mechanism,
                                                                    self.Q, self.query_sensitivity,
                                                                    self.check_correctness,
                                                                    self.data_handler.noise_zarr_path, self.data_handler.noisy_array_name)) as executor:

            def _submit(node):
                node_path = self.data_handler.spill_path(node.filter_dict)
                children_filter_dicts = [child.filter_dict for child in node.children]
                children_ids = [child.id for child in node.children]
                children_level = node.children[0].level
                is_leaf = node.children[0].is_leaf()

                return executor.submit(estimate_and_update_children, node.id, node_path,
                                     children_filter_dicts, children_ids, children_level, is_leaf)

            # Priority queue ordered by number of children: nodes with more children are
            # dispatched first so the executor stays busy with the heavier work.
            # id(node) breaks ties to avoid comparing HierarchicalNode objects.
            pending = []
            for node in root.children:
                heapq.heappush(pending, (-len(node.children), id(node), node))

            futures = {}

            def _fill_window():
                while pending and len(futures) < self.workers * 2:
                    _, _, node = heapq.heappop(pending)
                    futures[_submit(node)] = node

            _fill_window()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)

                for fut in done:
                    fut.result()
                    node = futures.pop(fut)

                    if not node.children[0].is_leaf():
                        for child in node.children:
                            heapq.heappush(pending, (-len(child.children), id(child), child))

                _fill_window()

    def _mat_and_noise(self, child: HierarchicalNode) -> Tuple[np.ndarray, List, Dict]:
        '''Materialize and noise a single child (run per-thread).

        Uses a dedicated DuckDB cursor so concurrent threads don't share a connection.
        Keeps main's per-child noise logic (pre-computed with fallback).

        Args:
            child (HierarchicalNode): The child node to materialize.

        Returns:
            Tuple: (noisy contingency vector, constraint list, child filter dict).
        '''
        con = self.data_handler.duckdb_con.cursor()
        try:
            child_vector, child_constraint = self.data_handler.materialize_node_data(
                child.filter_dict, self.constraints[child.level], self.Q, con=con)
        finally:
            con.close()

        try:
            self.privacy_mechanism.add_noise_from_precomputed(
                self.data_handler.noise_zarr_group[self.data_handler.noisy_array_name], child_vector, child.id)
        except:
            self.privacy_mechanism.add_noise(child_vector, child.level, self.query_sensitivity)

        return child_vector, child_constraint, child.filter_dict

    def _estimate_and_update_children_in_memory(self, node: HierarchicalNode) -> bool:
        '''Solve the joint optimization for a node's children in the main process.

        Mirrors the worker `estimate_and_update_children` but in memory: the children are
        materialized and noised with a thread pool (one cursor per thread), concatenated,
        and solved jointly under parent-child consistency. Children data is kept in local
        variables (never stored on the node objects) to avoid retaining vectors in RAM.

        Note: threads are used for now; this is expected to move to processes later.

        Args:
            node (HierarchicalNode): The parent node whose children are processed.

        Returns:
            bool: True if the children are leaves (microdata written), False otherwise.
        '''
        children = node.children
        is_leaf = children[0].is_leaf()
        n_cells = self.data_handler.n_cells
        num_children = len(children)

        # Materialize + noise each child in parallel; map preserves order so the joint
        # vector, active set and filter dicts stay aligned.
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            results = list(executor.map(self._mat_and_noise, children))

        children_vectors = [r[0] for r in results]
        children_constraints = [r[1] for r in results]
        children_filter_dicts = [r[2] for r in results]

        joint_contingency_vector = np.concatenate(children_vectors)
        children_vectors = None

        # Cells where the parent is non-zero. By non-negativity + consistency, children can
        # only be non-zero there; expand to joint-space indices {k*n_cells + j}.
        support = node.contingency_vector.indices
        active = [k * n_cells + int(j) for k in range(num_children) for j in support]
        joint_constraints = _combine_child_constraints(num_children, node.contingency_vector, children_constraints, set(active), n_cells=n_cells)

        optimizer = OptimizationModel(*self.optimizer)

        t1 = time.time()
        x_tilde = optimizer.non_negative_real_estimation(
            noisy_measurements=joint_contingency_vector,
            node_id=node.id,
            constraints=joint_constraints,
            query_matrix=self.Q,
            active=active
        )
        real_time = time.time() - t1

        t1 = time.time()
        joint_solution = optimizer.rounding_estimation(
            x_tilde=x_tilde,
            node_id=node.id,
            constraints=joint_constraints,
            active=active,
            n=num_children * n_cells
        )
        rounding_time = time.time() - t1

        if self.check_correctness: _check_node_correctness(node.contingency_vector, joint_solution)

        if not is_leaf:
            self.data_handler.update_child_vectors(joint_solution, children_filter_dicts)
        else:
            child_vectors = []
            start = 0
            for _ in children_filter_dicts:
                end = start + n_cells
                child_vectors.append(joint_solution[start:end])
                start = end
            self.data_handler.write_microdata(node.id, child_vectors, children_filter_dicts)

        print(f'  [Node {node.id}] - real {real_time:.1f}s - rounding {rounding_time:.1f}s')
        return is_leaf

    def _estimate_node_individually(self, node: HierarchicalNode) -> None:
        '''Solve optimization for a node's own contingency vector.

        Args:
            node (HierarchicalNode): The node to process.
        '''
        optimizer = OptimizationModel(*self.optimizer)

        t1 = time.time()
        x_tilde = optimizer.non_negative_real_estimation(
            noisy_measurements=node.contingency_vector,
            node_id=node.id,
            constraints=node.constraints,
            query_matrix=self.Q
        )
        real_time = time.time() - t1

        t1 = time.time()
        node.contingency_vector = optimizer.rounding_estimation(
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
