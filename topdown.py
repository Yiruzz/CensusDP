import heapq
import numpy as np
import scipy.sparse as sp
import dask

from dask.distributed import Client, LocalCluster, as_completed
dask.config.set({"distributed.worker.multiprocessing-method": "spawn"})

from parallel_utils.utils import WorkerInitializer
from parallel_utils.estimation_phase import estimate_and_update_children

from hierarchical_tree import HierarchicalTree
from hierarchical_node import HierarchicalNode
from data_handler import DataHandler
from constraints.constraint import Constraint
from optimizers.pyoptinterface import OptimizationModel
from optimizers.write_lp_directly import OptimizationModelLP
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
                 solver_options: dict = {}, domain: Optional[Dict[str, List]] = None,
                 check_correctness: bool = False, optimizer_backend: str = 'pyoptinterface') -> None:
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
        self.local_cluster = LocalCluster(
                                n_workers=num_workers,           
                                threads_per_worker=1,
                                processes=True)
        
        self.client = None

        self.check_correctness = check_correctness
        self.optimizer_backend = optimizer_backend

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

        self.data_handler.create_data_view(initialize=True)

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
        
        noise = self.data_handler.noisy_vectors_exist(self.tree._node_count, self.privacy_mechanism.param_spec)

        plugin = WorkerInitializer(
            constraints = self.constraints, optimizer_conf = self.optimizer, optimizer_backend = self.optimizer_backend, data_type = self.data_handler.dtype,
            input_file = self.data_handler.file_path, spill_dir = self.data_handler.spill_dir, microdata_dir = self.data_handler.microdata_dir,
            hierarchical_columns = self.hierarchical_columns, query_columns = self.query_columns,
            domains = self.data_handler.contingency_domain.domains,
            privacy_mech_name = self.privacy_mechanism.name, level_params = self.privacy_mechanism.level_params, Q = self.Q, sensitivity = self.query_sensitivity,
            zarr_path = self.data_handler.noise_zarr_path, noisy_array_name = self.data_handler.noisy_array_name,
            check_correctness = self.check_correctness
        )
        self.client = Client(self.local_cluster)
        self.client.register_plugin(plugin)

        # Pre-generate noise vectors for all nodes
        t1 = time.time()
        print(f'Pre-generating noise vectors if needed...', end=' ')
        if not noise:
            print("")
            self.data_handler.generate_noise_vectors(self.client, self.tree._node_count,
                                                     self.tree.iter_nodes_with_levels())
        print(f'{time.time() - t1:.2f} seconds.\n')

    def estimation_phase(self) -> None:
        '''Run the estimation phase of the TopDown algorithm.

        Processes the root and its children in memory, then solves the rest of the tree
        with a process pool, and finally merges the partial microdata files into the output.
        '''
        print(f'Running estimation phase...')
        t1 = time.time()
        self._estimation_phase_subtree()
        self.client.close()
        self.client.cluster.close()
        print(f'{time.time() - t1:.2f} seconds.\n')

        print(f'Merging microdata files...', end=' ')
        t_merge = time.time()
        self.data_handler.merge_microdata_files()
        print(f'{time.time() - t_merge:.2f} seconds.\n')

    def _estimation_phase_subtree(self) -> None:
        '''Solve the tree with a process pool.

        Uses breadth-first traversal with lazy materialization to minimize memory usage.
        Nodes are scheduled through a priority queue ordered by number of children, so
        nodes with more work are dispatched first and the executor stays busy.
        '''
        root = self.tree.root
        root.contingency_vector, root.constraints = self.data_handler.materialize_node_data(root.filter_dict, self.constraints[root.level], self.Q)
        try:
            self.privacy_mechanism.add_noise_from_precomputed(self.data_handler.noise_zarr_group[self.data_handler.noisy_array_name], root.contingency_vector, root.id)
        except:
            self.privacy_mechanism.add_noise(root.contingency_vector, root.level, self.query_sensitivity)

        # First phase: resolve root's own contingency vector
        self._estimate_node_individually(root)
        path = self.data_handler.spill_path(root.filter_dict)
        self.data_handler.spill_vector(path, root.contingency_vector)

        fut_to_node = {}

        def _submit(node):
            node_path = self.data_handler.spill_path(node.filter_dict)
            children_filter_dicts = [child.filter_dict for child in node.children]
            children_ids = [child.id for child in node.children]
            children_level = node.children[0].level
            is_leaf = node.children[0].is_leaf()

            fut = self.client.submit(
                estimate_and_update_children,
                node.id, node_path,
                children_filter_dicts, children_ids,
                children_level, is_leaf,
                priority=len(node.children)
            )
            fut_to_node[fut] = node
            return fut
        
        ac = as_completed([_submit(root)])

        for fut in ac:
            fut.result()
            node = fut_to_node.pop(fut)

            for child in node.children:
                if child.children:
                    ac.add(_submit(child))

    def _estimate_node_individually(self, node: HierarchicalNode) -> None:
        '''Solve optimization for a node's own contingency vector.

        Args:
            node (HierarchicalNode): The node to process.
        '''
        optimizer = OptimizationModel(*self.optimizer) if self.optimizer_backend == 'pyoptinterface' else OptimizationModelLP(*self.optimizer)

        t1 = time.time()
        x_tilde = optimizer.non_negative_real_estimation(
            noisy_measurements=[node.contingency_vector],
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
