import numpy as np
import scipy.sparse as sp

from collections import deque
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

from constraints.sparse_constraint import SparseConstraint


class TopDown():
    '''Represents the TopDown algorithm for generating differentially private microdata.

    The algorithm works by constructing a hierarchical tree structure, then adding noise to the data
    considering differential privacy principles. It propagates the noise to each node in the tree and
    finally it solves optimization problems to ensure consistency across the tree and adherence to
    specified constraints by the user.
    '''
    def __init__(self, data_path: str, hierarchy: List[str], query_columns: List[str],
                 privacy_mechanism: PrivacyMechanism, out_path: str = 'noisy_data.csv',
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

    def estimation_phase(self) -> None:
        '''Run the estimation phase of the TopDown algorithm.

        Processes the root and its children in memory, then solves the rest of the tree
        with a process pool, and finally merges the partial microdata files into the output.
        '''
        print(f'Running estimation phase...')
        t1 = time.time()
        self._estimation_phase_subtree()
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
        self.privacy_mechanism.add_noise(root.contingency_vector, root.level, self.query_sensitivity)

        # First phase: resolve root's own contingency vector
        self._estimate_node_individually(root)
        path = self.data_handler.spill_path(root.filter_dict)
        self.data_handler.spill_vector(path, root.contingency_vector)

        def _submit(node):
            node_path = self.data_handler.spill_path(node.filter_dict)
            children_filter_dicts = [child.filter_dict for child in node.children]
            children_ids = [child.id for child in node.children]
            children_level = node.children[0].level
            is_leaf = node.children[0].is_leaf()
            
            self.estimate_and_update_children(node.id, node_path, children_filter_dicts, children_ids, children_level, is_leaf)

        queue = deque([root])

        while queue:
            node = queue.popleft()
            _submit(node)

            if not node.children[0].is_leaf():
                for child in node.children:
                    queue.append(child)

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

    def _combine_child_constraints(self, num_children: int, contingency_vector: sp.csc_matrix, constraints: List, active_set: set, n_cells: Optional[int] = None) -> List[SparseConstraint]:
        '''Combine child publication constraints into joint SparseConstraints.

        Creates consistency constraints that ensure each parent cell equals the sum of corresponding child cells.

        The parent vector is a sparse CSC column vector, so consistency constraints are emitted only
        for its non-zero cells (its support). Cells where the parent is 0 need no constraint: the
        optimizer does not create child variables there, so they are structurally 0 and the
        consistency sum(children) == 0 holds automatically.

        This function translates child-local SparseConstraints (indexed 0..n_cells-1) to joint-space
        SparseConstraints (indexed over the global indices), filtering to only active cells
        (pruned cells are dropped from the coefficients).

        Args:
            num_children (int): Number of child nodes.
            contingency_vector (sp.csc_matrix): The parent's sparse cell-count vector, shape (n_cells, 1).
            constraints (List): List of Constraint objects (one list per child). Each constraint's
                to_sparse_constraint() method will be called to get SparseConstraint representations.
            active_set (set): Active joint-space global indices {k*n_cells + j} (parent support expanded
                over children). Cells outside it are pruned and dropped from constraints.
            n_cells (Optional[int]): Number of contingency cells. Defaults to the worker-global
                _data_handler.n_cells; callers in the main process (no worker globals) must pass it.

        Returns:
            List[SparseConstraint]: List of SparseConstraints for the joint optimization problem.
        '''
        joint_constraints = []
        
        # Per-child constraints: convert each constraint to SparseConstraint via to_sparse_constraint()
        # Offset them to joint space (base + j) and filter to only active cells.
        start = 0
        for child_constraints in constraints:
            base = start

            for sparse_constraint in child_constraints:
                new_sparse_constraint = sparse_constraint.prune_to_active_space(base, active_set)
                if new_sparse_constraint is not None: joint_constraints.append(new_sparse_constraint)
            start += n_cells

        # Consistency constraints: parent value at each non-zero cell = sum of child values at that cell.
        # Authored directly in joint space with all indices active (parent support ensures this).
        for index, value in zip(contingency_vector.indices, contingency_vector.data):
            index = int(index)
            indices_to_sum = np.array([index + i * n_cells for i in range(num_children)])

            joint_constraints.append(
                SparseConstraint(
                    indices=indices_to_sum,
                    sense="=",
                    rhs=float(value)
                )
            )

        return joint_constraints

    def _check_node_correctness(self, parent_vector: sp.csc_matrix, children_vectors: sp.csc_matrix) -> None:
        '''Checks that the sum of the values in the parent node vector
        is equal to the sum of the values in its children vectors.

        Args:
            parent_vector (sp.csc_matrix): Contingency vector of the parent node.
            children_vectors (sp.csc_matrix): Concatenated contingency vectors of the child nodes.
        '''
        parent_sum = parent_vector.data.sum()
        children_sum = children_vectors.data.sum()

        if parent_sum != children_sum:
            print(f"\nError: The sum of the children nodes' contingency vectors "
                f"({children_sum}) does not equal the parent node's contingency vector ({parent_sum}).")

    def estimate_and_update_children(self, node_id: int, node_path: str, children_filter_dicts: List[Dict],
                                    children_ids: List[int], children_level: int, is_leaf: bool = False) -> None:
        '''Solve optimization for a node considering its children and update their vectors.

        Args:
            node_id (int): The unique ID of the parent node.
            node_path (str): Path to contigency vector file.
            children_filter_dicts (List[Dict[str, Any]]): List of filter dictionaries for each child.
            children_ids (List[int]): List of node IDs for each child (for pre-computed noise lookup).
            children_level (int): Level of all children (they all share the same level).
            is_leaf (bool): Whether children are leaf nodes. Defaults to False.

        Returns:
            float: Time spent writing microdata files
        '''
        contingency_vector = self.data_handler.load_vector(node_path)

        # Materialize and combine children vectors and constraints
        children_vectors = []
        children_constraints = []

        for filter_dict, child_id in zip(children_filter_dicts, children_ids):
            child_vector, child_constraint = self.data_handler.materialize_node_data(filter_dict, self.constraints[children_level], self.Q)
            
            self.privacy_mechanism.add_noise(child_vector, children_level, self.query_sensitivity)
            children_vectors.append(child_vector)
            children_constraints.append(child_constraint)

        # Constraints are adapted to the new vector size.
        # Also create others to ensure consistency in the number of rows per category in the parent.
        # The number of rows in the parent category must match the sum of rows of that category across all children.
        n_cells = self.data_handler.n_cells
        num_children = len(children_filter_dicts)
        n_joint = num_children * n_cells

        # Cells where the parent is non-zero. By non-negativity + consistency, children can only
        # be non-zero on these cells. Expand the support to joint-space indices {k*n_cells + j}
        # so the optimizers instantiate variables only there. `active` stays an ordered list: the
        # optimizer aligns its solution positionally to it across the real -> rounding solves.
        #support = contingency_vector.indices
        #active = [k * n_cells + int(j) for k in range(num_children) for j in support]
        active = None

        # Combine receives the active set so it can bake prune-to-0 + reindexing into the constraints.
        joint_constraints = self._combine_child_constraints(num_children, contingency_vector,
                                                    children_constraints, active, self.data_handler.n_cells)

        optimizer = OptimizationModel(*self.optimizer) if self.optimizer_backend == 'pyoptinterface' else OptimizationModelLP(*self.optimizer)

        t1 = time.time()
        x_tilde = optimizer.non_negative_real_estimation(
            noisy_measurements=children_vectors,
            node_id=node_id,
            constraints=joint_constraints,
            query_matrix=self.Q,
            active=active
        )
        real_time = time.time() - t1

        t1 = time.time()
        joint_solution = optimizer.rounding_estimation(
            x_tilde=x_tilde,
            node_id=node_id,
            constraints=joint_constraints,
            active=active,
            n=n_joint
        )
        rounding_time = time.time() - t1

        if self.check_correctness: self._check_node_correctness(contingency_vector, joint_solution)

        if not is_leaf: self.data_handler.update_child_vectors(joint_solution, children_filter_dicts)
        else:
            child_vectors = []
            start = 0

            for _ in children_filter_dicts:
                end = start + n_cells
                updated_vector = joint_solution[start:end]
                child_vectors.append(updated_vector)
                start = end

            self.data_handler.write_microdata(node_id, child_vectors, children_filter_dicts)

        print(f' [Node {node_id}] - real {real_time:.1f}s - rounding {rounding_time:.1f}s')


