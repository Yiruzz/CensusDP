import heapq
import numpy as np
import scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context

from hierarchical_tree import HierarchicalTree
from hierarchical_node import HierarchicalNode
from data_handler import DataHandler
from constraints.constraint import Constraint
from constraints.domain_restriction import tree_wide
from parallel_utils.estimation_phase import init_process, estimate_and_update_children
from parallel_utils import marginal_estimation
from optimizers import build_optimizer
from graph import JunctionTree, MarginalSelectionStrategy, MaxSpanningTreeMI, PairwiseAssociation
from queries import QueryWorkload, is_identity_workload
from privacy import PrivacyMechanism

from typing import Dict, Iterable, List, Optional, Union, Tuple
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
                 check_correctness: bool = False, optimizer_backend: str = 'write_lp') -> None:
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
            structural (List[Constraint]): Those registered at every level, which are the only
                ones allowed to shape the cell space. Resolved in initialize().
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

        # Factored pipeline. Set via set_marginals() or set_marginal_selection(); when both
        # are unset the algorithm runs the full-joint pipeline over Q instead. The two modes
        # are mutually exclusive and both fully supported.
        self.marginal_cliques: Optional[List[Iterable[str]]] = None
        self.junction_tree: Optional[JunctionTree] = None
        self.marginal_strategy: Optional[MarginalSelectionStrategy] = None
        self.selection_budget_fraction: float = 0.0
        self._budget_split_applied: bool = False

        self.constraints: Dict[int, List[Constraint]] = {i: [] for i in range(len(hierarchy) + 1)}
        # The subset of the above that shapes the cell space
        self.structural: List[Constraint] = []

        self.tree: HierarchicalTree = HierarchicalTree()

        self.optimizer: Tuple[type, Optional[str], Dict] = (self.data_handler.dtype,
                                                            self.data_handler.lp_problems_dir,
                                                            solver_options)

        self.workers = num_workers
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

        # Structural zeros are folded into the cell space instead of being enforced as
        # optimizer rows. Only constraints that are tree-wide qualify. Resolved once here and
        # shipped to the workers verbatim, so every process shapes the same cell space.
        self.structural = tree_wide(self.constraints)
        self.data_handler.create_data_view()

        factored = self.marginal_cliques is not None or self.marginal_strategy is not None

        t1 = time.time()
        print(f'Building contingency domain...', end=' ')
        # The factored pipeline never materialises the joint, so only its per-bag domains are
        # restricted (in _build_junction_tree), restricting the joint here would just force a
        # size it is meant never to compute.
        self.data_handler.build_contingency_domain(None if factored else self.structural)
        print(f'{time.time() - t1:.2f} seconds.')

        t1 = time.time()
        if factored:
            print(f'Building junction tree...', end=' ')
            self._build_junction_tree(self.structural)
            print(f'{time.time() - t1:.2f} seconds.\n')
        else:
            self._build_query_workload(t1)

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

    def _reserve_selection_budget(self) -> float:
        '''Carve the marginal-selection share out of the total privacy budget.

        Selection reads the data, so it must be paid for. Rather than build a second
        mechanism, the selection is treated as one more level of the same sequential
        composition: the per-level parameters are scaled down by (1 - fraction) and the
        reserved share is appended as an extra entry. Under zCDP the parameters simply add
        up, so the total is unchanged - the tree just gets a smaller share of it.

            before:  [r0, r1, ..., rL]                       sum = T
            after:   [(1-f)r0, ..., (1-f)rL, f*T]            sum = T

        This works for every mechanism (PureDP, ZCDP, ApproximateDP, RenyiDP) without
        special-casing their extra parameters, and RenyiDP's joint-alpha calibration
        naturally accounts for the extra level. It also changes param_spec, so noise files
        from a run with a different split are never reused.

        Mutates the mechanism in place and is idempotent: calling initialize() twice does
        not shrink the budget twice.

        Returns:
            float: The privacy parameter reserved for selection.
        '''
        params = self.privacy_mechanism.level_params
        if self._budget_split_applied:
            return params[-1]

        total = sum(params)
        reserved = self.selection_budget_fraction * total
        remaining = 1.0 - self.selection_budget_fraction

        self.privacy_mechanism.level_params = [p * remaining for p in params] + [reserved]
        self._budget_split_applied = True

        print(f'\n  Selection budget: {reserved:.6g} of {total:.6g} '
              f'({self.selection_budget_fraction:.0%}); the tree keeps {total - reserved:.6g}')
        return reserved

    def _select_marginal_cliques(self) -> List[Iterable[str]]:
        '''Pick the marginals to measure, spending the reserved budget on the data.

        All 2-way marginals are measured over the whole dataset in one shot and noised
        together. A record falls in exactly one cell of each pair table, so the squared L2
        sensitivity is the number of pairs. Mutual information is then computed from the
        NOISY tables only - the raw data is never read by the selection - and handed to the
        strategy, which returns the cliques.

        Returns:
            List[Iterable[str]]: The selected cliques.
        '''
        selection_level = len(self.privacy_mechanism.level_params) - 1

        counts, pairs, offsets = self.data_handler.measure_pairwise_counts(self.query_columns)
        self.privacy_mechanism.add_noise(counts, selection_level, len(pairs))

        # Slice the noisy vector back into one 2-D table per pair.
        tables = {}
        for pair, offset in zip(pairs, offsets):
            domain = self.data_handler.contingency_domain.subdomain(pair)
            block = counts[offset:offset + domain.n_cells]
            tables[pair] = block.reshape(int(domain.sizes[0]), int(domain.sizes[1]))

        association = PairwiseAssociation().compute(
            self.query_columns, lambda a, b: tables[(a, b)])

        mandatory = {constraint.scope()
                     for level_constraints in self.constraints.values()
                     for constraint in level_constraints}
        cliques = self.marginal_strategy.select(
            self.query_columns, association, [scope for scope in mandatory if scope])

        print(f'  Selection measured {len(pairs)} pairwise marginals '
              f'({len(counts)} cells) with sensitivity {len(pairs)}')
        return cliques

    def _build_junction_tree(self, structural: List[Constraint]) -> None:
        '''Build the junction tree and bind the per-bag cell spaces (factored pipeline).

        Every constraint scope is embedded as a mandatory clique alongside the requested
        marginals, so each constraint is guaranteed to fit inside some bag.

        Args:
            structural (List[Constraint]): The tree-wide constraints, as tree_wide() selects
                them. Those declaring structural zeros are folded into each bag's cell space
                instead of being enforced as optimizer rows.
        '''
        cliques = list(self.marginal_cliques or [])
        if self.marginal_strategy is not None:
            self._reserve_selection_budget()
            cliques += self._select_marginal_cliques()

        # When selecting marginals with the strategy, there can be duplicates cliques when
        # adding the scopes from the constraints here, but since the junction tree is built
        # from a set of cliques, the duplicates are implicitly removed.
        scopes = {constraint.scope()
                  for level_constraints in self.constraints.values()
                  for constraint in level_constraints}
        cliques += [scope for scope in scopes if scope]

        # Per-column cardinalities drive a cardinality-aware triangulation: without them
        # the triangulation can fuse high-cardinality columns (region, year, occupation)
        # into a bag of tens of millions of cells even when every constraint scope is only
        # two columns wide.
        domain = self.data_handler.contingency_domain
        weights = {column: int(size) for column, size in zip(domain.columns, domain.sizes)}
        self.junction_tree = JunctionTree.build(self.query_columns, cliques, weights)
        self.data_handler.build_marginal_domains(self.junction_tree, structural)

        width = self.data_handler.marginal_width
        # No Q at all: in the factored pipeline the workload is structurally the identity -
        # each bag cell is measured directly by its own GROUP BY, never aggregated from others.
        # The optimizers take the separable objective path when query_matrix is None.
        self.Q = None
        self.query_sensitivity = self.junction_tree.n_bags

        print(f'\n  Bags: {self.junction_tree.bags}')
        print(f'  Marginal width: {width} (full joint would be {self.data_handler.n_cells}), '
              f'sensitivity={self.query_sensitivity}')
        self._report_disconnected_components()
        print(f'  Privacy mechanism: {self.privacy_mechanism.report_guarantee()}')

    def _report_disconnected_components(self) -> None:
        '''Announce when the declared marginals leave the interaction graph disconnected.

        A tree edge with an empty separator means its two bags share no column, i.e. the
        marginals say those attribute groups are independent. That is a legitimate modelling
        choice and it is not patched by inventing a correlation - measuring an extra marginal
        would cost privacy budget (sensitivity is the number of bags) to assert a dependence
        that was never claimed.

        What is imposed is the shared-total row on each such edge, because every record
        contributes 1 to every bag regardless of correlation. Without it the components'
        totals drift apart under noise and the microdata cannot be reconstructed at all - the
        join needs every bag to total the node's record count.
        '''
        unlinked = [(i, j) for i, j in self.junction_tree.edges()
                    if not self.junction_tree.separator(i, j)]
        if not unlinked:
            return

        print(f'  NOTE: the interaction graph is disconnected - {len(unlinked)} junction-tree '
              f'edge(s) join bags that share no column:')
        for i, j in unlinked:
            print(f'    {self.junction_tree.bags[i]}  <->  {self.junction_tree.bags[j]}')
        print('    Those attribute groups are treated as INDEPENDENT: no correlation between '
              'them\n    survives into the microdata. A shared-total constraint is imposed on '
              'each edge so the\n    bags agree on the node population and the microdata can '
              'be generated; it costs no\n    privacy budget. Declare a marginal spanning the '
              'groups if you want them correlated.')

    def _build_query_workload(self, t1: float) -> None:
        '''Resolve the workload matrix Q and its sensitivity (full-joint pipeline).

        Args:
            t1 (float): Start timestamp, for the elapsed-time report.
        '''
        print(f'Building query workload...', end=' ')
        n_cells = self.data_handler.contingency_domain.n_cells

        if isinstance(self.Q, QueryWorkload):
            self.Q = self.Q.build(self.data_handler.contingency_domain)
        elif not (isinstance(self.Q, np.ndarray) or sp.issparse(self.Q)):
            # No workload set: every cell is answered directly, i.e. the identity.
            self.Q = None

        # Check for identity matrix, since when Q is the the identity we can afford to drop it entirely
        # and avoid computations that depends on the form of Q.
        if is_identity_workload(self.Q):
            self.Q = None
            self.query_sensitivity = 1
            # Set explicitly rather than leaning on noise_width's fallback chain
            # (marginal_width -> query_width -> n_cells): that ordering is implicit coupling.
            self.data_handler.query_width = int(n_cells)
            print(f'\n  Query matrix: identity, n_queries={n_cells}, '
                  f'sensitivity={self.query_sensitivity}')
            print(f'  Privacy mechanism: {self.privacy_mechanism.report_guarantee()}')
            print(f'{time.time() - t1:.2f} seconds.\n')
            return

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
        # The node measurement is y = Q @ x, so the pre-computed noise must be this wide -
        # equal to n_cells only for the identity workload.
        self.data_handler.query_width = int(self.Q.shape[0])
        print(f'\n  Query matrix: n_queries={self.Q.shape[0]}, sensitivity={self.query_sensitivity}')
        print(f'  Privacy mechanism: {self.privacy_mechanism.report_guarantee()}')
        print(f'{time.time() - t1:.2f} seconds.\n')

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
        self._solve_root(root)

        initializer, initargs = self._pool_setup()
        worker = (marginal_estimation.estimate_and_update_children
                  if self.junction_tree is not None else estimate_and_update_children)

        with ProcessPoolExecutor(max_workers=self.workers, mp_context=get_context("spawn"),
                                initializer=initializer, initargs=initargs) as executor:

            def _submit(node):
                node_path = self.data_handler.spill_path(node.filter_dict)
                children_filter_dicts = [child.filter_dict for child in node.children]
                children_ids = [child.id for child in node.children]
                children_level = node.children[0].level
                is_leaf = node.children[0].is_leaf()

                return executor.submit(worker, node.id, node_path,
                                     children_filter_dicts, children_ids, children_level, is_leaf)

            # Priority queue ordered by number of children: nodes with more children are
            # dispatched first so the executor stays busy with the heavier work.
            pending = []
            heapq.heappush(pending, (-len(root.children), root.id, root))

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
                            heapq.heappush(pending, (-len(child.children), child.id, child))

                _fill_window()

    def _solve_root(self, root: HierarchicalNode) -> None:
        '''Materialize, noise and solve the root, then spill it for the worker pool.

        The root is the only node solved on its own (every other node is solved jointly
        with its siblings under the parent's totals), and it is solved in the main process.

        In the marginal pipeline the root also needs its own separator-consistency rows:
        nothing else would force its bags to agree with each other, and every descendant
        inherits its totals.

        Args:
            root (HierarchicalNode): The tree root.
        '''
        path = self.data_handler.spill_path(root.filter_dict)

        if self.junction_tree is not None: # Marginal case
            marginals, constraints = self.data_handler.materialize_node_marginals(
                root.filter_dict, self.constraints[root.level])
            measurement = np.concatenate(marginals)
            constraints = constraints + marginal_estimation.separator_constraints(self.data_handler)
        else: # Full-joint case
            measurement, constraints = self.data_handler.materialize_node_data(
                root.filter_dict, self.constraints[root.level], self.Q)

        try: # Add noise to the materialized measurement vector, using precomputed noise if available.
            self.privacy_mechanism.add_noise_from_precomputed(
                self.data_handler.noise_zarr_group[self.data_handler.noisy_array_name],
                measurement, root.id)
        except (IndexError, ValueError, KeyError, OSError):
            self.privacy_mechanism.add_noise(measurement, root.level, self.query_sensitivity)

        solution = self._estimate_node_individually(root.id, measurement, constraints)

        self.data_handler.spill_vector(path, solution)

    def _pool_setup(self) -> Tuple[callable, Tuple]:
        '''Return the worker-pool initializer and its arguments for the active pipeline.

        The two pipelines need different worker state - the factored one ships the junction
        tree and derives the bag layout from it, the full-joint one ships Q - so they have
        separate init_process functions rather than one with optional arguments.

        Returns:
            Tuple[callable, Tuple]: (initializer, initargs) for ProcessPoolExecutor.
        '''
        common = (self.data_handler.spill_dir, self.data_handler.microdata_dir,
                  self.data_handler.file_path, self.data_handler.contingency_domain.domains,
                  self.hierarchical_columns, self.query_columns)

        if self.junction_tree is not None:
            return marginal_estimation.init_process, (
                self.optimizer, self.optimizer_backend, self.constraints, self.structural, *common,
                self.junction_tree, self.privacy_mechanism, self.query_sensitivity,
                self.check_correctness,
                self.data_handler.noise_zarr_path, self.data_handler.noisy_array_name)

        return init_process, (
            self.optimizer, self.optimizer_backend, self.constraints, self.structural, *common,
            self.Q, self.privacy_mechanism, self.query_sensitivity, self.check_correctness,
            self.data_handler.noise_zarr_path, self.data_handler.noisy_array_name)

    def _estimate_node_individually(self, node_id: int, measurement: np.ndarray,
                                    constraints: List) -> sp.csc_matrix:
        '''Solve one node's own measurement vector, with no siblings to reconcile.

        Args:
            node_id (int): The node's ID, used to name the solver's temporary files.
            measurement (np.ndarray): The node's noisy measurement vector.
            constraints (List): SparseConstraints over that vector's index space.

        Returns:
            sp.csc_matrix: Integer estimate, shape (len(measurement), 1).
        '''
        optimizer = build_optimizer(self.optimizer_backend, self.optimizer)

        t1 = time.time()
        x_tilde = optimizer.non_negative_real_estimation(
            noisy_measurements=[measurement],
            node_id=node_id,
            constraints=constraints,
            query_matrix=self.Q
        )
        real_time = time.time() - t1

        t1 = time.time()
        solution = optimizer.rounding_estimation(
            x_tilde=x_tilde,
            node_id=node_id,
            constraints=constraints
        )
        rounding_time = time.time() - t1

        print(f'  [Node {node_id}] - real {real_time:.1f}s - rounding {rounding_time:.1f}s')
        return solution

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

    def set_marginals(self, cliques: Iterable[Iterable[str]]) -> None:
        '''Switch to the factored/marginal pipeline and declare the marginals to measure.

        Instead of one contingency vector over the full joint (whose length is the product
        of all column cardinalities, and therefore unusable past a handful of columns),
        each node holds one small marginal per junction-tree bag. Consistency between
        overlapping bags replaces the joint.

        Each clique is a set of columns to keep jointly. They are embedded in an
        interaction graph which is then triangulated; the resulting maximal cliques are the
        bags actually measured, so the final bags may be LARGER than what is passed here.
        Every constraint scope registered with set_constraint_to_level is added
        automatically, so constraints are always enforceable inside some bag.

        Calling this makes Q irrelevant: the two pipelines are mutually exclusive, and
        leaving it unset keeps the full-joint behaviour.

        Args:
            cliques (Iterable[Iterable[str]]): Column subsets to keep jointly, e.g.
                [('P01', 'P02'), ('CANT_HOG', 'CANT_PER')].

        Raises:
            ValueError: If a declared column is not among the query columns.
        '''
        cliques = [list(clique) for clique in cliques]

        # Validate that the cliques only contain columns that are in the query_columns list
        unknown = {column for clique in cliques for column in clique} - set(self.query_columns)
        if unknown:
            raise ValueError(
                f"Declared marginal columns not in query_columns: {sorted(unknown)}."
            )
        self.marginal_cliques = cliques

    def set_marginal_selection(self, strategy: Optional[MarginalSelectionStrategy] = None,
                               budget_fraction: float = 0.2) -> None:
        '''Switch to the factored pipeline and choose the marginals from the data, privately.

        Which columns are worth keeping jointly depends on how they are associated, and
        association can only be learned by looking at the data - so the selection consumes
        privacy budget like any other query. All 2-way marginals are measured once over the
        whole dataset and noised together; mutual information is computed from the noisy
        tables alone, and the strategy turns it into cliques. The raw data never reaches the
        selection.

        The reserved share is taken OUT OF the mechanism's total, not added on top: the
        per-level parameters are scaled down by (1 - budget_fraction) and the reserved share
        becomes an extra composition level, so the overall guarantee is exactly what you
        configured. Spending more here buys a better structure but leaves less for the
        counts themselves.

        Can be combined with set_marginals(): the declared cliques and the selected ones are
        both embedded. Constraint scopes are added automatically in either case.

        Args:
            strategy (Optional[MarginalSelectionStrategy]): Selection heuristic. Defaults to
                MaxSpanningTreeMI - 2-way marginals along a maximum spanning tree over
                mutual information, constrained to contain the constraint cliques.
            budget_fraction (float): Share of the total budget spent on selection, in
                [0, 1). Defaults to 0.2.

        Raises:
            ValueError: If budget_fraction is outside (0, 1).
        '''
        if not 0.0 < budget_fraction < 1.0:
            raise ValueError(f"budget_fraction must be in (0, 1), got {budget_fraction}.")

        self.marginal_strategy = strategy if strategy is not None else MaxSpanningTreeMI()
        self.selection_budget_fraction = budget_fraction

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
