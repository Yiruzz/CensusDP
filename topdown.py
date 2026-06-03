import pandas as pd
import numpy as np
from hierarchical_tree import HierarchicalTree
from data_handler import DataHandler
from optimizer import OptimizationModel
from constraints.constraint import Constraint
from queries import QueryWorkload
from privacy import PrivacyMechanism

from collections import deque
from typing import Dict, List, Union
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
                 out_path: str = 'noisy_data.csv', solver_name: str = 'gurobi',
                 solver_options: dict = None, optimizer_path: str = None) -> None:
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

        
        self.hierarchical_columns: List[str] = hierarchy
        self.query_columns: List[str] = query_columns

        self.data_handler: DataHandler = DataHandler(file_path=data_path, output_path=out_path)
        self.data_handler.hierarchical_columns = hierarchy
        self.data_handler.query_columns = query_columns

        self.privacy_mechanism: PrivacyMechanism = privacy_mechanism

        self.Q: Union[QueryWorkload, np.ndarray, None] = None  # set via set_query_workload(); resolved in initialize()
        self.query_sensitivity: int = 1  # L1 sensitivity of Q; computed in initialize() once Q is materialized

        self.constraints: Dict[int, List[Constraint]] = {i: [] for i in range(len(hierarchy) + 1)}

        self.tree: HierarchicalTree = HierarchicalTree()

        if solver_options is None:
            solver_options = {}

        self.optimizer: OptimizationModel = OptimizationModel(
            solver_name=solver_name,
            solver_options=solver_options,
            optimizer_path=optimizer_path
        )

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

        print(f'Computing query matrix Q...', end=' ')
        if isinstance(self.Q, QueryWorkload):
            self.Q = self.Q.build(self.data_handler.contingency_df)
        elif not isinstance(self.Q, np.ndarray):
            # No workload set — use identity. NOTE: np.eye(n_cells) is dense; avoid for large domains.
            self.Q = np.eye(len(self.data_handler.contingency_df), dtype=np.int_)
        # NOTE: Privacy guarantees rely on Q being binary so that the L1 sensitivity (max column sum) is well defined
        #       and coincides with the squared L2 sensitivity. If Q is not binary, the privacy guarantees may not hold.
        #assert np.all((self.Q == 0) | (self.Q == 1)), \
            "Q must be binary (entries in {0,1}) for the column-sum sensitivity reasoning."
        self.query_sensitivity = int(self.Q.sum(axis=0).max())
        print(f'\n  Query matrix: n_queries={self.Q.shape[0]}, sensitivity={self.query_sensitivity}')
        print(f'  Privacy mechanism: {self.privacy_mechanism.report_guarantee()}')

        t1 = time.time()
        print(f'Building hierarchical tree structure...', end=' ')
        self.tree = self.data_handler.build_hierarchical_tree()
        print(f'{time.time() - t1:.2f} seconds.\n')

        # Initialize output file with headers
        self.data_handler.initialize_output_file()

        print(self.tree, "\n")

    def estimation_phase(self) -> None:
        '''Perform the estimation phase of the TopDown algorithm.

        Uses breadth-first traversal with lazy materialization to minimize memory usage.
        '''
        print(f'Running estimation phase...')
        t1 = time.time()

        # Materialize root and its children immediately
        root = self.tree.root
        root.contingency_vector, root.constraints = self.data_handler.materialize_node_data(root.hierarchical_path, self.constraints[root.level])
        self.privacy_mechanism.add_noise(root.contingency_vector, root.level, self.query_sensitivity)

        # First phase: resolve root's own contingency vector
        self._estimate_node_individually(root)

        queue = deque()
        for child in root.children:
            child.contingency_vector, child.constraints = self.data_handler.materialize_node_data(child.hierarchical_path, self.constraints[child.level])
            self.privacy_mechanism.add_noise(child.contingency_vector, child.level, self.query_sensitivity)
            queue.append(child)

        # Second phase: resolve root considering its children and update their vectors
        self._estimate_and_update_children(root)
        self._check_correctness_node(root)

        # Free root's memory immediately after materializing children
        root.contingency_vector = None
        root.constraints = None

        # Process remaining nodes
        while queue:
            node = queue.popleft()

            # If node is a leaf, construct and write its microdata
            if node.is_leaf():
                self.data_handler.write_microdata_for_leaf(node)
                continue

            # Materialize all children's vectors
            for child in node.children:
                child.contingency_vector, child.constraints = self.data_handler.materialize_node_data(child.hierarchical_path, self.constraints[child.level])
                self.privacy_mechanism.add_noise(child.contingency_vector, child.level, self.query_sensitivity)
                queue.append(child)

            self._estimate_and_update_children(node)
            self._check_correctness_node(node)

            # Free memory: delete current node's vector
            node.contingency_vector = None
            node.constraints = None

        print(f'{time.time() - t1:.2f} seconds.\n')

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

    def _estimate_node_individually(self, node) -> None:
        '''Solve optimization for a node's own contingency vector.

        Args:
            node (HierarchicalNode): The node to process.
        '''
        print(f'  Estimating node {node.geo_id} individually...', end=' ')

        t1 = time.time()
        x_tilde = self.optimizer.non_negative_real_estimation(
            noisy_measurements=node.contingency_vector,
            node_id=node.geo_id,
            constraints=node.constraints,
            query_matrix=self.Q
        )
        non_neg_time = time.time() - t1

        t1 = time.time()
        node.contingency_vector = self.optimizer.rounding_estimation(
            x_tilde=x_tilde,
            node_id=node.geo_id,
            constraints=node.constraints
        )
        rounding_time = time.time() - t1

        print(f'non negative {non_neg_time:.1f}s - rounding {rounding_time:.1f}s')

    def _estimate_and_update_children(self, node) -> None:
        '''Solve optimization for a node considering its children and update their vectors.

        Args:
            node (HierarchicalNode): The node to process.
        '''
        print(f'  Estimating node {node.geo_id} with children...', end=' ')

        joint_contingency_vector = node.combine_child_vectors()
        constraints = node.combine_child_constraints()

        t1 = time.time()
        x_tilde = self.optimizer.non_negative_real_estimation(
            noisy_measurements=joint_contingency_vector,
            node_id=node.geo_id,
            constraints=constraints,
            query_matrix=self.Q
        )
        non_neg_time = time.time() - t1

        t1 = time.time()
        joint_solution = self.optimizer.rounding_estimation(
            x_tilde=x_tilde,
            node_id=node.geo_id,
            constraints=constraints
        )
        rounding_time = time.time() - t1

        node.update_child_vectors(joint_solution)

        print(f'non negative {non_neg_time:.1f}s - rounding {rounding_time:.1f}s')
  
    def _check_correctness_node(self, node) -> None:
        '''Checks that the sum of the values of the current node are equal to the sum of the values of its children.

        Args:
            node (HierarchicalNode): The node to check.
        '''
        node_sum = sum(node.contingency_vector)
        children_sum = 0
        for child in node.children:
            children_sum += np.sum(child.contingency_vector)

        if node_sum != children_sum:
            print(node_sum, children_sum)
            print(f'\nError: The sum of the contingency vectors of the children nodes is not equal to the parent node\'s contingency vector.')
    
    def run(self) -> None:
        '''Run the TopDown algorithm end-to-end.

        This method executes the full TopDown algorithm, including initialization,
        measurement phase, estimation phase, and microdata construction.

        Returns:
            pd.DataFrame: The constructed differentially private microdata.
        '''
        self.initialize()
        self.estimation_phase()