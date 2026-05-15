import pandas as pd
import numpy as np
from hierarchical_tree import HierarchicalTree
from data_handler import DataHandler
from optimizer import OptimizationModel
from constraints.constraint import Constraint
from queries import QueryWorkload
from privacy import PrivacyMechanism

from typing import Callable, Dict, List, Optional, Union
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
        self.optimizer: OptimizationModel = OptimizationModel(optimizer, solver_options, optimizer_path)
        
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

        return None

    def measurement_phase(self) -> None:
        '''Perform the measurement phase of the TopDown algorithm.

        Each node's contingency_vector already holds y = Q @ x from tree construction.
        This phase adds discrete noise calibrated to the sensitivity of Q in place: y <- y + noise.
        '''
        t1 = time.time()
        print(f'Running measurement phase (query_sensitivity={self.query_sensitivity})...\n')
        for level, nodes in self.tree.iterate_by_levels():
            t2 = time.time()
            print(f'Processing level {level} with {len(nodes)} nodes...', end=' ')
            for node in nodes:
                noise = np.array([
                    self.privacy_mechanism.sample_noise(level, self.query_sensitivity)
                    for _ in range(len(node.contingency_vector))
                ])
                node.contingency_vector = node.contingency_vector + noise
            print(f'{time.time() - t2:.2f} seconds.')
        print(f'Measurement phase completed in {time.time() - t1:.2f} seconds.\n')

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
        print(f'Running estimation phase...')

        # Root estimation (level 0)
        # Does not require consistency adjustments
        t2 = time.time()
        print(f'\nProcessing root node (level 0)... ', end=' ')
        x_tilde: np.ndarray = self.optimizer.non_negative_real_estimation(
            noisy_measurements=self.tree.root.contingency_vector,
            id_node=self.tree.root.id,
            constraints=self.tree.root.constraints,
            query_matrix=self.Q
        )
        self.tree.root.contingency_vector = self.optimizer.rounding_estimation(
            x_tilde=x_tilde,
            id_node=self.tree.root.id,
            constraints=self.tree.root.constraints
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
                vectors_length = self.Q.shape[1]  # n_cells per child in decision-variable space
                # Children still hold y = Q @ x + noise in their slot at this point.
                joint_noisy_measurements = np.concatenate([child.contingency_vector for child in node.children])
                # All nodes have the same length of the contingency vector
                joint_x_length = len(node.children) * vectors_length

                # Transform individual constraints for joint vector
                joint_constraints: List[Callable] = []
                start = 0
                for child in node.children:
                    end = start + vectors_length
                    for constraint in child.constraints:
                        # NOTE: We use default arguments to avoid late binding issues in lambdas
                        # This can lead to all constraints using the last values saved of start and end
                        # Build a sub-dict with keys 0..(e-s-1) so the constraint's indices still match
                        joint_constraints.append(lambda joint_array, s=start, e=end, c=constraint: c({i - s: joint_array[i] for i in range(s, e)}))
                    start = end

                # Consistency constraint: sum of children = parent
                for index in range(vectors_length):
                    # Parent's contingency vector value at 'index' must equal sum of children's values at 'index'
                    # Precompute the indices to sum to avoid slice notation incompatible with Pyomo vars
                    indices_to_sum = list(range(index, joint_x_length, vectors_length))
                    joint_constraints.append(lambda joint_array, idxs=indices_to_sum, value=node.contingency_vector[index]:
                                             sum(joint_array[j] for j in idxs) == value)

                # Solve for children nodes
                x_tilde = self.optimizer.non_negative_real_estimation(
                    noisy_measurements=joint_noisy_measurements,
                    id_node=node.id,
                    constraints=joint_constraints,
                    query_matrix=self.Q
                )
                joint_solution: np.ndarray = self.optimizer.rounding_estimation(
                    x_tilde=x_tilde,
                    id_node=node.id,
                    constraints=joint_constraints
                )

                # Save the solution back to each child node
                start = 0
                for child in node.children:
                    end = start + vectors_length
                    child.contingency_vector = joint_solution[start:end]
                    start = end

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
        self.data_handler.write_data(noisy_df)
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
        if self.tree.root is not None:
            print(f'Checking correctness of the tree...')
            time1 = time.time()
            self._check_correctness_node(self.tree.root)
            time2 = time.time()
            print(f'Finished checking correctness in {time2-time1} seconds.\n')


    def _check_correctness_node(self, node) -> None:
        '''Checks that the sum of the values of the current node are equal to the sum of the values of its children.

        Args:
            node (HierarchicalNode): The node to check.
        '''
        if node.children:
            # Check if the sum of the contingency vectors of the children nodes is equal to the parent node's contingency vector
            node_sum = np.sum(node.contingency_vector)
            children_sum = 0
            for child in node.children:
                children_sum += np.sum(child.contingency_vector)

            if node_sum != children_sum:            
                print(f'\nError: The sum of the contingency vectors of the children nodes is not equal to the parent node\'s contingency vector.')
                print(f'Parent node contingency vector: {node.contingency_vector}')
                raise ValueError('Tree correctness check failed.')
            else:
                for child in node.children:
                    self._check_correctness_node(child)