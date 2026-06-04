import os
import tempfile
import shutil
import pandas as pd
import numpy as np

from itertools import product
from typing import List, Optional, Tuple
from pathlib import Path

from constraints.constraint import Constraint
from constraints.contextual_constraints import ContextualAggregateConstraint
from hierarchical_tree import HierarchicalTree
from hierarchical_node import HierarchicalNode

class DataHandler:
    '''Class to handle data loading, preprocessing and postprocessing.'''

    def __init__(self, file_path: str, output_path: str = 'noisy_data.csv') -> None:
        '''Constructror for DataHandler class.
        
        Args:
            file_path (str): Path to the data file.
            output_path (str): Path to save the processed data.

        Attributes:
            file_path (str): Path to the data file.
            output_path (str): Path to save the processed data. Defaults to "noisy_data.csv".
            
            dataframe (Optional[pd.DataFrame]): DataFrame to hold the data.
            contingency_df (Optional[pd.DataFrame]): DataFrame to hold the contingency table.    
            contingency_df_length: Optional[int]: Length of the contingency dataframe.
            dtype (str): NumPy data type used for all arrays.

            query_columns (List[str]): List of columns to use for generating the contingency table.
            hierarchical_columns (List[str]): List of columns representing the hierarchical levels.
        '''
        # Input and output paths
        self.file_path: str = file_path

        path_obj = Path(output_path)
        if path_obj.parent.exists() and path_obj.parent.is_dir():
            self.output_path = output_path
        else:
            print(f"Warning: Output path {output_path} is not valid. 'noisy_data.csv' will be saved in the current directory instead.")
            self.output_path = 'noisy_data.csv'

        # Dataframe to hold the data (loaded from file_path).
        self.dataframe: Optional[pd.DataFrame] = None

        # Used to store and have an order on each unique combination of attributes.
        # The contingency vectors will have the same order of this dataframe.
        self.contingency_df: Optional[pd.DataFrame] = None
        self.contingency_df_length: Optional[pd.DataFrame] = None
        self.dtype: str = 'int64'

        # Hierarchical columns (first value highest hierarchy, last lowest).
        self.hierarchical_columns: List[str] = []

        # Query columns (not considered for the hierarchy).
        self.query_columns: List[str] = []

        # Define path to dir to save vectors
        self.spill_dir: str = os.path.join(tempfile.gettempdir(), f'topdown_spill_{os.getpid()}')

    def initialize_output_file(self) -> None:
        '''Initialize the output CSV file with column headers.'''
        # Create empty DataFrame with the expected columns
        empty_df = pd.DataFrame(columns=self.hierarchical_columns + self.query_columns)
        empty_df.to_csv(self.output_path, index=False, header=True)

    def read_data(self, columns: List[str], sep: str = ',', nrows: Optional[int] = None) -> pd.DataFrame:
        '''Read data from the file_path into a pandas DataFrame.
        
        Args:
            sep (str): Separator used in the CSV file. Defaults to ",".
            columns (List[str]): List of columns to read from the CSV file.
            nrows (Optional[int]): Number of rows to read from the CSV file. If None, read all rows. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        '''
        if nrows is not None and nrows > 0:
            self.dataframe = pd.read_csv(self.file_path, sep=sep, usecols=columns, nrows=nrows)
        else:
            self.dataframe = pd.read_csv(self.file_path, sep=sep, usecols=columns)

        return self.dataframe
    
    def generate_contingency_dataframe(self, query_columns: List[str]) -> pd.DataFrame:
        '''Generate a contingency dataframe from the loaded data.
        This method assumes that each column have all the possible values.

        Args:
            query_columns (List[str]): List of query columns to aggregate.

        Returns:
            pd.DataFrame: Contingency DataFrame with each unique combination of query columns.
        '''
        # Get unique values for each column
        # NOTE: Here we assume that each column contains all possible values
        # Example: if the domain of "Age" is [0, ..., 100], we assume that the column contains all those values
        assert self.dataframe is not None, "Dataframe is not loaded. Call read_data first."
        unique_values = [self.dataframe[col].unique() for col in query_columns]

        # Generate all possible combinations (Cartesian product)
        self.contingency_df = pd.DataFrame(list(product(*unique_values)), columns=query_columns)
        
        # Sort by the columns to ensure a consistent order
        self.contingency_df.sort_values(by=query_columns, inplace=True)
        self.contingency_df.reset_index(drop=True, inplace=True)

        self.contingency_df_length = self.contingency_df.shape[0]

        print("Contingency DataFrame generated with shape:", self.contingency_df.shape, "in", end=' ')

        return self.contingency_df
    
    def create_contingency_vector(self, hierarchical_path: List[int]) -> np.ndarray:
        '''Create a contingency vector from a dataframe, optionally filtered by hierarchical path.

        Args:
            hierarchical_path (Optional[List[Int]]): Filters df using this path before creating the vector.

        Returns:
            np.ndarray: Contingency vector with counts for each unique combination in contingency_df.
        '''
        if self.contingency_df is None:
            raise ValueError("Contingency DataFrame is not generated. Call generate_contingency_table first.")

        filtered_df = self.dataframe
        if len(hierarchical_path) > 1:
            for level, value in enumerate(hierarchical_path[1:]):
                column = self.hierarchical_columns[level]
                filtered_df = filtered_df[filtered_df[column] == value]

        queries = self.contingency_df.columns.tolist()

        # Group the data by the permutation columns and count occurrences
        grouped = filtered_df.value_counts(subset=queries).reset_index(name='frequency')

        # Merge to get frequencies for all combinations, ensuring correct order and filling missing with 0.
        # This merged dataframe now contains all combinations from self.contingency_df and their counts (0 if not present in df).
        merged = pd.merge(self.contingency_df, grouped, how='left', on=queries).fillna({'frequency': 0})

        # Directly extract the 'frequency' column as a numpy array.
        contingency_vector = merged['frequency'].to_numpy(dtype=int)

        return contingency_vector

    def build_hierarchical_tree(self) -> HierarchicalTree:
        '''Build a hierarchical tree structure based on hierarchical columns.

        This creates only the tree structure using hierarchical_path for each node.
        Contingency vectors, constraints, and query matrix information are NOT handled here.

        Returns:
            HierarchicalTree: The constructed hierarchical tree with hierarchical_path information.
        '''
        assert self.dataframe is not None, "Dataframe is not loaded. Call read_data first."
        assert self.hierarchical_columns, "Hierarchical columns not set."

        tree = HierarchicalTree()
        root = tree.root

        # Build tree structure recursively, starting with the full dataframe
        tree._node_count = self._build_subtree(root, 0, self.dataframe)
        tree._levels = 1+len(self.hierarchical_columns)

        return tree

    def _build_subtree(self, parent_node: HierarchicalNode, level_iterator: int, data: pd.DataFrame) -> int:
        '''Helper method to recursively build the subtree for a given parent node.

        Creates only the tree structure. Receives the pre-filtered dataframe for this node's subset.

        Args:
            parent_node (HierarchicalNode): The parent node to which children will be added.
            level_iterator (int): An iterator for the current level in the hierarchy.
            data (pd.DataFrame): The dataframe filtered to contain only rows for this node's subset.

        Returns:
            int: The number of nodes in the subtree.
        '''
        n_nodes = 1

        if level_iterator >= len(self.hierarchical_columns):
            return n_nodes

        current_column = self.hierarchical_columns[level_iterator]
        unique_hierarchical_values = data[current_column].unique()

        for value in unique_hierarchical_values:
            child_node = HierarchicalNode(geo_id=value, level=level_iterator+1)
            parent_node.add_child(child_node)

            # Filter data for this child and pass to recursion
            child_data = data[data[current_column] == value]
            n_nodes += self._build_subtree(child_node, level_iterator + 1, child_data)

        return n_nodes

    def _spill_path(self, node: HierarchicalNode) -> str:
        '''Get the spill file path for a node based on its hierarchical_path.

        Args:
            node (HierarchicalNode): The node whose spill path is being determined.

        Returns:
            str: File path for the spilled vector named by the node's hierarchical_path.
        '''
        name = '_'.join(str(p) for p in node.hierarchical_path)
        for ch in ('/', '\\', ' ', ':'):
            name = name.replace(ch, '_')
        return os.path.join(self.spill_dir, name + '.npy')

    def spill_vector(self, node: HierarchicalNode) -> None:
        '''Write node.contingency_vector to disk and free it from RAM.

        One file per node, named by its hierarchical_path, ensuring sibling nodes don't conflict.

        Args:
            node (HierarchicalNode): The node whose contingency vector should be spilled.
        '''
        os.makedirs(self.spill_dir, exist_ok=True)
        np.save(self._spill_path(node), np.ascontiguousarray(node.contingency_vector))
        node.contingency_vector = None
        node.constraints = None

    def load_vector(self, node: HierarchicalNode) -> None:
        '''Reload node.contingency_vector from disk and delete the file (used once).

        Args:
            node (HierarchicalNode): The node whose contingency vector should be reloaded.
        '''
        node.contingency_vector = np.load(self._spill_path(node))
        os.remove(self._spill_path(node))

    def cleanup_spill(self) -> None:
        '''Delete the spill directory and all remaining spilled vectors.'''
        if os.path.isdir(self.spill_dir):
            shutil.rmtree(self.spill_dir, ignore_errors=True)

    def _construct_microdata_for_leaf(self, node) -> pd.DataFrame:
        '''Construct microdata for a specific leaf node.

        Args:
            node (HierarchicalNode): The leaf node with materialized contingency vector.

        Returns:
            pd.DataFrame: Microdata for this leaf node.
        '''
        assert self.contingency_df is not None, "Contingency DataFrame is not generated. Call generate_contingency_table first."

        if not node.is_leaf():
            raise ValueError("Node must be a leaf node.")

        if node.contingency_vector is None:
            raise ValueError("Node contingency vector is not materialized.")

        # Get all possible query column combinations.
        query_values = self.contingency_df[self.query_columns].to_numpy()

        # Select only positive frequencies.
        nonzero_mask = node.contingency_vector > 0

        # Filter combinations to avoid processing zero-frequency rows.
        filtered_query_values = query_values[nonzero_mask]
        filtered_counts = node.contingency_vector[nonzero_mask]

        # Repeat each combination according to its frequency.
        expanded_rows = np.repeat(filtered_query_values, filtered_counts, axis=0)

        # Create DataFrame for the leaf.
        leaf_df = pd.DataFrame(expanded_rows, columns=self.query_columns)

        # Generate columns associated with hierarchical values.
        # Skip the root node because all records belong to it.
        for level, hierarchical_value in enumerate(node.hierarchical_path[1:]):
            # Repeat the hierarchical value for all rows.
            leaf_df[self.hierarchical_columns[level]] = hierarchical_value

        # Reorder columns to match output file order: hierarchical + query
        output_columns = self.hierarchical_columns + self.query_columns
        return leaf_df[output_columns]

    def write_microdata_for_leaf(self, node: HierarchicalNode) -> None:
        '''Construct microdata for a leaf node and append to output file.

        Args:
            node (HierarchicalNode): The leaf node with materialized contingency vector.
        '''
        # Construct microdata for this leaf
        leaf_microdata = self._construct_microdata_for_leaf(node)
        leaf_microdata.to_csv(self.output_path, mode='a', header=False, index=False)
        node.contingency_vector = None
        node.constraints = None

    def materialize_node_data(self, hierarchical_path: List[int], constraints: List[Constraint], query_matrix: np.ndarray) -> Tuple[np.ndarray, List]:
        '''Materialize contingency vector and prepare constraints in a single pass.

        Filters data once based on hierarchical path, then creates the contingency vector
        and prepares all constraints for the node.

        Args:
            hierarchical_path (List[int]): The node's hierarchical path for filtering.
            constraints (List[Constraint]): Constraints for the node considering its level.
            query_matrix (np.ndarray): Query matrix for aggregating contingency vectors.

        Returns:
            Tuple[np.ndarray, List[Constraint]]: Contingency vector and constraint callables for this node.
        '''
        if self.contingency_df is None:
            raise ValueError("Contingency DataFrame not generated. Call generate_contingency_dataframe first.")

        # Filter data once based on hierarchical path
        filtered_df = self.dataframe
        if len(hierarchical_path) > 1:
            for level_idx, value in enumerate(hierarchical_path[1:]):
                column = self.hierarchical_columns[level_idx]
                filtered_df = filtered_df[filtered_df[column] == value]

        # Create contingency vector from filtered data
        queries = self.contingency_df.columns.tolist()
        grouped = filtered_df.value_counts(subset=queries).reset_index(name='frequency')
        merged = pd.merge(self.contingency_df, grouped, how='left', on=queries).fillna({'frequency': 0})
        contingency_vector = query_matrix @ merged['frequency'].to_numpy(dtype=int) 

        # Prepare constraints using the same filtered data
        level_constraints = []
        for constraint in constraints:
            # Apply aggregation for constraints that compute dynamically
            match constraint:
                case ContextualAggregateConstraint():
                    constraint.apply_aggregation_function(filtered_df)

            # Convert to optimizer function
            level_constraints.append(constraint.to_constraint(self.contingency_df))

        return contingency_vector, level_constraints

