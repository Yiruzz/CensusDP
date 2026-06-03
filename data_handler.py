import pandas as pd
import numpy as np
from collections import deque

from multiprocessing import shared_memory
from itertools import product
from typing import List, Optional, Tuple
from pathlib import Path

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

    def write_data(self, data: pd.DataFrame, out_path: Optional[str] = None, cols: list[str] = None) -> None:
        '''Write the processed data to the output_path.
        
        Args:
            data (pd.DataFrame): DataFrame containing the processed data to write.
            out_path (Optional[str]): Optional path to save the processed data. If None, use self.output_path.
            cols (Optional[list[str]]): Optional columns to write. Useful to avoid reordering the DataFrame.
        '''
        data.to_csv((out_path or self.output_path), columns=cols,  index=False)
    
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
    
    def create_contingency_vector(self, df: pd.DataFrame) -> np.ndarray:
        '''Create a contingency vector from the given dataframe with the same order as contingency_df.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data to aggregate.

        Returns:
            np.ndarray: Contingency vector with counts for each unique combination in contingency_df.
        '''
        if self.contingency_df is None:
            raise ValueError("Contingency DataFrame is not generated. Call generate_contingency_table first.")

        queries = self.contingency_df.columns.tolist()

        # Group the data by the permutation columns and count occurrences
        grouped = df.value_counts(subset=queries).reset_index(name='frequency')

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

    def construct_microdata(self, tree: HierarchicalTree) -> pd.DataFrame:
        '''Construct microdata from the hierarchical tree.
        
        This method traverses the hierarchical tree and reconstructs the microdata
        based on the contingency vectors at each node.

        Args:
            tree (HierarchicalTree): The hierarchical tree containing contingency vectors.

        Returns:
            pd.DataFrame: The reconstructed microdata.
        '''
        
        assert self.contingency_df is not None, ("Contingency DataFrame is not generated. Call generate_contingency_table first.")

        # Get all possible query column combinations.
        query_values = self.contingency_df[self.query_columns].to_numpy()

        # Store partial DataFrames generated for each leaf.
        microdata_parts = []

        start_node_idx = tree._levels[-1]
        for leaf in tree.nodes[start_node_idx:]:
            # Generate rows associated with query values.
            # Select only positive frequencies.
            # After estimation, only the first n_cells slots of the row hold the x_hat values;
            # the trailing slots (if vector_length > n_cells) are unused padding.
            contingency_vector = tree._contingency_vectors[leaf.id][:tree.n_cells]
            nonzero_mask = contingency_vector > 0

            # Filter combinations to avoid processing zero-frequency rows.
            filtered_query_values = query_values[nonzero_mask]
            filtered_counts = contingency_vector[nonzero_mask]

            # Repeat each combination according to its frequency.
            expanded_rows = np.repeat(filtered_query_values, filtered_counts, axis=0)

            # Create partial DataFrame for the current leaf.
            leaf_df = pd.DataFrame(expanded_rows, columns=self.query_columns)

            # Generate columns associated with hierarchical values.
            # Skip the root node because all records belong to it.
            current_level = 0
            for hierarchical_value in leaf.hierarchical_path[1:]:
                # Repeat the hierarchical value for all rows.
                leaf_df[self.hierarchical_columns[current_level]] = hierarchical_value

                current_level += 1
            
            # Add the leaf node identifier.
            leaf_df[self.hierarchical_columns[current_level]] = leaf.geo_id
            microdata_parts.append(leaf_df)

        microdata = pd.concat(microdata_parts, ignore_index=True)
        return microdata

