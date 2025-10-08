import pandas as pd
import numpy as np

from itertools import product
from typing import List, Optional, Callable

from hierarchical_tree import HierarchicalTree
from hierarchical_node import HierarchicalNode

class DataHandler:
    '''Class to handle data loading, preprocessing and postprocessing.'''

    def __init__(self, file_path, output_path: str = 'noisy_data.csv') -> None:
        '''Constructror for DataHandler class.
        
        Args:
            file_path (str): Path to the data file.
            output_path (str): Path to save the processed data.

        Attributes:
            file_path (str): Path to the data file.
            output_path (str): Path to save the processed data. Defaults to "noisy_data.csv".
            
            dataframe (pd.DataFrame): DataFrame to hold the data.
            contingency_df (pd.DataFrame): DataFrame to hold the contingency table.    

            query_columns (List[str]): List of columns to use for generating the contingency table.
            hierarchical_columns (List[str]): List of columns representing the hierarchical levels.
        '''
        # Input and output paths
        self.file_path = file_path
        self.output_path = output_path

        # Dataframe to hold the data (loaded from file_path).
        self.dataframe = None

        # Used to store and have an order on each unique combination of attributes.
        # The contingency vectors will have the same order of this dataframe.
        self.contingency_df = None

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
        self.dataframe = pd.read_csv(self.file_path, sep=sep, usecols=columns, nrows=(nrows or 0))
        return self.dataframe

    def write_data(self, data: pd.DataFrame, out_path: Optional[str]) -> None:
        '''Write the processed data to the output_path.
        
        Args:
            data (pd.DataFrame): DataFrame containing the processed data to write.
            out_path (Optional[str]): Optional path to save the processed data. If None, use self.output_path.
        '''
        data.to_csv((out_path or self.output_path), index=False)
    
    def generate_contingency_dataframe(self, query_columns: List[str]) -> pd.DataFrame:
        '''Generate a contingency dataframe from the loaded data.
        This method assumes that each column have all the possible values.

        Args:
            query_columns (List[str]): List of query columns to aggregate.

        Returns:
            pd.DataFrame: Contingency DataFrame with each unique combination of query columns.
        '''
        # Get unique values for each column
        unique_values = [self.dataframe[col].unique() for col in query_columns]

        # Generate all possible combinations (Cartesian product)
        self.contingency_df = pd.DataFrame(list(product(*unique_values)), columns=query_columns)
        
        # Sort by the columns to ensure a consistent order
        self.contingency_df.sort_values(by=query_columns, inplace=True)
        self.contingency_df.reset_index(drop=True, inplace=True)

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

    def build_hierarchical_tree(self, constraints_config=None) -> HierarchicalTree:
        '''Build a hierarchical tree based on the hierarchical columns.
        It creates a contingency vector for each node in the tree.
        
        Args:
            constraints_config: Optional configuration for setting constraints on nodes.
        
        Returns:
            HierarchicalTree: The constructed hierarchical tree.
        '''

        tree = HierarchicalTree()

        # Generate the contingency table if not already done
        if self.contingency_df is None:
            self.generate_contingency_dataframe(self.query_columns)
        
        # Contingency vector for the root node (entire dataset)
        tree.root.contingency_vector = self.create_contingency_vector(self.dataframe)
        # TODO: Set constraints for root

        # Construct the tree recursively and count the nodes created
        tree._node_count = self._build_subtree(tree.root, 0+1, self.dataframe)

        return tree
    
    def _build_subtree(self, parent_node: HierarchicalNode, current_level: int, data: pd.DataFrame, constraints: List[Callable]) -> int:
        '''Helper method to recursively build the subtree for a given parent node.
        
        Args:
            parent_node (HierarchicalNode): The parent node to which children will be added.
            current_level (int): The current level in the hierarchy being processed.
            data (pd.DataFrame): The subset of data corresponding to the parent node.
            constraints (List[Callable]): List of constraints to apply to each node.
        
        Returns:
            int: The number of nodes in the subtree.
        '''
        # When there are no more levels to process, return the parent node
        if current_level >= len(self.hierarchical_columns):
            return 0
        
        n_nodes = 1
        # Get the current hierarchical column to split on
        current_column = self.hierarchical_columns[current_level]
        unique_hierarchical_values = data[current_column].unique()

        for value in unique_hierarchical_values:
            # Filter data for the current hierarchical value
            filtered_data = data[data[current_column] == value]

            # Create a new child node
            # TODO: Implement the constraint logic after the new format is defined
            child_node = HierarchicalNode(node_id=value, constraints=constraints)
            parent_node.add_child(child_node)

            # Set the hierarchical path for the child node
            child_node.hierarchical_path = parent_node.hierarchical_path + [value]

            # Create and assign the contingency vector for the child node
            child_node.contingency_vector = self.create_contingency_vector(filtered_data)

            child_node.parent = parent_node

            # Recursively build the subtree for the child node
            n_nodes += self._build_subtree(child_node, current_level + 1, filtered_data, constraints)

        return n_nodes