import pandas as pd
import numpy as np

from itertools import product
from typing import List, Optional

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

            query_columns (List[str]): List of columns to use for generating the contingency table.
            hierarchical_columns (List[str]): List of columns representing the hierarchical levels.
        '''
        # Input and output paths
        self.file_path: str = file_path
        self.output_path: str = output_path

        # Dataframe to hold the data (loaded from file_path).
        self.dataframe: Optional[pd.DataFrame] = None

        # Used to store and have an order on each unique combination of attributes.
        # The contingency vectors will have the same order of this dataframe.
        self.contingency_df: Optional[pd.DataFrame] = None

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

    def write_data(self, data: pd.DataFrame, out_path: Optional[str] = None) -> None:
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
        # NOTE: Here we assume that each column contains all possible values
        # Example: if the domain of "Age" is [0, ..., 100], we assume that the column contains all those values
        assert self.dataframe is not None, "Dataframe is not loaded. Call read_data first."
        unique_values = [self.dataframe[col].unique() for col in query_columns]

        # Generate all possible combinations (Cartesian product)
        self.contingency_df = pd.DataFrame(list(product(*unique_values)), columns=query_columns)
        
        # Sort by the columns to ensure a consistent order
        self.contingency_df.sort_values(by=query_columns, inplace=True)
        self.contingency_df.reset_index(drop=True, inplace=True)

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

    def build_hierarchical_tree(self, constraints: dict[int, List[Constraint]]) -> HierarchicalTree:
        '''Build a hierarchical tree based on the hierarchical columns.
        It creates a contingency vector for each node in the tree.
        
        Args:
            constraints (Dict[int, List[Callable]]): Dictionary mapping tree levels to their constraints.
        
        Returns:
            HierarchicalTree: The constructed hierarchical tree.
        '''

        tree = HierarchicalTree()

        # Generate the contingency table if not already done
        if self.contingency_df is None:
            self.generate_contingency_dataframe(self.query_columns)
        
        assert self.dataframe is not None, "Dataframe is not loaded. Call read_data first."
        # Contingency vector for the root node (entire dataset)
        tree.root.contingency_vector = self.create_contingency_vector(self.dataframe)
        
        # List of constraints for the root node
        root_contstraints = []
        if constraints and 0 in constraints:
            # Iterate over the constraints for the root node
            for constraint in constraints[0]:
                # Case when the constraint is a ContextualAggregateConstraint and needs to compute its value
                match constraint:
                    case ContextualAggregateConstraint():
                        constraint.apply_aggregation_function(self.dataframe)
                # Append the constraint function to the root constraints list
                assert self.contingency_df is not None, "Contingency DataFrame is not generated. Call generate_contingency_table first."
                root_contstraints.append(constraint.to_constraint(self.contingency_df))

        tree.root.constraints = root_contstraints

        # Construct the tree recursively and count the nodes created
        tree._node_count = self._build_subtree(tree.root, 0, self.dataframe, constraints)
        return tree
    
    def _build_subtree(self, parent_node: HierarchicalNode, level_iterator: int, data: pd.DataFrame, constraints: dict[int, List[Constraint]]) -> int:
        '''Helper method to recursively build the subtree for a given parent node.
        
        Args:
            parent_node (HierarchicalNode): The parent node to which children will be added.
            level_iterator (int): An iterator for the current level in the hierarchy. It has an offset of 1.
            data (pd.DataFrame): The subset of data corresponding to the parent node.
            constraints (List[Callable]): List of constraints to apply to each node.
        
        Returns:
            int: The number of nodes in the subtree.
        '''
        # When there are no more levels to process, return the parent node
        if level_iterator >= len(self.hierarchical_columns):
            return 0
        
        n_nodes = 1
        # Get the current hierarchical column to split on
        current_column = self.hierarchical_columns[level_iterator]
        unique_hierarchical_values = data[current_column].unique()

        for value in unique_hierarchical_values:
            # Filter data for the current hierarchical value
            filtered_data = data[data[current_column] == value]

            # Prepare constraints for the current level
            if constraints and level_iterator in constraints:
                level_constraints = []
                # Iterate over the constraints for the current level
                for constraint in constraints[level_iterator]:
                    # Case when the constraint is a ContextualAggregateConstraint and needs to compute its value
                    match constraint:
                        case ContextualAggregateConstraint():
                            constraint.apply_aggregation_function(filtered_data)
                    # Append the constraint function to the level constraints list
                    assert self.contingency_df is not None, "Contingency DataFrame is not generated. Call generate_contingency_table first."
                    level_constraints.append(constraint.to_constraint(self.contingency_df))


            # Create a new child node
            child_node = HierarchicalNode(node_id=value, constraints=level_constraints)
            parent_node.add_child(child_node)

            # Create and assign the contingency vector for the child node
            child_node.contingency_vector = self.create_contingency_vector(filtered_data)

            child_node.parent = parent_node

            # Recursively build the subtree for the child node
            n_nodes += self._build_subtree(child_node, level_iterator + 1, filtered_data, constraints)

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
        microdata_dict: dict[str, list] = {col: [] for col in self.hierarchical_columns+self.query_columns}
        for leaf in list(tree.iterate_by_levels())[-1][1]:
            # Create a Diccionary to store the microdata for the current node
            leaf_dict: dict[str, list] = {col: [] for col in self.hierarchical_columns+self.query_columns}
            
            assert self.contingency_df is not None, "Contingency DataFrame is not generated. Call generate_contingency_table first."
            # TODO: See if this can be optimized further. Iterrows is slow, but since the data type can vary, it is not trivial to vectorize.
            for index, (_, row) in enumerate(self.contingency_df.iterrows()):
                for col in self.query_columns:
                    # Add the value of the row[col] node.contingency_vector[index] times to the dictionary
                    leaf_dict[col].extend(np.repeat(row[col], leaf.contingency_vector[index]))

            # Add the hierarchical information for the current node
            # Determine how many microdata rows this leaf contributes (based on query columns)
            leaf_size = len(leaf_dict[self.query_columns[0]])
            # Offset to track the level in the hierarchy from top to bottom
            current_level = 0
            # We do not include the root node in the hierarchical path as it is redundant in the final
            # data because the root is the highest level of the hierarchy and all data belongs to it
            for hierarchical_value in leaf.hierarchical_path[1:]:
                leaf_dict[self.hierarchical_columns[current_level]] = list(np.repeat(hierarchical_value, leaf_size))
                current_level += 1
            
            # Also don't forget to add the information of the leaf node itself
            leaf_dict[self.hierarchical_columns[current_level]] = list(np.repeat(leaf.id, leaf_size))

            # Merge the leaf_dict into microdata_dict by concatenating lists for duplicate keys
            for key, values in leaf_dict.items():
                microdata_dict[key].extend(values)

        return pd.DataFrame(microdata_dict)

