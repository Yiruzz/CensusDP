import pandas as pd
import numpy as np
from collections import deque

from multiprocessing import shared_memory
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
    
    def convert_tree_representation(self, root: HierarchicalNode, n_nodes: int) -> Tuple[List[HierarchicalNode], List[int], np.ndarray, memoryview]:
        '''Retrieve the references of all nodes of the tree to facilitate access. 
        Also create a shared memory space with all contingency vectors copied,
        enabling multiprocessing without duplicating data.

        Args:
            root (HierarchicalNode): A hierarchical tree.
            n_nodes (int): Number of nodes, corresponding to the number of rows in the shared memory space.

        Return:
            tuple[list[HierarchicalNode], List[int], np.ndarray, memoryview]: A tuple containing all node references, the node index where each level starts,
                                                                              the view over the shared memory buffer, and the memoryview used to access the shared memory.
        '''
        shape = (n_nodes, self.contingency_df_length)
        size = int(np.prod(shape) * np.dtype(self.dtype).itemsize)

        # Create share memory space  
        shm = shared_memory.SharedMemory(create=True, size=size)

        # Create view to manipulate shared memory
        arr = np.ndarray(shape, dtype=self.dtype, buffer=shm.buf)

        # Retrieve node references
        nodes = []
        queue = deque([root]) 

        next_id = 0

        # Retrive starts levels starting COUNTRY (ROOT)
        levels = [0] * (1+len(self.hierarchical_columns))
        idx = 0

        while queue:
            node = queue.popleft()

            # Assign node IDs using BFS order
            node.id = next_id

            # Copy contingency vector into corresponding shared memory row
            arr[node.id, :] = node.contingency_vector[:]
            del node.contingency_vector
            nodes.append(node)

            if node.level == idx and levels[idx] == 0:
                levels[idx] = node.id
                idx += 1 

            for child in node.children:
                child.parent_id = node.id
                queue.append(child)

            next_id += 1
        return nodes, levels, arr, shm

    def build_hierarchical_tree(self, constraints: dict[int, List[Constraint]]) -> HierarchicalTree:
        '''Build a hierarchical tree based on the hierarchical columns.
        It creates a contingency vector for each node in the tree.
        
        Args:
            constraints (Dict[int, List[Callable]]): Dictionary mapping tree levels to their constraints.
        
        Returns:
            HierarchicalTree: The constructed hierarchical tree.
        '''

        tree = HierarchicalTree()
        root = tree.nodes[0]
        curr_level = 0

        # Generate the contingency table if not already done
        if self.contingency_df is None:
            self.generate_contingency_dataframe(self.query_columns)
        
        assert self.dataframe is not None, "Dataframe is not loaded. Call read_data first."
        # Contingency vector for the root node (entire dataset)
        root.contingency_vector = self.create_contingency_vector(self.dataframe)

        # List of constraints for the root node
        root_contstraints = []
        if constraints and curr_level in constraints:
            # Iterate over the constraints for the root node
            for constraint in constraints[curr_level]:
                # Case when the constraint is a ContextualAggregateConstraint and needs to compute its value
                match constraint:
                    case ContextualAggregateConstraint():
                        constraint.apply_aggregation_function(self.dataframe)
                # Append the constraint function to the root constraints list
                assert self.contingency_df is not None, "Contingency DataFrame is not generated. Call generate_contingency_table first."
                root_contstraints.append(constraint.to_constraint(self.contingency_df))

        root.constraints = root_contstraints

        # Construct the tree recursively and count the nodes created
        # Then change the representation to array and create a share memory space
        tree._node_count = self._build_subtree(root, curr_level, self.dataframe, constraints)
        tree.nodes, tree._levels, tree._contingency_vectors, tree._contingency_vectors_shm = self.convert_tree_representation(root, tree._node_count)
        return tree
    
    def _build_subtree(self, parent_node: HierarchicalNode, curr_level: int, data: pd.DataFrame, constraints: dict[int, List[Constraint]]) -> None:
        '''Helper method to recursively build the subtree for a given parent node.
        
        Args:
            parent_node (HierarchicalNode): The parent node to which children will be added.
            curr_level (int): An iterator for the current level in the hierarchy. It has an offset of 1.
            data (pd.DataFrame): The subset of data corresponding to the parent node.
            constraints (List[Callable]): List of constraints to apply to each node.
        '''
        n_nodes = 1

        # When there are no more levels to process, return the parent node
        if curr_level >= len(self.hierarchical_columns):
            return n_nodes
        
        # Get the current hierarchical column to split on
        current_column = self.hierarchical_columns[curr_level]
        unique_hierarchical_values = data[current_column].unique()

        for value in unique_hierarchical_values:
            # Filter data for the current hierarchical value
            filtered_data = data[data[current_column] == value]

            # Prepare constraints for the current level
            level_constraints = []
            if constraints and curr_level in constraints:
                # Iterate over the constraints for the current level
                for constraint in constraints[curr_level]:
                    # Case when the constraint is a ContextualAggregateConstraint and needs to compute its value
                    match constraint:
                        case ContextualAggregateConstraint():
                            constraint.apply_aggregation_function(filtered_data)
                    # Append the constraint function to the level constraints list
                    assert self.contingency_df is not None, "Contingency DataFrame is not generated. Call generate_contingency_table first."
                    level_constraints.append(constraint.to_constraint(self.contingency_df))

            # Create a new child node
            child_node = HierarchicalNode(geo_id=value, level=curr_level+1, constraints=level_constraints)
            parent_node.add_child(child_node)

            # Create and assign the contingency vector for the child node
            child_node.contingency_vector = self.create_contingency_vector(filtered_data)

            # Recursively build the subtree for the child node
            n_nodes += self._build_subtree(child_node, curr_level + 1, filtered_data, constraints)
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
            contingency_vector = tree._contingency_vectors[leaf.id]
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

