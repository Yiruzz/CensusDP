from collections import deque
from dataclasses import fields

import pandas as pd
import numpy as np

from itertools import product
from typing import List, Dict, Tuple, Optional

from constraints.constraint import Constraint
from constraints.contextual_constraints import ContextualAggregateConstraint
from hierarchical_tree import HierarchicalTree
from hierarchical_node import HierarchicalNode

class DataHandler:
    '''Class to handle data loading, preprocessing and postprocessing.'''

    def __init__(self, input_path: str, hierarchical_columns: List[str], query_columns: List[str],
                 output_tree: str, output_path: str) -> None:
        '''Constructror for DataHandler class.
        
        Args:
            input_path (str): Path to the input data file.
            hierarchical_columns (List[str]): List of columns representing the hierarchical levels, in order from highest to lowest.
            query_columns (List[str]): List of columns to use for generating the contingency table.
            output_tree (str): Path to the file containing the hierarchical tree structure and the contingency vectors or save them.
            output_path (str): Path to save the processed data.

        Attributes:
            data_path (str): Path to the data file.
            microdata_path (str): Path to save the processed microdata.

            tree_folder (str): Path to the file containing the hierarchical tree structure and the contingency vectors or save them.
            tree_file (str): Path to the file containing the hierarchical tree.
            raw_contingency_vectors_file (Tuple[Optional[str], bool]): Tuple that represents the file containing the raw contingency vectors and a boolean value indicating whether it is necessary to save it or not.
                                                                       We assume that if we have the filename, it means the file exists and can be loaded.       
            noisy_contingency_vectors_file (Tuple[Optional[str], bool]): Tuple that represents the file containing the noisy contingency vectors and a boolean value indicating whether it is necessary to save it or not.
                                                                         We assume that if we have the filename, it means the file exists and can be loaded.
            
            dataframe (Optional[pd.DataFrame]): DataFrame to hold the data.
            hierarchical_columns (List[str]): List of columns representing the hierarchical levels.
            query_columns (List[str]): List of columns to use for generating the contingency table.

            contingency_df (Optional[pd.DataFrame]): DataFrame use to store and have an order on each unique combination of attributes.  
        '''
        # Input and output paths
        self.data_path: str = input_path
        self.microdata_path: str = output_path

        # Paths related to the tree structure and the contingency vectors.
        self.tree_folder: str = output_tree
        self.tree_file: Optional[str] = None
        self.raw_contingency_vectors_file: Tuple[Optional[str], bool] = (None, False)        
        self.noisy_contingency_vectors_file: Tuple[Optional[str], bool] = (None, False)

        # DataFrame to hold the data (loaded from file_path).
        self.dataframe: Optional[pd.DataFrame] = None
        self.hierarchical_columns: List[str] = hierarchical_columns
        self.query_columns: List[str] = query_columns

        # The contingency vectors will have the same order of this dataframe.
        self.contingency_df: Optional[pd.DataFrame] = None

    def read_data(self, sep: str = ',') -> None:
        '''Read data from the file_path into a pandas DataFrame.
        
        Args:
            file_path (str): Path to the data file.
        '''
        self.dataframe = pd.read_csv(self.data_path, sep=sep)
        return None
    
    def reduce_data(self) -> None:
        '''Reduce the dataframe to the necessary columns.

        Args:
            columns_to_use (List[str]): List of columns to keep in the dataframe.
        '''
        self.dataframe = self.dataframe[self.hierarchical_columns + self.query_columns]
        return None
    
    def sort_data_by_hierarchy(self) -> None:
        '''Sort the data by the hierarchical columns in ascending order.
        '''
        self.dataframe.sort_values(by=self.hierarchical_columns, inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)
        return None

    def write_data(self, data: pd.DataFrame, out_path: Optional[str] = None) -> None:
        '''Write the processed data to the output_path.
        
        Args:
            data (pd.DataFrame): DataFrame containing the processed data to write.
            out_path (Optional[str]): Optional path to save the processed data. If None, use self.output_path.
        '''
        data.to_csv((out_path or self.microdata_path), sep=';', index=False, encoding='utf-8')
        return None
    
    def generate_contingency_dataframe(self) -> pd.DataFrame:
        '''Generate a contingency dataframe from the loaded data.
        This method assumes that each column have all the possible values.

        Returns:
            pd.DataFrame: Contingency DataFrame with each unique combination of query columns.
        '''
        # Get unique values for each column
        # NOTE: Here we assume that each column contains all possible values
        # Example: if the domain of "Age" is [0, ..., 100], we assume that the column contains all those values
        unique_values = [self.dataframe[col].unique() for col in self.query_columns]

        # Generate all possible combinations (Cartesian product)
        self.contingency_df = pd.DataFrame(list(product(*unique_values)), columns=self.query_columns)
        
        # Sort by the columns to ensure a consistent order
        self.contingency_df.sort_values(by=self.query_columns, inplace=True)
        self.contingency_df.reset_index(drop=True, inplace=True)

        print(" Contingency DataFrame generated with shape:", self.contingency_df.shape)

        return self.contingency_df
    
    def create_contingency_vector(self, df: pd.DataFrame) -> np.ndarray:
        '''Create a contingency vector from the given dataframe with the same order as contingency_df.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data to aggregate.

        Returns:
            np.ndarray: Contingency vector with counts for each unique combination in contingency_df.
        '''
        queries = self.contingency_df.columns.tolist()

        # Group the data by the permutation columns and count occurrences
        grouped = df.value_counts(subset=queries).reset_index(name='frequency')

        # Merge to get frequencies for all combinations, ensuring correct order and filling missing with 0.
        # This merged dataframe now contains all combinations from self.contingency_df and their counts (0 if not present in df).
        merged = pd.merge(self.contingency_df, grouped, how='left', on=queries).fillna({'frequency': 0})

        # Directly extract the 'frequency' column as a numpy array.
        contingency_vector = merged['frequency'].to_numpy(dtype=int)
        
        return contingency_vector
    
    def create_contingency_vectors(self, tree: HierarchicalTree) -> None:
        '''Create contingency vectors for each node in the tree based on the corresponding segment of the data for that node.

        Args: 
            tree (HierarchicalTree): The hierarchical tree to which the contingency vectors will be added.
                                     The contingency vector for each node will be generated based on the segment of the data corresponding to that node,
                                     which is determined by the df_range attribute of the node.
        '''
        for node in tree.nodes:
            # Get the segment of the data corresponding to the current node
            start, end = node.df_range
            df_node = self.dataframe.iloc[start:end+1]

            # Create the contingency vector for the current node
            node.contingency_vector = self.create_contingency_vector(df_node)
            
        return None
    
    def add_constraints(self, tree: HierarchicalTree, constraints: Dict[int, List[Constraint]]) -> None:
        '''Add constraints to the nodes of the tree based on the specified levels.
        
        Args: 
            tree (HierarchicalTree): The hierarchical tree to which the constraints will be added.
            constraints (Dict[int, List[Constraint]]): A dictionary where the keys are the levels of the tree and the values are lists of constraints to apply to those levels.        
        '''
        # For each level in the constraints dictionary
        for level in constraints.keys():

            # Get the range of nodes corresponding to that level in the tree
            range = tree.node_ranges_by_level[level]

            # For each node in that range, add the corresponding constraints 
            for node in tree.nodes[range[0]:range[1]+1]:
                level_constraints = []
                for constraint in constraints[level]:
                    # Case when the constraint is a ContextualAggregateConstraint and needs to compute its value
                    match constraint:
                        case ContextualAggregateConstraint():
                            start, end = node.df_range
                            constraint.apply_aggregation_function(self.dataframe.iloc[start:end+1])
                    level_constraints.append(constraint.to_constraint(self.contingency_df))
                node.constraints.extend(level_constraints) 
        return None
    
    def load_contingency_vectors(self, tree: HierarchicalTree, filename: str) -> None:
        '''Load the contingency vectors from the file, mapping each row index to the corresponding node index.

        Args:
            tree (HierarchicalTree): The hierarchical tree to which the contingency vectors will be added.
            filename (str): Name of the CSV file (located in `self.tree_folder`) containing the contingency vectors.
        '''
        df = pd.read_csv(f"{self.tree_folder}/{filename}", sep=";")
        vectores = df.values.astype(int)
    
        index = 0
        for node in tree.nodes:
            node.contingency_vector = vectores[index]
            index += 1
        return None

    def load_tree(self) -> HierarchicalTree:
        '''Load the hierarchical tree structure based on a file created by the function create_tree.
        For each row, create a node with the corresponding information and put it in the right place in the tree.

        Returns:
            HierarchicalTree: The loaded hierarchical tree structure.
        '''
        nodes = []
        node_ranges_by_level = []

        index_last_hierarchical_column = len(self.hierarchical_columns)

        # Read the tree file into a DataFrame
        tree_df = pd.read_csv(f"{self.tree_folder}/{self.tree_file}", sep=';')

        curr_level = 0

        # Iterate over the rows of the DataFrame and create nodes
        for _, row in tree_df.iterrows():
            node_id = int(row['id'])
            parent_id = int(row['parent']) if not pd.isna(row['parent']) else None
            geo_value = int(row['geo'])
            level = int(row['level'])

            # If we have reached a new level in the tree,
            # update the node_ranges_by_level list because finished a level.
            if level > curr_level:
                node_ranges_by_level.append((None, node_id))
                curr_level = level

            # Stop loading nodes if we have reached a level greater
            # than the last hierarchical column index specified.
            # Since the file was created in order, all subsequent nodes
            # will have a level greater than this, so we can skip them.
            if level > index_last_hierarchical_column:
                break

            df_range = tuple(map(int, row['df_range'].strip('()').split(',')))
            children_range = tuple(map(int, row['children_range'].strip('()').split(','))) if not pd.isna(row['children_range']) else None

            if level == index_last_hierarchical_column:
                children_range= None

            node = HierarchicalNode(node_id=node_id, parent_id=parent_id, geo_value=geo_value,
                                    level=level, df_range=df_range, children_range=children_range)
            nodes.append(node)

        # Complete the node_ranges_by_level list
        node_ranges_by_level[0] = (0, node_ranges_by_level[0][1])
        for i in range(len(node_ranges_by_level) - 1):
            node_ranges_by_level[i+1] = (node_ranges_by_level[i][1], node_ranges_by_level[i+1][1])

        if len(node_ranges_by_level) < index_last_hierarchical_column + 1:
            # If we reach the end of all levels in the tree without breaking
            # due to the level condition, we need to add the remaining level.
            node_ranges_by_level.append((node_ranges_by_level[-1][1], len(nodes)))

        return HierarchicalTree(nodes, node_ranges_by_level)

    def create_tree(self) -> pd.DataFrame:
        '''Create a tree based on the hierarchical columns.
        Using the sorted DataFrame, it gets the indices for each node related to the DataFrame.
        This allows saving the tree structure to a file, which can later be loaded with "load_tree" and
        create different contingency vectors dividing the work efficiently, because the data for each node is independent.

        Returns:
            df (pd.DataFrame): Tree DataFrame to save.
        '''
        # Inialize the list of nodes 
        nodes = []

        # Get rows relationated to the hierarchical columns as a numpy array
        data_geo = self.dataframe[self.hierarchical_columns].values             
        rows = len(self.dataframe)

        # Create the root node
        curr_id = 0
        curr_level = 0
        root_node = HierarchicalNode(node_id=curr_id, parent_id=None, geo_value=curr_id,
                                     level=curr_level, df_range=(0, rows-1), children_range=None)
        
        nodes.append(root_node)

        curr_id += 1
        curr_level += 1

        # Start BFS
        q = deque()
        q.append(root_node)

        while q:
            node = q.popleft()
            child_level = node.level + 1

            # If the child level is greater than the number of hierarchical columns,
            # we have reached the leaf nodes, so we stop creating children since the level
            if child_level > len(self.hierarchical_columns):
                break

            # Take the segment of the data corresponding to the current hierarchical level
            # Find the indices where the value changes, which will correspond to the children nodes. 
            # This is possible because the data is sorted by the hierarchical columns, so each node corresponds to a contiguous block of rows in the DataFrame,
            # and each child node corresponds to a contiguous block of rows within that block, where the value of the current hierarchical column is constant
            start, end = node.df_range
            segment = data_geo[start:end+1, child_level - 1] 
            changes = np.where(segment[:-1] != segment[1:])[0]
            indices = np.concatenate(([start], changes + start + 1, [end + 1]))

            # Prepare the range of the children nodes for the parent node
            start_child_range = curr_id

            # For each child 
            for i in range(len(indices) - 1):
                # Create a node
                start_index, end_index = indices[i], indices[i + 1] - 1
                geo_value = data_geo[start_index, child_level - 1] 
                child = HierarchicalNode(node_id=curr_id, parent_id=node.id, geo_value=geo_value,
                                         level=child_level, df_range=(start_index, end_index), children_range=None)

                # Append the child node to the tree and to the queue for BFS
                nodes.append(child)
                q.append(child)
                curr_id += 1
        
            # Complete the children range for the parent node
            end_child_range = curr_id - 1
            node.children_range = (start_child_range, end_child_range)

        # Define the fields to save for each node in the tree
        fields = ['id', 'parent', 'geo', 'level', 'df_range', 'children_range']

        # Convert the list of nodes to a list of dictionaries with the specified fields
        data = []
        for node in nodes:
            node_dict = {
                'id': node.id,
                'parent': node.parent if node.parent is not None else '',
                'geo' : node.geo,
                'level': node.level,
                'df_range': (int(node.df_range[0]), int(node.df_range[1])),
                'children_range': (int(node.children_range[0]), int(node.children_range[1])) if node.children_range else ''
            }
            data.append(node_dict)

        # Create the DataFrame
        df = pd.DataFrame(data, columns=fields)

        return df
    
    def get_contingency_vectors(self, tree: HierarchicalTree) -> pd.DataFrame:
        '''Extracts the contingency vectors from all nodes in a hierarchical tree.
        
        Args:
            tree (HierarchicalTree): A hierarchical tree structure containing nodes with contingency vectors.

        Returns:
            pd.DataFrame: A DataFrame with one column named "contingency_vector", where each row corresponds to the contingency vector of a node in the tree.
        '''
        vectors_list = []
        for node in tree.nodes:
            vectors_list.append(node.contingency_vector)
        df = pd.DataFrame(vectors_list, columns=[f"v{i}" for i in range(len(vectors_list[0]))])
        return df
  
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
        for leaf in tree.nodes[tree.node_ranges_by_level[-1][0]:]:
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
            curr_level = len(self.hierarchical_columns) - 1
            curr_node = leaf

            # We do not include the root node in the hierarchical path as it is redundant in the final
            # data because the root is the highest level of the hierarchy and all data belongs to it
            while not curr_node.is_root():            
                geo_value = curr_node.id
                leaf_dict[self.hierarchical_columns[curr_level]] = list(np.repeat(geo_value, leaf_size))
                curr_node = tree.nodes[curr_node.parent]
                curr_level -= 1

            # Merge the leaf_dict into microdata_dict by concatenating lists for duplicate keys
            for key, values in leaf_dict.items():
                microdata_dict[key].extend(values)

        return pd.DataFrame(microdata_dict)

