import os
import tempfile
import shutil
import pandas as pd
import numpy as np
import scipy.sparse as sp

from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

from constraints.constraint import Constraint
from constraints.contextual_constraints import ContextualAggregateConstraint
from domain import ContingencyDomain
from hierarchical_tree import HierarchicalTree
from hierarchical_node import HierarchicalNode

class DataHandler:
    '''Class to handle data loading, preprocessing and postprocessing.'''

    def __init__(self, file_path: str, output_path: str = 'noisy_data.csv',
                 domain: Optional[Dict[str, Sequence]] = None) -> None:
        '''Constructror for DataHandler class.

        Args:
            file_path (str): Path to the data file.
            output_path (str): Path to save the processed data.
            domain (Optional[Dict[str, Sequence]]): Per-column set of all possible
                values, defining the contingency cell space. This should be
                data-independent for a sound DP guarantee (so valid-but-absent
                values still get a cell + noise). When None, or for any query column
                omitted, the domain is inferred from the observed data
                (np.sort(unique)) with a warning.

        Attributes:
            file_path (str): Path to the data file.
            output_path (str): Path to save the processed data. Defaults to "noisy_data.csv".

            dataframe (Optional[pd.DataFrame]): DataFrame to hold the data.
            contingency_domain (Optional[ContingencyDomain]): Mixed-radix cell space
                that replaces the dense Cartesian-product contingency table.
            contingency_df_length: Optional[int]: Number of contingency cells (n_cells).
            dtype (str): NumPy data type used for all arrays.

            query_columns (List[str]): List of columns to use for generating the contingency table.
            hierarchical_columns (List[str]): List of columns representing the hierarchical levels.
            spill_dir (str): Directory where node vectors are spilled to disk during estimation.
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

        # Mixed-radix cell space; the contingency vectors are indexed by its flat
        # cell index. Replaces the dense Cartesian-product DataFrame.
        self.domain: Optional[Dict[str, Sequence]] = domain
        self.contingency_domain: Optional[ContingencyDomain] = None
        self.contingency_df_length: Optional[int] = None
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

    def build_contingency_domain(self, query_columns: List[str]) -> ContingencyDomain:
        '''Build the mixed-radix contingency domain for the query columns.

        Replaces the dense Cartesian-product DataFrame: the cell space is described
        structurally by per-column sorted value sets + strides, so no (k x n_cells)
        table is materialised. Per-column values come from the user-declared
        self.domain when available.

        Args:
            query_columns (List[str]): List of query columns to aggregate.

        Returns:
            ContingencyDomain: The constructed cell space.
        '''
        assert self.dataframe is not None, "Dataframe is not loaded. Call read_data first."

        self.contingency_domain = ContingencyDomain.build(
            columns=query_columns, data=self.dataframe, declared=self.domain
        )
        self.contingency_df_length = self.contingency_domain.n_cells

        print("Contingency domain built with n_cells:", self.contingency_domain.n_cells, "in", end=' ')

        return self.contingency_domain

    def create_contingency_vector(self, df: pd.DataFrame) -> sp.csr_matrix:
        '''Create a sparse contingency (column) vector for the given records.

        Uses the domain's mixed-radix encoder to map each record to its flat cell
        index and scatters the counts — replacing the pandas value_counts + merge
        against a full Cartesian table. Returned sparse so the raw histogram x is
        never densified before Q @ x.

        Args:
            df (pd.DataFrame): DataFrame containing the records to aggregate.

        Returns:
            scipy.sparse.csr_matrix: Column vector of shape (n_cells, 1) with the
                count for each contingency cell.
        '''
        if self.contingency_domain is None:
            raise ValueError("Contingency domain is not built. Call build_contingency_domain first.")

        domain = self.contingency_domain
        flat_indices = domain.encode(df)
        nz, counts = np.unique(flat_indices, return_counts=True)
        return sp.csr_matrix(
            (counts, (nz, np.zeros(len(nz), dtype=np.int64))),
            shape=(domain.n_cells, 1),
            dtype=self.dtype,
        )

    def _query_answers(self, query_matrix, records: pd.DataFrame) -> np.ndarray:
        '''Compute the noiseless query answers y = Q @ x for a set of records.

        Builds the sparse raw histogram x for records and applies the (sparse or
        dense) query matrix, returning a dense 1-D integer array of length n_queries.

        Args:
            query_matrix: Query matrix Q (scipy sparse CSR or dense ndarray),
                shape (n_queries, n_cells).
            records (pd.DataFrame): The node's records.

        Returns:
            np.ndarray: Dense length-n_queries answers.
        '''
        x = self.create_contingency_vector(records)  # sparse (n_cells, 1)
        if sp.issparse(query_matrix):
            y = np.asarray((query_matrix @ x).todense()).ravel()
        else:
            y = query_matrix @ x.toarray().ravel()
        # Q is binary and x integer, so y is integer-valued; cast exactly.
        return y.astype(self.dtype)

    def build_hierarchical_tree(self) -> HierarchicalTree:
        '''Build a hierarchical tree structure based on hierarchical columns.

        This creates only the tree structure using hierarchical_path for each node.
        Contingency vectors, constraints, and query matrix information are NOT handled here;
        they are materialized on demand during the estimation phase.

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

        # Assign unique incremental IDs to all nodes via BFS
        tree._index_nodes()

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
        assert self.contingency_domain is not None, "Contingency domain is not built. Call build_contingency_domain first."
        domain = self.contingency_domain

        if not node.is_leaf():
            raise ValueError("Node must be a leaf node.")

        if node.contingency_vector is None:
            raise ValueError("Node contingency vector is not materialized.")

        # After estimation, the node's vector holds the estimated cell counts x_hat
        # (length n_cells). Select only positive frequencies.
        contingency_vector = node.contingency_vector
        nonzero_idx = np.flatnonzero(contingency_vector > 0)

        # Decode only the nonzero cells into their attribute combinations,
        # avoiding any full (n_cells x k) combination table.
        filtered_query_values = domain.decode(nonzero_idx)
        filtered_counts = contingency_vector[nonzero_idx]

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

        Filters data once based on hierarchical path, then creates the (sparse-backed)
        measurement vector y = Q @ x and prepares all constraints for the node against
        the mixed-radix contingency domain.

        Args:
            hierarchical_path (List[int]): The node's hierarchical path for filtering.
            constraints (List[Constraint]): Constraints for the node considering its level.
            query_matrix (np.ndarray): Query matrix for aggregating contingency vectors.

        Returns:
            Tuple[np.ndarray, List[Constraint]]: Contingency vector and constraint callables for this node.
        '''
        if self.contingency_domain is None:
            raise ValueError("Contingency domain is not built. Call build_contingency_domain first.")

        # Filter data once based on hierarchical path
        filtered_df = self.dataframe
        if len(hierarchical_path) > 1:
            for level_idx, value in enumerate(hierarchical_path[1:]):
                column = self.hierarchical_columns[level_idx]
                filtered_df = filtered_df[filtered_df[column] == value]

        # Build the measurement vector y = Q @ x from the sparse histogram of the
        # filtered records (raw cell counts x are never densified).
        contingency_vector = self._query_answers(query_matrix, filtered_df)

        # Prepare constraints using the same filtered data, targeting the sparse domain.
        level_constraints = []
        for constraint in constraints:
            # Apply aggregation for constraints that compute dynamically
            match constraint:
                case ContextualAggregateConstraint():
                    constraint.apply_aggregation_function(filtered_df)

            # Convert to optimizer callable against the contingency domain
            level_constraints.append(constraint.to_constraint(self.contingency_domain))

        return contingency_vector, level_constraints
