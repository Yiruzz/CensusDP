import pandas as pd
import numpy as np
import scipy.sparse as sp
from collections import deque

from multiprocessing import shared_memory
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
    
    def convert_tree_representation(self, root: HierarchicalNode, n_nodes: int,
                                    n_queries: int, vector_length: int) -> Tuple[List[HierarchicalNode], List[int], np.ndarray, memoryview]:
        '''Retrieve the references of all nodes of the tree to facilitate access.
        Also create a shared memory space with all contingency vectors copied,
        enabling multiprocessing without duplicating data.

        Each row is sized to vector_length = max(n_queries, n_cells) so the same slot holds
        both the noisy measurement y (length n_queries, written here) and later the estimated
        cell counts x_hat (length n_cells, written by the estimation phase). The unused
        trailing slots are zero-initialized and ignored until estimation overwrites them.

        Args:
            root (HierarchicalNode): A hierarchical tree.
            n_nodes (int): Number of nodes, corresponding to the number of rows in the shared memory space.
            n_queries (int): Length of the measurement vector y currently held by each node.
            vector_length (int): Physical row length = max(n_queries, n_cells).

        Return:
            tuple[list[HierarchicalNode], List[int], np.ndarray, memoryview]: A tuple containing all node references, the node index where each level starts,
                                                                              the view over the shared memory buffer, and the memoryview used to access the shared memory.
        '''
        shape = (n_nodes, vector_length)
        size = int(np.prod(shape) * np.dtype(self.dtype).itemsize)

        # Create share memory space
        shm = shared_memory.SharedMemory(create=True, size=size)

        # View over the shared buffer. shared_memory contents are uninitialized, so zero the
        # whole array — the trailing slots beyond n_queries must start at 0, since the
        # measurement phase only writes the first n_queries entries.
        arr = np.ndarray(shape, dtype=self.dtype, buffer=shm.buf)
        arr.fill(0)

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

            # Copy the node's noisy-measurement-sized vector into the first n_queries slots.
            arr[node.id, :n_queries] = node.contingency_vector[:]
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

    def build_hierarchical_tree(self, constraints: dict[int, List[Constraint]], query_matrix: np.ndarray) -> HierarchicalTree:
        '''Build a hierarchical tree based on the hierarchical columns.
        It creates a contingency vector for each node in the tree.

        Each node stores y = Q @ x (shape (n_queries,)) — the noiseless query answers.
        The measurement phase adds noise in place; the estimation phase later overwrites
        this slot with the estimated cell-counts x_hat (shape (n_cells,)). Keeping a
        single slot avoids retaining both x and y simultaneously.

        Args:
            constraints (Dict[int, List[Callable]]): Dictionary mapping tree levels to their constraints.
            query_matrix (np.ndarray): Query matrix Q of shape (n_queries, n_cells), applied to each
                node's raw cell counts so the slot holds Q @ x instead of x.

        Returns:
            HierarchicalTree: The constructed hierarchical tree.
        '''

        tree = HierarchicalTree()
        root = tree.nodes[0]
        curr_level = 0

        # Build the contingency domain if not already done
        if self.contingency_domain is None:
            self.build_contingency_domain(self.query_columns)

        assert self.dataframe is not None, "Dataframe is not loaded. Call read_data first."
        # Query answers for the root node (entire dataset). Raw x is discarded after the multiply.
        root.contingency_vector = self._query_answers(query_matrix, self.dataframe)

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
                assert self.contingency_domain is not None, "Contingency domain is not built. Call build_contingency_domain first."
                root_contstraints.append(constraint.to_constraint(self.contingency_domain))

        root.constraints = root_contstraints

        # Construct the tree recursively and count the nodes created
        # Then change the representation to array and create a share memory space
        tree._node_count = self._build_subtree(root, 0, self.dataframe, constraints, query_matrix)

        # Single-buffer sizing: rows of max(n_queries, n_cells) hold both y (n_queries) before
        # estimation and x_hat (n_cells) after, so the same allocation serves both phases.
        n_queries, n_cells = query_matrix.shape
        tree.n_queries = n_queries
        tree.n_cells = n_cells
        tree.vector_length = max(n_queries, n_cells)

        tree.nodes, tree._levels, tree._contingency_vectors, tree._contingency_vectors_shm = self.convert_tree_representation(
            root, tree._node_count, n_queries, tree.vector_length
        )
        return tree

    def _build_subtree(self, parent_node: HierarchicalNode, level_iterator: int, data: pd.DataFrame, constraints: dict[int, List[Constraint]], query_matrix: np.ndarray) -> int:
        '''Helper method to recursively build the subtree for a given parent node.

        Args:
            parent_node (HierarchicalNode): The parent node to which children will be added.
            level_iterator (int): An iterator for the current level in the hierarchy. It has an offset of 1.
            data (pd.DataFrame): The subset of data corresponding to the parent node.
            constraints (List[Callable]): List of constraints to apply to each node.
            query_matrix (np.ndarray): Query matrix Q applied to each child's raw cell counts.

        Returns:
            int: The number of nodes in the subtree.
        '''
        n_nodes = 1

        # When there are no more levels to process, return the parent node
        if level_iterator >= len(self.hierarchical_columns):
            return n_nodes
        
        # Get the current hierarchical column to split on
        current_column = self.hierarchical_columns[level_iterator]
        unique_hierarchical_values = data[current_column].unique()

        for value in unique_hierarchical_values:
            # Filter data for the current hierarchical value
            filtered_data = data[data[current_column] == value]

            # Prepare constraints for the current level
            level_constraints = []
            if constraints and level_iterator in constraints:
                # Iterate over the constraints for the current level
                for constraint in constraints[level_iterator]:
                    # Case when the constraint is a ContextualAggregateConstraint and needs to compute its value
                    match constraint:
                        case ContextualAggregateConstraint():
                            constraint.apply_aggregation_function(filtered_data)
                    # Append the constraint function to the level constraints list
                    assert self.contingency_domain is not None, "Contingency domain is not built. Call build_contingency_domain first."
                    level_constraints.append(constraint.to_constraint(self.contingency_domain))

            # Create a new child node
            child_node = HierarchicalNode(geo_id=value, level=level_iterator+1, constraints=level_constraints)
            parent_node.add_child(child_node)

            # Store query answers Q @ x; raw cell counts x are not retained.
            child_node.contingency_vector = self._query_answers(query_matrix, filtered_data)

            # Recursively build the subtree for the child node
            n_nodes += self._build_subtree(child_node, level_iterator + 1, filtered_data, constraints, query_matrix)

        return n_nodes
    
    # NOTE: This method will not work if the query matrix contains any workload that is not a simple count of the contingency cells.
    # TODO: Make a more complete version of construction of output data, it does not need to be microdata.
    #       For example it can be just aggregate data, where each count have the information of the query that produces it:
    #       (df['Sex'] == 'Male') & (df['Age'] == 30) -> 10
    #       (df['Sex'] == 'Female') & (df['Age'] == 30) -> 15
    #       ... and so on on the other queries.
    def construct_microdata(self, tree: HierarchicalTree) -> pd.DataFrame:
        '''Construct microdata from the hierarchical tree.
        
        This method traverses the hierarchical tree and reconstructs the microdata
        based on the contingency vectors at each node.

        Args:
            tree (HierarchicalTree): The hierarchical tree containing contingency vectors.

        Returns:
            pd.DataFrame: The reconstructed microdata.
        '''
        
        assert self.contingency_domain is not None, ("Contingency domain is not built. Call build_contingency_domain first.")
        domain = self.contingency_domain

        # Store partial DataFrames generated for each leaf.
        microdata_parts = []

        start_node_idx = tree._levels[-1]
        for leaf in tree.nodes[start_node_idx:]:
            # Generate rows associated with query values.
            # Select only positive frequencies.
            # After estimation, only the first n_cells slots of the row hold the x_hat values;
            # the trailing slots (if vector_length > n_cells) are unused padding.
            contingency_vector = tree._contingency_vectors[leaf.id][:tree.n_cells]
            nonzero_idx = np.flatnonzero(contingency_vector > 0)

            # Decode only the nonzero cells into their attribute combinations,
            # avoiding any full (n_cells x k) combination table.
            filtered_query_values = domain.decode(nonzero_idx)
            filtered_counts = contingency_vector[nonzero_idx]

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

