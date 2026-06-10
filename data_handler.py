import os
import tempfile
import shutil
import warnings
import pandas as pd
import numpy as np
import scipy.sparse as sp
import duckdb

from constraints.constraint import Constraint
from constraints.contextual_constraints import ContextualAggregateConstraint
from domain import ContingencyDomain
from hierarchical_tree import HierarchicalTree
from hierarchical_node import HierarchicalNode

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path

class DataHandler:
    '''Class to handle data loading, preprocessing and postprocessing.'''

    def __init__(self, file_path: Optional[str] = None, output_path: str = 'noisy_data.csv', domain: Optional[Dict[str, Sequence]] = None) -> None:
        '''Constructor for DataHandler class.

        Args:
            file_path (Optional[str]): Path to the data file.
            output_path (str): Path to save the processed data.
            domain (Optional[Dict[str, Sequence]]): Per-column set of all possible
                values, defining the contingency cell space. This should be
                data-independent for a sound DP guarantee (so valid-but-absent
                values still get a cell + noise). When None, or for any query column
                omitted, the domain is inferred from the observed data
                (np.sort(unique)) with a warning.

        Attributes:
            file_path (Optional[str]): Path to the data file.
            output_path (str): Path to save the processed data. Defaults to "noisy_data.csv".

            dataframe (Optional[pd.DataFrame]): DataFrame to hold the data.
            contingency_domain (Optional[ContingencyDomain]): Mixed-radix cell space that replaces the dense Cartesian-product contingency table.
            contingency_df_length (Optional[int]): Number of contingency cells (n_cells).
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

        # Directory for temporary microdata files
        self.microdata_dir: Optional[str] = None

        # Worker-specific microdata file path
        self.worker_microdata_file: Optional[str] = None

        # DuckDB connection for queries
        self.duckdb_con: Optional[duckdb.DuckDBPyConnection] = None

    def initialize_output_file(self) -> None:
        '''Initialize the output CSV file with column headers.'''
        # Create empty DataFrame with the expected columns
        empty_df = pd.DataFrame(columns=self.hierarchical_columns + self.query_columns)
        empty_df.to_csv(self.output_path, index=False, header=True)

    def initialize_microdata_dir(self) -> None:
        '''Initialize the temporary directory for worker microdata files.'''
        self.microdata_dir = os.path.join(tempfile.gettempdir(), f'topdown_microdata_{os.getpid()}')
        os.makedirs(self.microdata_dir, exist_ok=True)

    def cleanup_microdata_dir(self) -> None:
        '''Delete the temporary microdata directory.'''
        if self.microdata_dir and os.path.isdir(self.microdata_dir):
            shutil.rmtree(self.microdata_dir, ignore_errors=True)

    def convert_csv_to_parquet(self) -> None:
        '''Convert CSV file to Parquet format using DuckDB.

        Converts the input CSV to Parquet and stores it in the same directory
        with the same filename but .parquet extension. Updates file_path to
        point to the Parquet file.
        '''
        if self.file_path is None:
            return

        path_obj = Path(self.file_path)
        if path_obj.suffix.lower() != '.csv':
            return

        parquet_path = path_obj.with_suffix('.parquet')

        if parquet_path.exists():
            print(f'\n Parquet file already exists: \n  {parquet_path}')
            self.file_path = str(parquet_path)
            return

        con = duckdb.connect()
        con.execute(f"""
            COPY (
                SELECT *
                FROM read_csv_auto('{self.file_path}')
            )
            TO '{parquet_path}'
            (FORMAT PARQUET);
        """)
        con.close()
        self.file_path = str(parquet_path)

    def create_data_view(self) -> None:
        '''Create a DuckDB view for querying the data.

        Creates a view named 'data' that reads from the Parquet file and includes
        all hierarchical and query columns. Sets up the DuckDB connection for use
        in tree construction queries. Always uses threads=1.
        '''
        self.duckdb_con = duckdb.connect(config={'threads': 1})

        all_cols = self.hierarchical_columns + self.query_columns
        cols_str = ', '.join(all_cols)

        self.duckdb_con.execute(f"""
            CREATE OR REPLACE VIEW data AS
            SELECT {cols_str}
            FROM read_parquet('{self.file_path}')
        """)

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
        assert self.duckdb_con is not None, "DuckDB connection not initialized. Call create_data_view first."

        # Build declared domain by querying missing columns from DuckDB
        declared = dict(self.domain) if self.domain else {}

        for col in query_columns:
            if col not in declared:
                warnings.warn(
                    f"No domain declared for column '{col}'; inferring it from the data "
                    f"(SELECT DISTINCT). This is data-dependent (not DP-safe) and may "
                    f"omit valid-but-absent values. Pass domain={{'{col}': [...]}} to fix.",
                    stacklevel=2,
                )
                # Query unique values from DuckDB view
                query = f"SELECT DISTINCT {col} FROM data ORDER BY {col}"
                result = self.duckdb_con.execute(query).fetchall()
                declared[col] = np.array([row[0] for row in result])

        self.contingency_domain = ContingencyDomain(columns=query_columns, domains=declared)
        self.contingency_df_length = self.contingency_domain.n_cells

        print("\n Contingency domain built with n_cells:", self.contingency_domain.n_cells, "in", end=' ')

        return self.contingency_domain

    def _reduce_dataframe(self, filters: Dict[str, Any]) -> pd.DataFrame:
        '''Query contingency table with optional filters from DuckDB.

        Executes a SQL query to compute the contingency table grouped by query columns
        with counts. Applies optional filters from filter_dict.

        Args:
            filters (Dict[str, Any]): Dictionary mapping column names to values for filtering.

        Returns:
            pd.DataFrame: DataFrame with query columns and "count" column.
        '''
        if self.duckdb_con is None:
            raise ValueError("DuckDB connection not initialized. Call create_data_view first.")

        where = ""

        if filters:
            conditions = [f'"{col}" = {repr(val)}' for col, val in filters.items()]
            where = "WHERE " + " AND ".join(conditions)

        cols_sql = ", ".join(f'"{c}"' for c in self.query_columns)

        query = f"""
            SELECT {cols_sql}, COUNT(*) AS count
            FROM data
            {where}
            GROUP BY {cols_sql}
        """

        result = self.duckdb_con.execute(query).fetchall()
        col_names = self.query_columns + ["count"]
        return pd.DataFrame(result, columns=col_names)

    def _create_contingency_vector(self, filters: Optional[Dict[str, Any]] = None) -> Tuple[sp.csr_matrix, np.ndarray]:
        '''Create a sparse contingency (column) vector for records matching filters.

        Queries the contingency table using tabla_contingencia, encodes cell indices,
        and builds a sparse matrix. Returned sparse so the raw histogram x is
        never densified before Q @ x.

        Args:
            filters (Optional[Dict[str, Any]]): Dictionary mapping column names to values for filtering.

        Returns:
            Tuple[scipy.sparse.csr_matrix, np.ndarray]: Column vector of shape (n_cells, 1) with
                the count for each contingency cell, and the counts array (int64) for aggregation.
        '''
        if self.contingency_domain is None:
            raise ValueError("Contingency domain is not built. Call build_contingency_domain first.")

        data = self._reduce_dataframe(filters)
        flat_indices = self.contingency_domain.encode(data)
        counts = data["count"].values.astype(self.dtype)

        sparse_vector = sp.csr_matrix(
            (counts, (flat_indices, np.zeros(len(flat_indices), dtype=np.int64))),
            shape=(self.contingency_domain.n_cells, 1),
            dtype=self.dtype,
        )

        return sparse_vector, counts

    def build_hierarchical_tree(self) -> HierarchicalTree:
        '''Build a hierarchical tree structure based on hierarchical columns.

        This creates only the tree structure using hierarchical_path for each node.
        Contingency vectors, constraints, and query matrix information are NOT handled here;
        they are materialized on demand during the estimation phase.

        Returns:
            HierarchicalTree: The constructed hierarchical tree with hierarchical_path information.
        '''
        assert self.hierarchical_columns, "Hierarchical columns not set."
        assert self.duckdb_con is not None, "DuckDB connection not initialized. Call create_data_view first."

        tree = HierarchicalTree()
        root = tree.root

        # Build tree structure recursively using DuckDB queries
        tree._node_count = self._build_subtree(root, 0, None)
        tree._levels = 1+len(self.hierarchical_columns)

        return tree

    def _build_subtree(self, parent_node: HierarchicalNode, level_iterator: int, filter_dict: Optional[Dict[str, Any]] = None) -> int:
        '''Helper method to recursively build the subtree for a given parent node.

        Creates only the tree structure using DuckDB queries and filter conditions.

        Args:
            parent_node (HierarchicalNode): The parent node to which children will be added.
            level_iterator (int): An iterator for the current level in the hierarchy.
            filter_dict (Optional[Dict[str, Any]]): Dictionary mapping column names to values for WHERE clause.

        Returns:
            int: The number of nodes in the subtree.
        '''
        n_nodes = 1

        if level_iterator >= len(self.hierarchical_columns):
            return n_nodes

        current_column = self.hierarchical_columns[level_iterator]

        # Build WHERE clause from filter_dict
        where_clause = ""
        if filter_dict:
            conditions = [f"{col} = '{val}'" for col, val in filter_dict.items()]
            where_clause = " WHERE " + " AND ".join(conditions)

        # Query unique values for current level using DuckDB
        query = f"SELECT DISTINCT {current_column} FROM data {where_clause} ORDER BY {current_column}"
        result = self.duckdb_con.execute(query).fetchall()
        unique_hierarchical_values = [row[0] for row in result]

        for value in unique_hierarchical_values:
            child_node = HierarchicalNode(geo_id=value, level=level_iterator+1)
            parent_node.add_child(child_node)

            # Create and assign filter dict for child
            new_filter_dict = (filter_dict.copy() if filter_dict else {})
            new_filter_dict[current_column] = value
            child_node.filter_dict = new_filter_dict
            n_nodes += self._build_subtree(child_node, level_iterator + 1, new_filter_dict)

        return n_nodes
    
    def spill_path(self, filter_dict: Dict[str, Any]) -> str:
        '''Get the spill file path from a filter dictionary.

        Args:
            filter_dict (Dict[str, Any]): The filter dictionary (column -> value mapping).

        Returns:
            str: File path for the spilled vector named by the filter values.
        '''
        if filter_dict:
            name = '_'.join(f"{k}={v}" for k, v in filter_dict.items())
        else:
            name = "root"
        for ch in ('/', '\\', ' ', ':'):
            name = name.replace(ch, '_')
        return os.path.join(self.spill_dir, name + '.npy')

    def spill_vector(self, path: str, contingency_vector: np.ndarray) -> None:
        '''Write contingency vector to disk and free it from RAM.

        One file per node, named by its hierarchical_path, ensuring sibling nodes don't conflict.

        Args:
            path (str): The file path where the vector will be spilled.
            contingency_vector (np.ndarray): The contingency vector to write to disk.
        '''
        os.makedirs(self.spill_dir, exist_ok=True)
        np.save(path, np.ascontiguousarray(contingency_vector))

    def load_vector(self, path: str) -> np.ndarray:
        '''Reload contingency vector from disk and delete the file.

        Args:
            path (str): The file path to load the vector from.

        Returns:
            np.ndarray: The loaded contingency vector.
        '''
        contingency_vector = np.load(path)
        os.remove(path)
        return contingency_vector

    def update_child_vectors(self, joint_solution: np.ndarray, vectors_length: int, filter_dicts: List[Dict[str, Any]]) -> None:
        '''Split joint solution into individual child vectors and spill to disk.

        Args:
            joint_solution (np.ndarray): The combined solution vector for all children.
            vectors_length (int): The length of each individual child vector.
            filter_dicts (List[Dict[str, Any]]): List of filter dictionaries for each child.
        '''
        start = 0
        for filter_dict in filter_dicts:
            end = start + vectors_length
            path = self.spill_path(filter_dict)
            self.spill_vector(path, joint_solution[start:end])
            start = end

    def cleanup_spill(self) -> None:
        '''Delete the spill directory and all remaining spilled vectors.'''
        if os.path.isdir(self.spill_dir):
            shutil.rmtree(self.spill_dir, ignore_errors=True)

    def merge_microdata_files(self) -> None:
        '''Merge microdata files from workers into the output CSV.'''
        worker_files = [
            os.path.join(self.microdata_dir, f) for f in os.listdir(self.microdata_dir)
            if f.startswith('worker_') and f.endswith('.csv')
        ]

        for worker_file in worker_files:
            try:
                with open(worker_file, 'rb') as src:
                    with open(self.output_path, 'ab') as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                os.remove(worker_file)
            except Exception as e:
                print(f"Warning: Error merging microdata file {worker_file}: {e}")

    def _construct_microdata_for_leaf(self, contingency_vector: np.ndarray, filter_dict: Dict[str, Any]) -> pd.DataFrame:
        '''Construct microdata for a leaf node.

        Args:
            contingency_vector (np.ndarray): The contingency vector (cell counts).
            filter_dict (Dict[str, Any]): Filter dictionary for hierarchical values.

        Returns:
            pd.DataFrame: Microdata for this leaf node.
        '''
        assert self.contingency_domain is not None, "Contingency domain is not built. Call build_contingency_domain first."
        domain = self.contingency_domain

        # Select only positive frequencies
        nonzero_idx = np.flatnonzero(contingency_vector > 0)

        # Decode only the nonzero cells into their attribute combinations
        filtered_query_values = domain.decode(nonzero_idx)
        filtered_counts = contingency_vector[nonzero_idx]

        # Repeat each combination according to its frequency
        expanded_rows = np.repeat(filtered_query_values, filtered_counts, axis=0)

        # Create DataFrame for the leaf
        leaf_df = pd.DataFrame(expanded_rows, columns=self.query_columns)

        # Generate columns associated with hierarchical values from filter_dict
        for column_name, value in filter_dict.items():
            leaf_df[column_name] = value

        # Reorder columns to match output file order: hierarchical + query
        output_columns = self.hierarchical_columns + self.query_columns
        return leaf_df[output_columns]

    def materialize_node_data(self, filter_dict: Dict[str, Any], constraints: List[Constraint], query_matrix: Union[sp.csr_matrix, np.ndarray]) -> Tuple[np.ndarray, List]:
        '''Materialize contingency vector and prepare constraints in a single pass.

        Queries the contingency table based on filter_dict, then creates the (sparse-backed)
        measurement vector y = Q @ x and prepares all constraints for the node against
        the mixed-radix contingency domain.

        Args:
            filter_dict (Dict[str, Any]): The node's filter conditions (column -> value mapping).
            constraints (List[Constraint]): Constraints for the node considering its level.
            query_matrix (Union[sp.csr_matrix, np.ndarray]): Query matrix for aggregating contingency vectors.

        Returns:
            Tuple[np.ndarray, List[Constraint]]: Contingency vector and constraint callables for this node.
        '''
        if self.contingency_domain is None:
            raise ValueError("Contingency domain is not built. Call build_contingency_domain first.")

        # Build the measurement vector y = Q @ x from the sparse histogram using DuckDB query
        x, counts = self._create_contingency_vector(filter_dict)  # sparse (n_cells, 1), counts array

        if sp.issparse(query_matrix):
            y = np.asarray((query_matrix @ x).todense()).ravel()
        else:
            y = query_matrix @ x.toarray().ravel()

        contingency_vector = y.astype(self.dtype)

        # Prepare constraints for the node (placeholder for constraint implementation)
        level_constraints = []
        for constraint in constraints:
            # Apply aggregation for constraints that compute dynamically
            match constraint:
                case ContextualAggregateConstraint():
                    constraint.apply_aggregation_function(counts)

            # Convert to optimizer callable against the contingency domain
            level_constraints.append(constraint.to_constraint(self.contingency_domain))

        return contingency_vector, level_constraints
