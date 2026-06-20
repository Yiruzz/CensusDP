import os
import shutil
import warnings
import pandas as pd
import numpy as np
import scipy.sparse as sp
import duckdb
import zarr
import numcodecs
import itertools
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from constraints.constraint import Constraint
from constraints.contextual_constraints import ContextualAggregateConstraint
from domain import ContingencyDomain
from parallel_utils.noise_generation import initialize_mechanism, generate_noise_row
from hierarchical_tree import HierarchicalTree
from hierarchical_node import HierarchicalNode

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from pathlib import Path

class DataHandler:
    '''Class to handle data loading, preprocessing and postprocessing.'''

    def __init__(self, file_path: Optional[str] = None, output_path: Optional[str] = None, domain: Optional[Dict[str, Sequence]] = None) -> None:
        '''Constructor for DataHandler class.

        Args:
            file_path (Optional[str]): Path to the data file. Can be CSV or Parquet.
            output_path (Optional[str]): Path to save the processed noisy data.
            domain (Optional[Dict[str, Sequence]]): Per-column set of all possible values, defining the contingency cell space.
                Should be data-independent for a sound DP guarantee. When None or for any query column omitted,
                the domain is inferred from the observed data (SELECT DISTINCT) with a warning.

        Attributes:
            file_path (Optional[str]): Path to the input data file.
            output_path (Optional[str]): Path where the noisy output CSV will be saved.

            domain (Optional[Dict[str, Sequence]]): User-provided domain for query columns.
            contingency_domain (Optional[ContingencyDomain]): Mixed-radix cell space that replaces dense Cartesian-product table.
            contingency_df_length (Optional[int]): Total number of contingency cells (n_cells).
            dtype (str): NumPy data type for all arrays (default: 'int64').

            hierarchical_columns (List[str]): Columns defining the tree hierarchy levels.
            query_columns (List[str]): Columns for generating the contingency table.

            spill_dir (Optional[str]): Temporary directory for spilled node vectors. Set by initialize_directories().
            microdata_dir (Optional[str]): Temporary directory for worker microdata files. Set by initialize_directories().
            worker_microdata_file (Optional[str]): Path to the current worker's microdata file.

            duckdb_con (Optional[duckdb.DuckDBPyConnection]): DuckDB connection for queries.
            data_view_name (str): Name of the DuckDB view for the input data.
        '''
        # Input and output paths
        self.file_path: Optional[str] = file_path
        self.output_path: Optional[str] = output_path

        # Validate output file path
        if output_path is not None:
            path_obj = Path(output_path)
            if not (path_obj.parent.exists() and path_obj.parent.is_dir()):
                print(f"Warning: Output path {output_path} is not valid. 'noisy_data.csv' will be saved in the current directory instead.")
                self.output_path = 'noisy_data.csv'

        # Mixed-radix cell space; the contingency vectors are indexed by its flat
        # cell index. Replaces the dense Cartesian-product DataFrame.
        self.domain: Optional[Dict[str, Sequence]] = domain
        self.contingency_domain: Optional[ContingencyDomain] = None
        self.contingency_df_length: Optional[int] = None
        self.dtype: str = 'int32'

        # Columns to use
        self.hierarchical_columns: List[str] = []
        self.query_columns: List[str] = []

        # Define path to directories for vectors and temporary microdata files
        self.spill_dir: Optional[str] = None
        self.noisy_dir: Optional[str] = None
        self.microdata_dir: Optional[str] = None
        self.worker_microdata_file: Optional[str] = None

        # DuckDB connection for queries
        self.duckdb_con: Optional[duckdb.DuckDBPyConnection] = None
        self.data_view_name: str = 'data'

        self.cache_dir: str = os.path.join(os.getcwd(), 'data/data_cache')

        self.noise_zarr_group: Optional[zarr.hierarchy.Group] = None
        self.noisy_array_name = "Noise"
        self.noise_zarr_path: Optional[str] = None
        self.COMPRESSION_LEVEL = 9
      
        os.makedirs(self.cache_dir, exist_ok=True)

    def initialize_directories(self) -> None:
        '''Create directories for spilled vectors and temporary microdata in project root.'''
        pid = os.getpid()
        self.noisy_dir = os.path.join(self.cache_dir, f'topdown_noisy')
        self.spill_dir = os.path.join(self.cache_dir, f'topdown_spill_{pid}')
        self.microdata_dir = os.path.join(self.cache_dir, f'topdown_microdata_{pid}')

        os.makedirs(self.noisy_dir, exist_ok=True)
        os.makedirs(self.spill_dir, exist_ok=True)
        os.makedirs(self.microdata_dir, exist_ok=True)

    def cleanup_directories(self) -> None:
        '''Delete spill and microdata directories recursively.'''
        if self.spill_dir and os.path.isdir(self.spill_dir): shutil.rmtree(self.spill_dir, ignore_errors=True)
        if self.microdata_dir and os.path.isdir(self.microdata_dir): shutil.rmtree(self.microdata_dir, ignore_errors=True)

    def convert_csv_to_parquet(self) -> None:
        '''Convert CSV file to Parquet format using DuckDB.

        Converts the input CSV to Parquet and stores it in the same directory
        with the same filename but .parquet extension. Updates file_path to
        point to the Parquet file.
        '''

        # Create path to a new .parquet file
        path_obj = Path(self.file_path)
        parquet_path = path_obj.with_suffix('.parquet')

        # Skip creation if file already exists
        if parquet_path.exists():
            print(f'\n Parquet file already exists: \n  {parquet_path}')
            self.file_path = str(parquet_path)
            return

        # Create connection using all available threads
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

        # Set connection for data handler using one thread
        self.duckdb_con = duckdb.connect(config={'threads': 1})

        # Combine all columns for SELECT clause
        all_cols = self.hierarchical_columns + self.query_columns
        cols_str = ', '.join(all_cols)

        # Create a unique view for this connection
        self.duckdb_con.execute(f"""
            CREATE VIEW {self.data_view_name} AS
            SELECT {cols_str}
            FROM read_parquet('{self.file_path}')
        """)

    def build_contingency_domain(self) -> None:
        '''Build the mixed-radix contingency domain for the query columns.

        Replaces the dense Cartesian-product DataFrame: the cell space is described
        structurally by per-column sorted value sets + strides, so no (k x n_cells)
        table is materialised. Per-column values come from the user-declared
        self.domain when available.

        '''
        assert self.duckdb_con is not None, "DuckDB connection not initialized. Call create_data_view first."

        # Build declared domain by querying missing columns from DuckDB
        declared = dict(self.domain) if self.domain else {}

        # Check whether a domain is defined for each column.
        # If not, the domain is inferred from the column values.
        for col in self.query_columns:
            if col not in declared:
                warnings.warn(
                    f"No domain declared for column '{col}'; inferring it from the data "
                    f"(SELECT DISTINCT). This is data-dependent (not DP-safe) and may "
                    f"omit valid-but-absent values. Pass domain={{'{col}': [...]}} to fix.",
                    stacklevel=2,
                )
                # Query unique values from DuckDB view
                query = f"SELECT DISTINCT {col} FROM {self.data_view_name} ORDER BY {col}"
                result = self.duckdb_con.execute(query).fetchall()
                declared[col] = np.array([row[0] for row in result])

        self.contingency_domain = ContingencyDomain(columns=self.query_columns, domains=declared)
        self.contingency_df_length = self.contingency_domain.n_cells

        print("\n Contingency domain built with n_cells:", self.contingency_domain.n_cells, "in", end=' ')

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
        tree._node_count = self._build_subtree(root, 0, root.filter_dict)
        tree._levels = 1+len(self.hierarchical_columns)

        # Assign unique incremental IDs to all nodes via BFS traversal
        tree._index_nodes()

        return tree
    
    def _build_subtree(self, parent_node: HierarchicalNode, level_iterator: int, filter_dict: Dict[str, Any]) -> int:
        '''Helper method to recursively build the subtree for a given parent node.

        Creates only the tree structure using DuckDB queries and filter conditions.

        Args:
            parent_node (HierarchicalNode): The parent node to which children will be added
            level_iterator (int): Current level in the hierarchy
            filter_dict (Dict[str, Any]): Dictionary mapping column names to values for WHERE clause

        Returns:
            int: The number of nodes in the subtree
        '''
        n_nodes = 1

        if level_iterator >= len(self.hierarchical_columns):
            return n_nodes

        current_column = self.hierarchical_columns[level_iterator]

        # Build WHERE clause from filter_dict
        conditions = [f"{col} = '{val}'" for col, val in filter_dict.items()]
        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""

        # Query unique values for current level using DuckDB
        query = f"SELECT DISTINCT {current_column} FROM {self.data_view_name} {where_clause} ORDER BY {current_column}"
        result = self.duckdb_con.execute(query).fetchall()
        unique_hierarchical_values = [row[0] for row in result]

        # Create nodes for the current hierarchy level
        for value in unique_hierarchical_values:
            # Use the filter_dict passed
            # adding the value related to this partition for this level
            new_filter_dict = filter_dict.copy()
            new_filter_dict[current_column] = value

            child_node = HierarchicalNode(level=level_iterator+1, filter_dict=new_filter_dict)
            parent_node.add_child(child_node)

            n_nodes += self._build_subtree(child_node, level_iterator + 1, new_filter_dict)

        return n_nodes
    
    def _reduce_dataframe(self, filters: Dict[str, Any]) -> pd.DataFrame:
        '''Query contingency table with optional filters from DuckDB.

        Executes a SQL query to compute the contingency table grouped by query columns
        with counts. Applies optional filters from filter_dict.

        Args:
            filters (Dict[str, Any]): Dictionary mapping column names to values for filtering

        Returns:
            pd.DataFrame: DataFrame with query columns and "count" column
        '''
        assert self.duckdb_con is not None, "DuckDB connection not initialized. Call create_data_view first."

        # Build and combine filter conditions.
        # Filter the data to the subset represented by the node.
        conditions = [f'"{col}" = {repr(val)}' for col, val in filters.items()]
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cols_sql = ", ".join(f'"{c}"' for c in self.query_columns)

        # Filter the data and compute counts for each combination of values.
        # This creates a compact, aggregated dataset.
        query = f"""
            SELECT {cols_sql}, COUNT(*) AS count
            FROM {self.data_view_name}
            {where}
            GROUP BY {cols_sql}
        """
        result = self.duckdb_con.execute(query).fetchall()

        # Return only the columns associated with the queries and counts.
        # Hierarchical columns are not used.
        col_names = self.query_columns + ["count"]
        return pd.DataFrame(result, columns=col_names)
    
    def _create_contingency_vector(self, filters: Optional[Dict[str, Any]] = None) -> sp.csr_matrix:
        '''Create a sparse contingency (column) vector for records matching filters.

        Queries the contingency table based on filters, encodes cell indices using the
        contingency domain, and builds a sparse column vector. Returned sparse to avoid
        densification before matrix multiplication (Q @ x).

        Args:
            filters (Optional[Dict[str, Any]]): Dictionary mapping column names to values for filtering. If None, uses all records

        Returns:
            sp.csr_matrix: Sparse vector of shape (n_cells, 1) with cell counts
        '''
        assert self.contingency_domain is not None, "Contingency domain is not built. Call build_contingency_domain first."

        data = self._reduce_dataframe(filters)
        flat_indices = self.contingency_domain.encode(data)
        counts = data["count"].values.astype(self.dtype)

        sparse_vector = sp.csr_matrix(
            (counts, (flat_indices, np.zeros(len(flat_indices), dtype=self.dtype))),
            shape=(self.contingency_domain.n_cells, 1),
            dtype=self.dtype,
        )

        return sparse_vector
    
    def materialize_node_data(self, filter_dict: Dict[str, Any], constraints: List[Constraint], query_matrix: Union[sp.csr_matrix, np.ndarray]) -> Tuple[np.ndarray, List]:
        '''Materialize contingency vector and prepare constraints in a single pass.

        Queries the contingency table based on filter_dict, then creates the (sparse-backed)
        measurement vector y = Q @ x and prepares all constraints for the node against
        the mixed-radix contingency domain.

        Args:
            filter_dict (Dict[str, Any]): The node's filter conditions (column -> value mapping)
            constraints (List[Constraint]): Constraints for the node considering its level
            query_matrix (Union[sp.csr_matrix, np.ndarray]): Query matrix for aggregating contingency vectors

        Returns:
            Tuple[np.ndarray, List]: Contingency vector and constraint callables for this node
        '''
        assert self.contingency_domain is not None, "Contingency domain is not built. Call build_contingency_domain first."

        # Build the measurement vector y = Q @ x from the sparse histogram using DuckDB query
        x = self._create_contingency_vector(filter_dict)  # sparse (n_cells, 1)

        if sp.issparse(query_matrix):
            y = np.asarray((query_matrix @ x).todense()).ravel()
        else:
            y = query_matrix @ x.toarray().ravel()

        contingency_vector = y.astype(self.dtype)

        # Generate the constraints corresponding to the node's level.
        level_constraints = []
        for constraint in constraints:
            match constraint:
                case ContextualAggregateConstraint():
                    constraint.apply_aggregation_function(x.data)

            # Convert to optimizer callable against the contingency domain
            level_constraints.append(constraint.to_constraint(self.contingency_domain))

        return contingency_vector, level_constraints

    def spill_path(self, filter_dict: Dict[str, Any]) -> str:
        '''Get the spill file path from a filter dictionary.

        Args:
            filter_dict (Dict[str, Any]): The filter dictionary (column -> value mapping)

        Returns:
            str: File path for the spilled vector named by the filter values
        '''

        # Create filename
        if filter_dict: name = '_'.join(f"{k}={v}" for k, v in filter_dict.items())
        else: name = "root"

        # Avoid problematic characters
        for ch in ('/', '\\', ' ', ':'):
            name = name.replace(ch, '_')
        
        # Build file path
        return os.path.join(self.spill_dir, name + '.npy')

    def spill_vector(self, path: str, contingency_vector: np.ndarray) -> None:
        '''Write contingency vector to disk and free it from RAM.

        One file per node, named by its hierarchical path, ensuring sibling nodes don't conflict.

        Args:
            path (str): The file path where the vector will be spilled
            contingency_vector (np.ndarray): The contingency vector to write to disk
        '''
        os.makedirs(self.spill_dir, exist_ok=True)
        np.save(path, np.ascontiguousarray(contingency_vector))

    def load_vector(self, path: str) -> np.ndarray:
        '''Reload contingency vector from disk and delete the file.

        Args:
            path (str): The file path to load the vector from

        Returns:
            np.ndarray: The loaded contingency vector
        '''
        contingency_vector = np.load(path)
        os.remove(path)
        return contingency_vector

    def update_child_vectors(self, joint_solution: np.ndarray, vectors_length: int, filter_dicts: List[Dict[str, Any]]) -> None:
        '''Split joint solution into individual child vectors and spill to disk.

        Args:
            joint_solution (np.ndarray): The combined solution vector for all children
            vectors_length (int): The length of each individual child vector
            filter_dicts (List[Dict[str, Any]]): List of filter dictionaries for each child
        '''
        start = 0
        for child_filter_dict in filter_dicts:
            end = start + vectors_length
            path = self.spill_path(child_filter_dict)
            self.spill_vector(path, joint_solution[start:end])
            start = end

    def merge_microdata_files(self) -> None:
        '''Merge Parquet microdata files into the output CSV using DuckDB.'''
        parquet_pattern = os.path.join(self.microdata_dir, '*.parquet')

        try:
            self.duckdb_con.execute(f"""
                COPY (SELECT * FROM read_parquet('{parquet_pattern}'))
                TO '{self.output_path}'
                (FORMAT CSV, HEADER TRUE, DELIMITER ';')
            """)
        except Exception as e:
            print(f"Warning: Error merging microdata files: {e}")

    def _construct_microdata_for_leaf(self, contingency_vector: np.ndarray, filter_dict: Dict[str, Any]) -> pd.DataFrame:
        '''Construct microdata for a leaf node.

        Args:
            contingency_vector (np.ndarray): The contingency vector (cell counts)
            filter_dict (Dict[str, Any]): Filter dictionary for hierarchical values

        Returns:
            pd.DataFrame: Microdata for this leaf node
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

    def write_microdata(self, node_id: int, contingency_vectors: List[np.ndarray], filter_dicts: List[Dict[str, Any]]) -> List[str]:
        '''Construct microdata for each child and write to separate Parquet files.

        For each contingency vector, reconstructs the original microdata by decoding cells
        according to the contingency domain and repeating rows by frequency. Results are
        written to individual Parquet files for later merging.

        Args:
            node_id (int): The parent node ID (used for naming the output files)
            contingency_vectors (List[np.ndarray]): List of contingency vectors (cell counts) for each child
            filter_dicts (List[Dict[str, Any]]): List of filter dictionaries for each child (hierarchical values)

        Returns:
            List[str]: Paths to the created Parquet files
        '''
        output_paths = []

        for i, (contingency_vector, filter_dict) in enumerate(zip(contingency_vectors, filter_dicts)):
            df = self._construct_microdata_for_leaf(contingency_vector, filter_dict)
            output_path = os.path.join(self.microdata_dir, f'node_{node_id}_child_{i}_microdata.parquet')
            df.to_parquet(output_path, index=False)
            output_paths.append(output_path)

        return output_paths

    def noisy_vectors_exist(self, n_nodes: int, mech_param_spec: str) -> bool:
        '''Check if compatible pre-computed noise vectors exist and reuse them if possible.

        Searches for Zarr files matching the mechanism and parameters. Selects the first
        compatible file with n_nodes >= requested and n_cells >= requested. Requires exact
        match on mechanism and parameters; compatible files can have more nodes/cells.

        Args:
            n_nodes (int): Requested number of nodes
            mech_param_spec (str): Mechanism parameter specification string (e.g., 'Laplace_1.0')

        Returns:
            bool: True if compatible vectors were found and loaded, False if new file created
        '''
        n_cells = self.contingency_df_length

        # Try to find a compatible existing file
        compatible_file = self._find_compatible_noisy_vector(n_nodes, n_cells, mech_param_spec)
        if compatible_file:
            print(f"\n Reusing compatible noise vectors file: {os.path.basename(compatible_file)}")
            self.noise_zarr_path = compatible_file
            self.noise_zarr_group = zarr.open_group(self.noise_zarr_path, mode="r")
            return True

        # No compatible file found; create a new one
        zarr_filename = f"noisy_vectors_{n_nodes}_{n_cells}_{mech_param_spec}.zarr"
        self.noise_zarr_path = os.path.join(self.noisy_dir, zarr_filename)
        self._create_noisy_file(self.noise_zarr_path, n_nodes)
        return False

    def _find_compatible_noisy_vector(self, n_nodes: int, n_cells: int, mech_param_spec: str) -> Optional[str]:
        '''Search for a compatible pre-computed noise vector file.

        Looks for files matching the pattern noisy_vectors_*_{mech_param_spec}.zarr
        and selects one with n_nodes_file >= n_nodes and n_cells_file >= n_cells.
        If multiple compatible files exist, returns the one with the smallest dimensions
        to minimize memory usage.

        Args:
            n_nodes (int): Required number of nodes
            n_cells (int): Required number of cells
            mech_param_spec (str): Exact mechanism and parameters to match

        Returns:
            Optional[str]: Path to a compatible file, or None if none exist
        '''
        if not os.path.isdir(self.noisy_dir):
            return None

        candidates = []
        pattern = f"noisy_vectors_"

        for filename in os.listdir(self.noisy_dir):
            if not filename.startswith(pattern) or not filename.endswith(".zarr"):
                continue

            # Parse filename: noisy_vectors_{n_nodes}_{n_cells}_{mech_param_spec}.zarr
            # Remove prefix and suffix
            name_without_ext = filename[len(pattern):-5]  # Remove "noisy_vectors_" and ".zarr"
       
            # Split by '_' but the mech_param_spec can contain underscores
            # Strategy: split from the right to extract mech_param_spec first
            parts = name_without_ext.split('_', 2)  # Split from right, max 2 splits
            if len(parts) != 3:
                continue
            try:
                file_n_nodes = int(parts[0])
                file_n_cells = int(parts[1])
                file_mech_spec = parts[2]
            except ValueError:
                continue

            # Check if mechanism matches
            if file_mech_spec != mech_param_spec:
                continue

            # Check if file has enough capacity
            if file_n_nodes >= n_nodes and file_n_cells >= n_cells:
                candidates.append((filename, file_n_nodes, file_n_cells))

        if not candidates:
            return None

        # Return the file with smallest number of nodes
        candidates.sort(key=lambda x: x[2])
        chosen_file = candidates[0][0]
        return os.path.join(self.noisy_dir, chosen_file)
            
    def _create_noisy_file(self, zarr_path: str, n_nodes: int) -> None:
        '''Create a new Zarr file to store pre-computed noise vectors.

        Initializes a Zarr group with zstd compression and creates the noise array
        with appropriate metadata and chunking (1 row per chunk for efficient row-wise writes).
        Each row stores a noise vector for one node.

        Args:
            zarr_path (str): Path where the Zarr file will be created
            n_nodes (int): Number of nodes (rows in the noise array)
        '''
        compressor = numcodecs.Blosc(cname="zstd", clevel=self.COMPRESSION_LEVEL, shuffle=numcodecs.Blosc.BITSHUFFLE)

        self.noise_zarr_group = zarr.open_group(zarr_path, mode="w", zarr_format=2)
        self.noise_zarr_group.create_array(self.noisy_array_name, shape=(n_nodes, self.contingency_df_length),
                                      chunks=(1, self.contingency_df_length), dtype=self.dtype, compressor=compressor)

        self.noise_zarr_group.attrs["n_expected_nodes"] = n_nodes
        self.noise_zarr_group.attrs["n_cells"] = self.contingency_df_length
        self.noise_zarr_group.attrs["n_rows_generated"] = 0

    def generate_noise_vectors(self, n_workers: int, n_nodes: int, gen: Iterable[Tuple[int, int]],
                               privacy_mech_name: str, level_params: List[float], query_sensitivity: int) -> None:
        '''Generate pre-computed noise vectors in parallel and write to Zarr file.

        Uses a sliding window with ProcessPoolExecutor to maintain constant worker load.
        Workers generate noise independently; the main thread writes results to avoid
        Zarr concurrency issues.

        Args:
            n_workers (int): Number of worker processes in the pool
            n_nodes (int): Total number of nodes (for progress reporting)
            gen (Iterable[Tuple[int, int]]): Iterator of (node_id, level) tuples to process
            privacy_mech_name (str): Name of the privacy mechanism to use
            level_params (List[float]): Parameters for the privacy mechanism per level
        '''
        WINDOW_SIZE = 5000
        REFILL_BATCH = 2000

        pending = {}
        done = 0
        task_iter = iter(gen)
        arr = self.noise_zarr_group[self.noisy_array_name]

        with ProcessPoolExecutor(max_workers=n_workers, initializer=initialize_mechanism,
                                 initargs=(privacy_mech_name, level_params,
                                           self.contingency_df_length, self.dtype, query_sensitivity)) as executor:

            for node_id, level in itertools.islice(task_iter, WINDOW_SIZE):
                fut = executor.submit(generate_noise_row, level)
                pending[fut] = node_id

            completed_since_refill = 0
            while pending:
                finished, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)

                for fut in finished:
                    node_id = pending.pop(fut)
                    
                    try:
                        row_data = fut.result()
                        arr[node_id, :] = row_data
                        done += 1

                        if done % max(1, n_nodes // 20) == 0:
                            print(f"    Progress: {done}/{n_nodes}")
                    except Exception as e:
                        print(f"    Error on row {node_id}: {e}")

                completed_since_refill += len(finished)
                if completed_since_refill >= REFILL_BATCH:
                    new_tasks = list(itertools.islice(task_iter, completed_since_refill))
                    for node_id, level in new_tasks:
                        fut = executor.submit(generate_noise_row, level)
                        pending[fut] = node_id

                    completed_since_refill = 0
        self.noise_zarr_group.attrs["n_rows_generated"] = done


