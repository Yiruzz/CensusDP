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
from constraints.sparse_constraint import SparseConstraint
from domain import ContingencyDomain
from graph import JunctionTree
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
            n_cells (Optional[int]): Total number of contingency cells.
            query_width (Optional[int]): Length of a full-joint node measurement Q @ x. It drives noise_width in the joint pipeline. Set by TopDown.
            dtype (str): NumPy data type for all arrays (default: 'int64').

            junction_tree (Optional[JunctionTree]): Factored representation of the cell space.
                When set, the pipeline measures one marginal per bag instead of the full joint.
            bag_domains (List[ContingencyDomain]): Sub-domain of each bag, aligned to junction_tree.bags.
            bag_offsets (List[int]): Start index of each bag inside the node's concatenated marginal vector.
            marginal_width (Optional[int]): Length of that concatenated vector (= sum of bag n_cells).

            hierarchical_columns (List[str]): Columns defining the tree hierarchy levels.
            query_columns (List[str]): Columns for generating the contingency table.

            spill_dir (Optional[str]): Temporary directory for spilled node vectors. Set by initialize_directories().
            microdata_dir (Optional[str]): Temporary directory for worker microdata files. Set by initialize_directories().
            worker_microdata_file (Optional[str]): Path to the current worker's microdata file.

            noisy_dir (Optional[str]): Directory holding pre-computed noise Zarr files. Set by initialize_directories().
            noise_zarr_group (Optional[zarr.hierarchy.Group]): Zarr group of pre-computed noise vectors.
            noisy_array_name (str): Name of the noise array within the Zarr group.
            noise_zarr_path (Optional[str]): Path to the noise Zarr group on disk.

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
        self.n_cells: Optional[int] = None
        # Length of a full-joint node measurement y = Q @ x. Set by TopDown.
        self.query_width: Optional[int] = None
        self.dtype: str = 'int64'

        # Factored (junction-tree) cell space. Built by build_marginal_domains(),
        # left empty when running the full-joint pipeline.
        self.junction_tree: Optional[JunctionTree] = None
        self.bag_domains: List[ContingencyDomain] = []
        self.bag_offsets: List[int] = []
        self.marginal_width: Optional[int] = None
        self._separator_projections: Dict[Tuple[int, Tuple[str, ...]], np.ndarray] = {}

        # Non-contextual constraints depend only on the bag's ub-domain, never on the node, 
        # so they are compiled once per run. 
        # (id(constraint), bag) -> (constraint, indices, coefs, sense, rhs).
        self._compiled_constraints: Dict[Tuple[int, int], Tuple] = {}

        # Columns to use
        self.hierarchical_columns: List[str] = []
        self.query_columns: List[str] = []

        # Define path to directories for vectors and temporary microdata files
        self.spill_dir: Optional[str] = None
        self.microdata_dir: Optional[str] = None
        self.worker_microdata_file: Optional[str] = None
        self.lp_problems_dir: Optional[str] = None

        # Pre-computed noise storage (Zarr)
        self.noisy_dir: Optional[str] = None
        self.noise_zarr_group: Optional[zarr.hierarchy.Group] = None
        self.noisy_array_name: str = "Noise"
        self.noise_zarr_path: Optional[str] = None
        self.COMPRESSION_LEVEL: int = 9

        # DuckDB connection for queries
        self.duckdb_con: Optional[duckdb.DuckDBPyConnection] = None
        self.data_view_name: str = 'data'

    def initialize_directories(self) -> None:
        '''Create directories for spilled vectors and temporary microdata in project root.'''
        cache_dir = os.path.join(os.getcwd(), 'data', 'data_cache')

        pid = os.getpid()
        self.spill_dir = os.path.join(cache_dir, f'topdown_spill_{pid}')
        self.microdata_dir = os.path.join(cache_dir, f'topdown_microdata_{pid}')
        self.lp_problems_dir = os.path.join(cache_dir, 'topdown_solver_problems')
        # Shared across runs (no pid) so compatible noise files can be reused.
        self.noisy_dir = os.path.join(cache_dir, 'topdown_noisy')

        os.makedirs(self.spill_dir, exist_ok=True)
        os.makedirs(self.microdata_dir, exist_ok=True)
        os.makedirs(self.lp_problems_dir, exist_ok=True)
        os.makedirs(self.noisy_dir, exist_ok=True)

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
        self.n_cells = self.contingency_domain.n_cells

        print("\n Contingency domain built with n_cells:", self.contingency_domain.n_cells, "in", end=' ')

    # ------------------------------------------------------------------
    # Factored (junction-tree) cell space
    # ------------------------------------------------------------------

    def build_marginal_domains(self, junction_tree: JunctionTree) -> None:
        '''Bind a junction tree and derive the per-bag cell spaces.

        Each bag gets its own small ContingencyDomain (a subdomain of the global one, so
        per-column value ranks stay consistent). The bags are laid out back-to-back in a
        single per-node vector of length marginal_width; bag_offsets gives each block's
        start. That layout is what gets measured, noised and spilled — the global joint
        (n_cells) is never materialized.

        Args:
            junction_tree (JunctionTree): Tree whose bags are the marginals to measure.
        '''
        assert self.contingency_domain is not None, "Contingency domain is not built. Call build_contingency_domain first."

        self.junction_tree = junction_tree
        self.bag_domains = [self.contingency_domain.subdomain(bag) for bag in junction_tree.bags]

        offsets: List[int] = []
        total = 0
        for domain in self.bag_domains:
            offsets.append(total)
            total += domain.n_cells
        self.bag_offsets = offsets
        self.marginal_width = total
        self._separator_projections = {}
        self._compiled_constraints = {}

    def separator_projection(self, bag_index: int, columns: Sequence[str]) -> np.ndarray:
        '''Map each cell of a bag to its cell index in the separator sub-domain.

        Cached per (bag, separator) since the same maps are reused at every node for the
        separator-consistency constraints and the microdata reconstruction.

        Args:
            bag_index (int): Index of the bag in junction_tree.bags.
            columns (Sequence[str]): Separator columns (a subset of the bag's columns).

        Returns:
            np.ndarray: Length-(bag n_cells) array of separator cell indices.
        '''
        key = (bag_index, tuple(columns))
        projection = self._separator_projections.get(key)
        if projection is None:
            projection = self.bag_domains[bag_index].project_to(columns)
            self._separator_projections[key] = projection
        return projection

    @property
    def noise_width(self) -> int:
        '''Length of a node's measurement vector, which the pre-computed noise must match.

        Factored pipeline: the concatenated marginals (marginal_width). Full-joint pipeline:
        the query-space measurement y = Q @ x. n_cells when using identity Q.
        '''
        if self.marginal_width is not None:
            return self.marginal_width
        if self.query_width is not None:
            return self.query_width
        return self.n_cells

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
            parent_node (HierarchicalNode): The parent node to which children will be added.
            level_iterator (int): An iterator for the current level in the hierarchy.
            filter_dict (Dict[str, Any]): Dictionary mapping column names to values for WHERE clause.

        Returns:
            int: The number of nodes in the subtree.
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

    def _reduce_dataframe(self, filters: Dict[str, Any], con: Optional["duckdb.DuckDBPyConnection"] = None,
                          columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        '''Query contingency table with optional filters from DuckDB.

        Executes a SQL query to compute the contingency table grouped by the requested
        columns with counts. Applies optional filters from filter_dict.

        Args:
            filters (Dict[str, Any]): Dictionary mapping column names to values for filtering.
            con (Optional[duckdb.DuckDBPyConnection]): Per-thread cursor; defaults to the shared connection.
            columns (Optional[Sequence[str]]): Columns to group by. Defaults to all query
                columns (full joint); a junction-tree bag passes its own column subset so
                the aggregation stays small.

        Returns:
            pd.DataFrame: DataFrame with the grouped columns and a "count" column.
        '''
        # Allow callers (e.g. parallel in-memory materialization) to pass a per-thread
        # cursor; DuckDB connections are not safe to share concurrently across threads.
        con = con if con is not None else self.duckdb_con
        assert con is not None, "DuckDB connection not initialized. Call create_data_view first."

        # Build and combine filter conditions.
        # Filter the data to the subset represented by the node.
        columns = list(columns) if columns is not None else self.query_columns
        conditions = [f'"{col}" = {repr(val)}' for col, val in filters.items()]
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cols_sql = ", ".join(f'"{c}"' for c in columns)

        # Filter the data and compute counts for each combination of values.
        # This creates a compact, aggregated dataset.
        query = f"""
            SELECT {cols_sql}, COUNT(*) AS count
            FROM {self.data_view_name}
            {where}
            GROUP BY {cols_sql}
        """
        result = con.execute(query).fetchall()

        # Return only the columns associated with the queries and counts.
        # Hierarchical columns are not used.
        return pd.DataFrame(result, columns=columns + ["count"])

    def _create_contingency_vector(self, filters: Optional[Dict[str, Any]] = None, con: Optional["duckdb.DuckDBPyConnection"] = None) -> sp.csr_matrix:
        '''Create a sparse contingency (column) vector for records matching filters.

        Queries the contingency table using tabla_contingencia, encodes cell indices,
        and builds a sparse matrix. Returned sparse so the raw histogram x is
        never densified before Q @ x.

        Args:
            filters (Optional[Dict[str, Any]]): Dictionary mapping column names to values for filtering.

        Returns:
            scipy.sparse.csr_matrix:
        '''
        assert self.contingency_domain is not None, "Contingency domain is not built. Call build_contingency_domain first."

        data = self._reduce_dataframe(filters, con)
        flat_indices = self.contingency_domain.encode(data)
        counts = data["count"].values.astype(self.dtype)

        sparse_vector = sp.csr_matrix(
            (counts, (flat_indices, np.zeros(len(flat_indices), dtype=self.dtype))),
            shape=(self.contingency_domain.n_cells, 1),
            dtype=self.dtype,
        )

        return sparse_vector

    def materialize_node_data(self, filter_dict: Dict[str, Any], constraints: List[Constraint], query_matrix: Union[sp.csr_matrix, np.ndarray], con: Optional["duckdb.DuckDBPyConnection"] = None) -> Tuple[np.ndarray, List]:
        '''Materialize contingency vector and prepare constraints in a single pass.

        Queries the contingency table based on filter_dict, then creates the (sparse-backed)
        measurement vector y = Q @ x and prepares all constraints for the node against
        the mixed-radix contingency domain.

        Args:
            filter_dict (Dict[str, Any]): The node's filter conditions (column -> value mapping).
            constraints (List[Constraint]): Constraints for the node considering its level.
            query_matrix (Union[sp.csr_matrix, np.ndarray]): Query matrix for aggregating contingency vectors.
            con (Optional[duckdb.DuckDBPyConnection]): Per-thread DuckDB cursor for concurrent
                materialization. Defaults to the shared connection when None.

        Returns:
            Tuple[np.ndarray, List[Constraint]]: Contingency vector and constraint callables for this node.
        '''
        assert self.contingency_domain is not None, "Contingency domain is not built. Call build_contingency_domain first."

        # Build the measurement vector y = Q @ x from the sparse histogram using DuckDB query
        x = self._create_contingency_vector(filter_dict, con)  # sparse (n_cells, 1)

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
            level_constraints.append(constraint.to_sparse_constraint(self.contingency_domain))

        return contingency_vector, level_constraints

    def _create_marginal_vector(self, bag_index: int, filters: Dict[str, Any],
                                con: Optional["duckdb.DuckDBPyConnection"] = None) -> np.ndarray:
        '''Measure one bag's marginal for the records matching filters.

        A GROUP BY over the bag's columns only, encoded into the bag's sub-domain. The
        result is dense but small (the bag's n_cells), so no sparse container is needed.

        Args:
            bag_index (int): Index of the bag in junction_tree.bags.
            filters (Dict[str, Any]): The node's filter conditions (column -> value mapping).
            con (Optional[duckdb.DuckDBPyConnection]): Per-thread cursor for concurrent measurement.

        Returns:
            np.ndarray: Length-(bag n_cells) vector of counts.
        '''
        domain = self.bag_domains[bag_index]
        data = self._reduce_dataframe(filters, con, columns=domain.columns)

        # Encode the marginal counts into a dense vector of length bag.n_cells.
        counts = np.zeros(domain.n_cells, dtype=self.dtype)
        if len(data): 
            np.add.at(counts, domain.encode(data), data["count"].values.astype(self.dtype))
        return counts

    def measure_pairwise_counts(self, columns: Sequence[str],
                                con: Optional["duckdb.DuckDBPyConnection"] = None) -> Tuple[np.ndarray, List[Tuple[str, str]], List[int]]:
        '''Measure every 2-way marginal over the whole dataset, concatenated into one vector.

        Used by the private marginal-selection step: the association between columns has to
        be estimated from the data, and doing so costs privacy budget like any other query.
        Returning ONE vector matters - the caller noises it in a single shot, so the
        sensitivity argument is the number of pairs (a record falls in exactly one cell of
        each pair table), exactly as the number of bags is for the per-node measurement.

        Args:
            columns (Sequence[str]): Columns to pair up.
            con (Optional[duckdb.DuckDBPyConnection]): Cursor; defaults to the shared connection.

        Returns:
            Tuple[np.ndarray, List[Tuple[str, str]], List[int]]:
                the concatenated counts, the pairs in layout order (i < j), and the start
                offset of each pair's table inside the vector.
        '''
        assert self.contingency_domain is not None, "Contingency domain is not built. Call build_contingency_domain first."

        columns = list(columns)
        pairs = [(columns[i], columns[j])
                 for i in range(len(columns)) for j in range(i + 1, len(columns))]

        blocks: List[np.ndarray] = []
        offsets: List[int] = []
        total = 0
        for pair in pairs:
            pair_domain = self.contingency_domain.subdomain(pair)
            data = self._reduce_dataframe({}, con, columns=list(pair))

            counts = np.zeros(pair_domain.n_cells, dtype=self.dtype)
            if len(data): # Add the counts to the correct indices in the pair's contingency vector. Use np.add.at to handle duplicate indices correctly.
                np.add.at(counts, pair_domain.encode(data), data["count"].values.astype(self.dtype))

            blocks.append(counts)
            offsets.append(total)
            total += pair_domain.n_cells

        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=self.dtype), pairs, offsets

    def materialize_node_marginals(self, filter_dict: Dict[str, Any], constraints: List[Constraint],
                                   con: Optional["duckdb.DuckDBPyConnection"] = None) -> Tuple[List[np.ndarray], List[SparseConstraint]]:
        '''Measure every bag marginal for a node and prepare its constraints.

        The factored counterpart of materialize_node_data: instead of one Q @ x over the
        full joint, each bag is measured independently. Constraints are assigned to a bag
        whose columns contain their whole scope (guaranteed by the junction-tree build,
        which embeds each scope as a mandatory clique) and their indices are shifted into
        that bag's block of the node's concatenated marginal vector.

        Args:
            filter_dict (Dict[str, Any]): The node's filter conditions (column -> value mapping).
            constraints (List[Constraint]): Constraints for the node considering its level.
            con (Optional[duckdb.DuckDBPyConnection]): Per-thread cursor for concurrent measurement.

        Returns:
            Tuple[List[np.ndarray], List[SparseConstraint]]: One marginal per bag (aligned to
                junction_tree.bags) and the constraints in concatenated-marginal index space.

        Raises:
            ValueError: If a constraint's scope is not contained in any bag.
        '''
        assert self.junction_tree is not None, "No junction tree bound. Call build_marginal_domains first."

        marginals = [self._create_marginal_vector(i, filter_dict, con) for i in range(len(self.bag_domains))]

        node_constraints: List[SparseConstraint] = []
        for constraint in constraints:
            scope = constraint.scope()
            # We only need to find one bag that contains the constraint's scope, since in the optimizer, 
            # we need to apply the constraint just once. This is because we also have a constraint to
            # ensure that the marginals of the bags are consistent with each other, so if one bag satisfies
            # the constraint, all other bags that contain the scope will also satisfy it by transitivity.
            bag_index = self.junction_tree.bag_of_scope(scope)
            if bag_index is None:
                raise ValueError(
                    f"Constraint scope {set(scope)} is not contained in any bag; it must be "
                    f"passed as a mandatory clique when building the junction tree."
                )

            match constraint:
                case ContextualAggregateConstraint():
                    # The bag's marginal sums to the node total, so contextual values
                    # (e.g. the real total) are computed from it directly.
                    constraint.apply_aggregation_function(marginals[bag_index])

            indices, coefs, sense, cached_rhs = self._compiled_constraint(constraint, bag_index)
            # Only a contextual constraint's right-hand side varies per node; its cells do not.
            rhs = (float(constraint.value)
                   if isinstance(constraint, ContextualAggregateConstraint) else cached_rhs)
            node_constraints.append(SparseConstraint(indices, coefs, sense, rhs))

        return marginals, node_constraints

    def _compiled_constraint(self, constraint: Constraint, bag_index: int):
        '''Cells a constraint selects inside a bag, compiled once and reused.

        The cache key pins the constraint object as part of the value, so a garbage-collected
        constraint cannot have its id() reused by a different one while the entry lives.

        Args:
            constraint (Constraint): The constraint to compile.
            bag_index (int): Index of the bag whose sub-domain it is compiled against.

        Returns:
            Tuple: (indices shifted into the node's space, coefs, sense, rhs as compiled).
        '''
        key = (id(constraint), bag_index)
        hit = self._compiled_constraints.get(key)
        if hit is not None:
            _pin, indices, coefs, sense, rhs = hit
            return indices, coefs, sense, rhs

        sparse = constraint.to_sparse_constraint(self.bag_domains[bag_index])
        indices = sparse.indices + self.bag_offsets[bag_index]
        self._compiled_constraints[key] = (constraint, indices, sparse.coefs,
                                           sparse.sense, sparse.rhs)
        return indices, sparse.coefs, sparse.sense, sparse.rhs

    def spill_path(self, filter_dict: Dict[str, Any]) -> str:
        '''Get the spill file path from a filter dictionary.

        Args:
            filter_dict (Dict[str, Any]): The filter dictionary (column -> value mapping).

        Returns:
            str: File path for the spilled vector named by the filter values.
        '''

        # Create filename
        if filter_dict: name = '_'.join(f"{k}={v}" for k, v in filter_dict.items())
        else: name = "root"

        # Avoid problematic characters
        for ch in ('/', '\\', ' ', ':'):
            name = name.replace(ch, '_')
        # Build file path
        return os.path.join(self.spill_dir, name + '.npz')

    @staticmethod
    def canonical_column(vector: Union[np.ndarray, sp.spmatrix]) -> sp.csc_matrix:
        '''Put any vector into the canonical form the pipeline assumes downstream.

        Canonical means all four of: CSC format, shape (n, 1), no stored zeros, ascending
        indices. scipy can represent ~24 combinations of those and exactly one is valid here,
        and this is the single place that establishes that contract.

        Why each property matters, since none of the failures are loud:
          - CSC + (n, 1): in a CSR column `.indices` holds COLUMN indices (all zeros), so a
            CSR would read back as an empty support with nothing raised. A 1-D dense array
            handed straight to csc_matrix becomes a 1 x n ROW, whose row-slices are empty.
          - no stored zeros: nnz counts stored entries, so a COO built with a 0 in its data
            keeps it and `.indices` would no longer be the support.
          - sorted: the microdata reconstruction pairs records to cells positionally within
            a separator group, and its stable argsort only reproduces the dense pairing when
            the support arrives in cell order.

        Callers only need this at the two border points where a vector enters the pipeline
        (spill_vector on the way to disk, split_marginals on the way to reconstruction).

        Args:
            vector (Union[np.ndarray, sp.spmatrix]): Dense array or sparse matrix holding a
                single logical vector, in any orientation or format.

        Returns:
            sp.csc_matrix: The same values in canonical form.
        '''
        if not sp.issparse(vector):
            column = sp.csc_matrix(np.asarray(vector).reshape(-1, 1))
        else:
            # reshape BEFORE tocsc: scipy's sparse reshape returns COO when the shape changes.
            column = vector.reshape((-1, 1)).tocsc()
        column.eliminate_zeros()
        column.sort_indices()
        return column

    def spill_vector(self, path: str, contingency_vector: Union[np.ndarray, sp.spmatrix]) -> None:
        '''Write contingency vector to disk and free it from RAM.

        The vector is serialized with scipy's sparse .npz format. One file per node, named
        by its filter values, ensuring sibling nodes don't conflict.

        This is one of the two canonical-form borders: save_npz stores the format verbatim
        and load_npz restores it, so normalising on the way in is what makes load_vector's
        documented CSC contract true for its callers.

        Args:
            path (str): The file path where the vector will be spilled.
            contingency_vector (Union[np.ndarray, sp.spmatrix]): The contingency vector to write to disk.
        '''
        os.makedirs(self.spill_dir, exist_ok=True)
        sp.save_npz(path, self.canonical_column(contingency_vector))

    def load_vector(self, path: str) -> sp.csc_matrix:
        '''Reload contingency vector from disk and delete the file.

        Args:
            path (str): The file path to load the vector from.

        Returns:
            scipy.sparse.csc_matrix: The loaded contingency vector, shape (n, 1).
        '''
        contingency_vector = sp.load_npz(path)
        os.remove(path)
        return contingency_vector

    def split_marginals(self, vector: Union[np.ndarray, sp.spmatrix]) -> List[sp.csc_matrix]:
        '''Cut a node's concatenated marginal vector into one SPARSE column per bag.

        Inverse of np.concatenate(marginals): uses bag_offsets to slice the length-
        marginal_width vector at the bag boundaries. Accepts the sparse column the
        optimizer returns as well as a dense array.

        This is the second canonical-form border. The whole vector is canonicalised once
        and the pieces inherit it. See canonical_column for details.

        Args:
            vector (Union[np.ndarray, sp.spmatrix]): Length-marginal_width vector (or an
                (marginal_width, 1) sparse column).

        Returns:
            List[sp.csc_matrix]: One (bag n_cells, 1) column per bag, aligned to
                junction_tree.bags.
        '''
        vector = self.canonical_column(vector)

        if vector.shape[0] != self.marginal_width:
            raise ValueError(
                f"Expected a vector of length {self.marginal_width}, got {vector.shape[0]}."
            )

        return [vector[offset:offset + domain.n_cells]
                for offset, domain in zip(self.bag_offsets, self.bag_domains)]


    def update_child_marginals(self, joint_solution: sp.csc_matrix, filter_dicts: List[Dict[str, Any]]) -> None:
        '''Split a joint solution into per-child blocks and spill them to disk.

        Marginal counterpart of update_child_vectors: the joint vector holds one
        marginal_width block per child. The block is spilled whole, as the same sparse
        column the full-joint pipeline uses - it is only cut into per-bag pieces at the
        leaves, where the microdata is actually reconstructed.

        Args:
            joint_solution (sp.csc_matrix): Combined solution, shape (n_children * marginal_width, 1).
            filter_dicts (List[Dict[str, Any]]): Filter dictionary for each child, in block order.
        '''
        width = self.marginal_width
        for child_index, child_filter_dict in enumerate(filter_dicts):
            block = joint_solution[child_index * width:(child_index + 1) * width]
            self.spill_vector(self.spill_path(child_filter_dict), block)

    def update_child_vectors(self, joint_solution: sp.csc_matrix, filter_dicts: List[Dict[str, Any]]) -> None:
        '''Split joint solution into individual child vectors and spill to disk.

        Args:
            joint_solution (sp.csc_matrix): The combined sparse solution vector for all children,
                shape (num_children * n_cells, 1).
            filter_dicts (List[Dict[str, Any]]): List of filter dictionaries for each child.
        '''
        start = 0
        for child_filter_dict in filter_dicts:
            end = start + self.n_cells
            path = self.spill_path(child_filter_dict)
            self.spill_vector(path, joint_solution[start:end])
            start = end

    def merge_microdata_files(self) -> None:
        '''Merge Parquet microdata files into the output CSV using DuckDB.

        Uses a fresh connection with the default thread count (all cores) so the final
        COPY runs in parallel. The shared self.duckdb_con stays single-threaded for tree
        queries and is left untouched.
        '''
        parquet_pattern = os.path.join(self.microdata_dir, '*.parquet')

        try:
            con = duckdb.connect()
            con.execute(f"""
                COPY (SELECT * FROM read_parquet('{parquet_pattern}'))
                TO '{self.output_path}'
                (FORMAT CSV, HEADER TRUE, DELIMITER ';')
            """)
            con.close()
        except Exception as e:
            print(f"Warning: Error merging microdata files: {e}")

    def _construct_microdata_for_leaf(self, contingency_vector: sp.csc_matrix, filter_dict: Dict[str, Any]) -> pd.DataFrame:
        '''Construct microdata for a leaf node.

        Args:
            contingency_vector (sp.csc_matrix): The estimated cell counts as a sparse CSC
                column (n_cells, 1), storing only positive cells.
            filter_dict (Dict[str, Any]): Filter dictionary for hierarchical values.

        Returns:
            pd.DataFrame: Microdata for this leaf node.
        '''
        assert self.contingency_domain is not None, "Contingency domain is not built. Call build_contingency_domain first."
        domain = self.contingency_domain

        # The vector stores only positive cells (canonical CSC: no explicit zeros).
        nonzero_idx = contingency_vector.indices
        filtered_counts = contingency_vector.data

        # Decode only the nonzero cells into their attribute combinations
        filtered_query_values = domain.decode(nonzero_idx)

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

    def _construct_microdata_from_marginals(self, marginals: List[sp.csc_matrix], filter_dict: Dict[str, Any]) -> pd.DataFrame:
        '''Reconstruct a leaf's microdata from its per-bag marginals.

        Walks the junction tree from the root bag outwards, in the order that guarantees
        every bag is visited after its parent. The root bag's counts expand into partial
        records holding its columns; each subsequent bag then joins on the separator with
        its parent, filling in the columns it adds.

        The join is an exact integer allocation. The estimation phase already forced overlapping
        bags to agree on their separator, so within every separator group the number of partial
        records equals the number the child bag accounts for. Consequently the reconstructed
        records reproduce every estimated marginal exactly.

        Assigning records within a separator group is arbitrary in the sense that any
        assignment reproduces the same marginals - the bags constrain the joint only
        through what they share.

        Every per-bag quantity is computed on the bag's support only (``m.indices`` /
        ``m.data``), never over its whole cell space.

        Args:
            marginals (List[sp.csc_matrix]): Estimated integer counts per bag as
                (bag n_cells, 1) sparse columns with sorted indices, aligned to
                junction_tree.bags.
            filter_dict (Dict[str, Any]): Hierarchical values for this leaf.

        Returns:
            pd.DataFrame: One row per record, hierarchical columns followed by query columns.

        Raises:
            ValueError: If a bag's totals do not match the records to place, which means the
                separator consistency the estimation phase should have enforced is broken.
        '''
        assert self.junction_tree is not None, "No junction tree bound. Call build_marginal_domains first."
        junction_tree = self.junction_tree
        column_index = {column: i for i, column in enumerate(self.query_columns)}

        # Expand the root bag's counts into one partial record per person. Each
        # occupied cell is repeated as many times as its count, so cells[r] is the root-bag
        # cell that record r sits in.
        root_bag = junction_tree.root
        root_domain = self.bag_domains[root_bag]
        occupied, counts = marginals[root_bag].indices, marginals[root_bag].data
        cells = np.repeat(occupied, counts)

        n_records = len(cells)
        if n_records == 0:
            return pd.DataFrame(columns=self.hierarchical_columns + self.query_columns)

        # Per-record rank on every query column; -1 marks "not assigned yet". int32 rather
        # than int64: these are per-column ranks, bounded by the largest declared domain,
        # and this matrix is the memory ceiling of the whole factored pipeline
        # (n_records x n_columns for a whole leaf).
        records = np.full((n_records, len(self.query_columns)), -1, dtype=np.int32)

        # Fill in the root bag's columns first, then each bag in order after its parent.
        # cell_ranks is evaluated on `cells` (length n_records) instead of building the
        # bag-wide axis_ranks table and indexing into it.
        for column in root_domain.columns:
            records[:, column_index[column]] = root_domain.cell_ranks(cells, column)

        # order[1:] = every bag after the root, each visited once its parent (hence its
        # separator columns) is already filled in.
        for bag_index in junction_tree.order[1:]:
            separator = junction_tree.parent_separator[bag_index]
            domain = self.bag_domains[bag_index]
            occupied, counts = marginals[bag_index].indices, marginals[bag_index].data

            if counts.sum() != n_records:
                raise ValueError(
                    f"Bag {junction_tree.bags[bag_index]} totals {counts.sum()} but the node "
                    f"holds {n_records} records; the bags are not separator-consistent."
                )

            # Which separator group each partial record already belongs to. The separator's
            # columns were filled in by an earlier bag (running-intersection property), so
            # the mixed-radix id can be rebuilt from the ranks already assigned.
            record_group = np.zeros(n_records, dtype=np.int64)
            if separator:
                separator_domain = self.contingency_domain.subdomain(separator)
                for position, column in enumerate(separator):
                    record_group += (records[:, column_index[column]].astype(np.int64)
                                     * int(separator_domain.strides[position]))

            # Both sides encode `separator` with the same mixed-radix id (subdomain here vs
            # project_cells_to below), so the group ids are comparable.
            # Sorted by group id they line up positionally: within each group the two totals
            # are equal, so the p-th cell belongs to the p-th record.
            # expanded = this bag's occupied cells to place, group-ordered and repeated by
            # count. `counts` must be permuted by the same `order` as `occupied` before the
            # repeat - otherwise each cell would be repeated by another cell's count and the
            # reconstruction would be wrong.
            sep_ids = domain.project_cells_to(occupied, separator)
            order = np.argsort(sep_ids, kind="stable")
            expanded = np.repeat(occupied[order], counts[order])
            records_by_group = np.argsort(record_group, kind="stable")

            # Pair them positionally within each group: record r now knows its cell in this bag.
            assigned = np.empty(n_records, dtype=np.int64)
            assigned[records_by_group] = expanded

            # Now we have in assigned[r] the index in the bag's domain that record r belongs to.
            # To fill it we need to just get the rank of the columns not in the separator and write
            # those ranks into the resulting records.
            for column in domain.columns:
                if column not in separator:  # separator columns are already filled in
                    records[:, column_index[column]] = domain.cell_ranks(assigned, column)

        # Check that every column has been assigned a value for every record. 
        # If any column has a -1, it means that some records were not assigned a value for that column
        # and an error occurred during the reconstruction process. This should not happend.
        unassigned = np.flatnonzero((records < 0).any(axis=0))
        if len(unassigned):
            missing = [self.query_columns[i] for i in unassigned]
            raise ValueError(f"Columns {missing} are in no bag, so no value was reconstructed.")

        # Ranks -> declared values, column by column.
        leaf_df = pd.DataFrame({
            column: self.contingency_domain.domains[column][records[:, i]]
            for i, column in enumerate(self.query_columns)
        })
        # Stamp this leaf's hierarchical values (region, comuna, ...) onto every row.
        for column_name, value in filter_dict.items():
            leaf_df[column_name] = value

        return leaf_df[self.hierarchical_columns + self.query_columns]

    def write_microdata_from_marginals(self, node_id: int, children_marginals: List[List[sp.csc_matrix]],
                                       filter_dicts: List[Dict[str, Any]]) -> None:
        '''Reconstruct and write each leaf child's microdata (factored pipeline).

        Args:
            node_id (int): Parent node ID, used to name the output files.
            children_marginals (List[List[sp.csc_matrix]]): Per-child list of per-bag
                estimates, as the sparse columns split_marginals returns.
            filter_dicts (List[Dict[str, Any]]): Filter dictionary per child.
        '''
        for child_index, (marginals, filter_dict) in enumerate(zip(children_marginals, filter_dicts)):
            frame = self._construct_microdata_from_marginals(marginals, filter_dict)
            output_path = os.path.join(self.microdata_dir, f'node_{node_id}_child_{child_index}_microdata.parquet')
            frame.to_parquet(output_path, index=False)

    def write_microdata(self, node_id: int, contingency_vectors: List[sp.csc_matrix], filter_dicts: List[Dict[str, Any]]) -> None:
        '''Construct microdata for each child and write to separate Parquet files using DuckDB.

        Args:
            node_id (int): The parent node ID (used for naming the output files).
            contingency_vectors (List[sp.csc_matrix]): List of sparse cell-count vectors for each child.
            filter_dicts (List[Dict[str, Any]]): List of filter dictionaries for each child.
        '''

        for i, (contingency_vector, filter_dict) in enumerate(zip(contingency_vectors, filter_dicts)):
            df = self._construct_microdata_for_leaf(contingency_vector, filter_dict)
            output_path = os.path.join(self.microdata_dir, f'node_{node_id}_child_{i}_microdata.parquet')
            df.to_parquet(output_path, index=False)

    def noisy_vectors_exist(self, n_nodes: int, mech_param_spec: str) -> bool:
        '''Check if compatible pre-computed noise vectors exist and reuse them if possible.

        Searches for Zarr files matching the mechanism and parameters. Selects the first
        compatible file with n_nodes >= requested and width >= requested. Requires exact
        match on mechanism and parameters; compatible files can have more nodes/cells.
        The width is the node measurement length: the concatenated marginals under the
        factored pipeline, the full joint otherwise.

        Args:
            n_nodes (int): Requested number of nodes
            mech_param_spec (str): Mechanism parameter specification string (e.g., 'Laplace_1.0')

        Returns:
            bool: True if compatible vectors were found and loaded, False if new file created
        '''
        n_cells = self.noise_width

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
        Verifies that n_rows_generated >= n_expected_nodes to ensure completeness.
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

                # Validate metadata: ensure all expected nodes were actually generated
                zarr_path = os.path.join(self.noisy_dir, filename)
                try:
                    zarr_group = zarr.open_group(zarr_path, mode="r")
                    n_expected = zarr_group.attrs.get("n_expected_nodes", 0)
                    n_generated = zarr_group.attrs.get("n_rows_generated", 0)

                    # Only accept if generation is complete
                    if n_generated >= n_expected:
                        candidates.append((filename, file_n_nodes, file_n_cells))

                except Exception as e:
                    print(f"    Warning: Could not read metadata from {filename}: {e}")
                    continue

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

        width = self.noise_width

        self.noise_zarr_group = zarr.open_group(zarr_path, mode="w", zarr_format=2)
        self.noise_zarr_group.create_array(self.noisy_array_name, shape=(n_nodes, width),
                                      chunks=(1, width), dtype=self.dtype, compressor=compressor)

        self.noise_zarr_group.attrs["n_expected_nodes"] = n_nodes
        self.noise_zarr_group.attrs["n_cells"] = width
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
                                           self.noise_width, self.dtype, query_sensitivity)) as executor:

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

                        self.noise_zarr_group.attrs["n_rows_generated"] = done

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

            if done % max(1, n_nodes // 20) != 0:
                print(f"    Progress: {done}/{n_nodes}")

        self.noise_zarr_group.attrs["n_rows_generated"] = done
