import numpy as np
import scipy.sparse as sp
from itertools import product
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .expression import Expr, col

# A resolver is a closure produced by each query-definition method.
# It captures the query parameters and, when called with the ContingencyDomain,
# returns COO components for its block of query rows:
#   (rows, cols, n_rows, names)
# where rows/cols index a local (n_rows x domain.n_cells) 0/1 submatrix and names
# labels each local row. build() stacks the blocks into one CSR matrix.
_Resolver = Callable[[Any], Tuple[np.ndarray, np.ndarray, int, List[str]]]


class Recode:
    """A coarsened axis. One column's declared values bucketed into groups.

    Passed in place of a column name to QueryWorkload.value_counts() to measure a marginal
    over a *binned* version of a column.

    The groups must partition the column's declared domain: every value in exactly one
    group. That is not a stylistic requirement, it is what makes the block a marginal, i.e.
    what makes its column sum 1 and therefore its L1 sensitivity 1 (doubled for bounded DP).
    A value left out would silently drop cells from every row of the block; a value in two
    groups would silently double the block's sensitivity. Both raise instead.

    Args:
        column: The column being coarsened. Must be in the domain.
        groups: {label: [values]} or [(label, [values])], in the order the groups should be
            numbered.
        name: Optional label for the coarsened axis, used in query names. Defaults to
            '<column>_recoded'.

    Example:
        age_groups_64 = Recode('age', {'0 to 63': list(range(64)),
                                       '64 or more': list(range(64, 116))},
                               name='ageGroups64')
        workload.value_counts([age_groups_64, 'sex'])
    """

    def __init__(self, column: str,
                 groups: Union[Mapping[Any, Sequence[Any]], Sequence[Tuple[Any, Sequence[Any]]]],
                 name: str = '') -> None:
        items = list(groups.items()) if isinstance(groups, Mapping) else list(groups)
        if not items:
            raise ValueError(f"Recode on '{column}' has no groups.")
        self.column: str = column
        self.labels: List[Any] = [label for label, _ in items]
        self.values: List[Sequence[Any]] = [list(values) for _, values in items]
        self.name: str = name or f'{column}_recoded'

    def __len__(self) -> int:
        return len(self.labels)

    def digit_lookup(self, domain) -> np.ndarray:
        """Map each rank of self.column to its group index, validating the partition.

        Returns:
            np.ndarray: Length len(domain.domains[column]) array of group indices.

        Raises:
            ValueError: If the groups do not partition the column's declared domain.
        """
        declared = domain.domains[self.column]
        lookup = np.full(len(declared), -1, dtype=np.int64)
        for index, values in enumerate(self.values):
            matching = np.flatnonzero(np.isin(declared, values))
            clash = matching[lookup[matching] >= 0]
            if clash.size:
                raise ValueError(
                    f"Recode '{self.name}' on column '{self.column}' puts "
                    f"{list(declared[clash][:5])} in more than one group, so the block would "
                    f"not be a marginal and its sensitivity would not be 1. Groups must "
                    f"partition the declared domain."
                )
            lookup[matching] = index
        uncovered = np.flatnonzero(lookup < 0)
        if uncovered.size:
            raise ValueError(
                f"Recode '{self.name}' on column '{self.column}' leaves "
                f"{uncovered.size} declared value(s) in no group, e.g. "
                f"{list(declared[uncovered][:5])}. Those cells would be counted by no row of "
                f"the block. Groups must partition the declared domain."
            )
        return lookup

    def __repr__(self) -> str:
        return f"Recode({self.column!r}, {len(self.labels)} groups, name={self.name!r})"


def _axis_digits(domain, axis: Union[str, Recode]) -> Tuple[np.ndarray, int, List[Any], str]:
    """Resolve one value_counts axis to (per-cell digit, number of digits, labels, name).

    A plain column name is its own mixed-radix digit, a Recode maps that digit through its
    group lookup. Both go through domain.axis_ranks, which dispatches to cell_ranks and so
    stays correct on a RestrictedDomain.
    """
    if isinstance(axis, Recode):
        lookup = axis.digit_lookup(domain)
        return lookup[domain.axis_ranks(axis.column)], len(axis), axis.labels, axis.name
    declared = domain.domains[axis]
    return domain.axis_ranks(axis), len(declared), list(declared), axis


class QueryWorkload:
    """Builds a sparse linear query workload matrix Q for the TopDown algorithm.

    Maps a pandas-like syntax to the linear queries.
    Each query becomes one row of Q; the workload answers on data vector x are Q @ x.

    Q is built lazily and sparsely: resolvers capture the query intent at
    definition time and materialise into a scipy CSR matrix when
    .build(domain) is called - no dense intermediate is ever formed.

    Args:
        schema: Optional domain description {attr: [values]}, corresponding to
                dom(R) in McKenna et al. Used for documentation only.

    Example:
        from queries import QueryWorkload, col

        qw = (QueryWorkload(schema={'Sex': ['M', 'F'], 'Age': list(range(1, 6))})
              .value_counts(['Sex'])
              .value_counts(['Sex', 'Age'])
              .range_query('Age', 2, 4)
              .add(col('Income') == 'high', name='count_high_income'))

        Q       = qw.build(domain)              # scipy CSR (n_queries x n_cells)
        answers = qw.answer(x, domain)          # Q @ x
    """

    def __init__(self, schema: Optional[Dict[str, List[Any]]] = None) -> None:
        self.schema = schema
        self._resolvers: List[_Resolver] = []
        self._query_names: List[str] = []  # populated after each .build()
        self._block_offsets: List[int] = []  # populated after each .build()

    # ── Query definition ───────────────────────────────────────────────────────

    def value_counts(self, attributes: List[Union[str, Recode]]) -> 'QueryWorkload':
        """Add one counting query per unique combination of attributes.

        Mirrors pandas df.value_counts(attributes): each unique combination of
        attribute values becomes one row in Q that counts all contingency cells
        matching that combination (a marginal query in McKenna et al.'s terms).

        Every cell belongs to exactly one combination, so this whole block is the
        cell→group assignment: row = group id, col = cell, computed in O(n_cells)
        from the mixed-radix structure with no per-combination evaluation.

        Args:
            attributes: Column names to count over, each either a plain name present in the
                domain or a Recode coarsening one. A Recode partitions its column, so the
                block stays a marginal either way.

        Returns:
            self, for chaining.
        """
        def resolve(domain) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
            # group id of each cell = mixed radix over `attributes` (first attr most
            # significant), so groups are ordered lexicographically like the values.
            axes = [_axis_digits(domain, a) for a in attributes]
            gid = np.zeros(domain.n_cells, dtype=np.int64)
            weight = 1
            for digits, size, _labels, _name in reversed(axes):
                gid += digits * weight
                weight *= size
            num_groups = int(weight)

            rows = gid
            cols = np.arange(domain.n_cells, dtype=np.int64)
            names = [
                ', '.join(f"{name}={v!r}" for (_d, _s, _l, name), v in zip(axes, combo))
                for combo in product(*(labels for _d, _s, labels, _n in axes))
            ]
            return rows, cols, num_groups, names

        self._resolvers.append(resolve)
        return self

    def add(self, expression: Expr, name: str = '') -> 'QueryWorkload':
        """Add a single counting query defined by a boolean predicate.

        Args:
            expression: A workload expression built with col(...).
            name: Optional label shown in query_names.

        Returns:
            self, for chaining.
        """
        def resolve(domain) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
            cols = domain.select(expression.evaluate(domain)).astype(np.int64)
            rows = np.zeros(len(cols), dtype=np.int64)
            return rows, cols, 1, [name or repr(expression)]

        self._resolvers.append(resolve)
        return self

    def range_query(self, attr: str, start: Any, end: Any) -> 'QueryWorkload':
        """Add a single counting query for attr in [start, end] (inclusive).

        Shorthand for .add((col(attr) >= start) & (col(attr) <= end)).

        Args:
            attr:  Attribute name.
            start: Lower bound (inclusive).
            end:   Upper bound (inclusive).

        Returns:
            self, for chaining.
        """
        expr = (col(attr) >= start) & (col(attr) <= end)
        return self.add(expr, name=f'{attr}[{start},{end}]')

    # ── Matrix construction ────────────────────────────────────────────────────

    def build(self, domain) -> sp.csr_matrix:
        """Materialise Q as a sparse (n_queries x n_cells) CSR matrix.

        Each row is a 0/1 indicator of which contingency cells the query includes.

        Args:
            domain: ContingencyDomain produced by DataHandler.build_contingency_domain().

        Returns:
            scipy.sparse.csr_matrix of shape (n_queries, n_cells).

        Raises:
            ValueError: If no queries have been defined.
        """
        if not self._resolvers:
            raise ValueError(
                "QueryWorkload has no queries. "
                "Call .value_counts(), .add(), or .range_query() first."
            )

        all_rows: List[np.ndarray] = []
        all_cols: List[np.ndarray] = []
        all_names: List[str] = []
        offsets = [0]
        base = 0
        for resolve in self._resolvers:
            rows, cols, n_rows, names = resolve(domain)
            all_rows.append(rows + base)
            all_cols.append(cols)
            all_names.extend(names)
            base += n_rows
            offsets.append(base)

        rows = np.concatenate(all_rows) if all_rows else np.empty(0, dtype=np.int64)
        cols = np.concatenate(all_cols) if all_cols else np.empty(0, dtype=np.int64)
        data = np.ones(len(rows), dtype=np.float64)

        self._query_names = all_names
        self._block_offsets = offsets
        return sp.coo_matrix((data, (rows, cols)), shape=(base, domain.n_cells)).tocsr()

    def answer(self, x: np.ndarray, domain) -> np.ndarray:
        """Compute Q @ x (workload answers on data vector x).

        Args:
            x:      Data vector, one count per contingency cell.
            domain: The ContingencyDomain.

        Returns:
            np.ndarray of shape (n_queries,).
        """
        result = self.build(domain) @ x
        return np.asarray(result).ravel()

    # ── Introspection ──────────────────────────────────────────────────────────

    @property
    def query_names(self) -> List[str]:
        """Row labels from the last .build() call (empty before first build)."""
        return list(self._query_names)

    @property
    def block_offsets(self) -> List[int]:
        """Row boundaries of each query definition, from the last .build() call.

        Length len(self) + 1: block i occupies rows [block_offsets[i], block_offsets[i+1]).
        These are the units the privacy budget is split over, each .value_counts() /
        .add() / .range_query() call is one block.
        """
        return list(self._block_offsets)

    def __len__(self) -> int:
        return len(self._resolvers)

    def __repr__(self) -> str:
        return f"QueryWorkload({len(self._resolvers)} query spec(s))"


def is_identity_workload(query_matrix) -> bool:
    """True when Q is the identity, i.e. every cell is answered directly and Q can be dropped.

    Args:
        query_matrix: A scipy sparse matrix, a dense ndarray, or None.

    Returns:
        bool: True when the matrix is exactly the identity. None counts as the identity, since
            that is how "no workload" is represented.
    """
    if query_matrix is None:
        return True

    n_rows, n_cols = query_matrix.shape
    if n_rows != n_cols:
        return False

    if sp.issparse(query_matrix):
        matrix = query_matrix.tocsr()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return (matrix.nnz == n_rows
                and np.array_equal(matrix.indptr, np.arange(n_rows + 1))
                and np.array_equal(matrix.indices, np.arange(n_rows))
                and bool(np.all(matrix.data == 1)))

    # Dense: only reachable for small n (a dense Q is not representable otherwise), so the
    # per-element scan is fine. np.eye is deliberately NOT materialised.
    dense = np.asarray(query_matrix)
    occupied = np.flatnonzero(dense.ravel())
    return (len(occupied) == n_rows
            and np.array_equal(occupied, np.arange(n_rows) * (n_cols + 1))
            and bool(np.all(dense[dense != 0] == 1)))

