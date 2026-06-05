import numpy as np
import scipy.sparse as sp
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple

from .expression import Expr, col

# A resolver is a closure produced by each query-definition method.
# It captures the query parameters and, when called with the ContingencyDomain,
# returns COO components for its block of query rows:
#   (rows, cols, n_rows, names)
# where rows/cols index a local (n_rows x domain.n_cells) 0/1 submatrix and names
# labels each local row. build() stacks the blocks into one CSR matrix.
_Resolver = Callable[[Any], Tuple[np.ndarray, np.ndarray, int, List[str]]]


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

    # ── Query definition ───────────────────────────────────────────────────────

    def value_counts(self, attributes: List[str]) -> 'QueryWorkload':
        """Add one counting query per unique combination of attributes.

        Mirrors pandas df.value_counts(attributes): each unique combination of
        attribute values becomes one row in Q that counts all contingency cells
        matching that combination (a marginal query in McKenna et al.'s terms).

        Every cell belongs to exactly one combination, so this whole block is the
        cell→group assignment: row = group id, col = cell, computed in O(n_cells)
        from the mixed-radix structure with no per-combination evaluation.

        Args:
            attributes: Column names to count over. Must be present in the domain.

        Returns:
            self, for chaining.
        """
        def resolve(domain) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
            # group id of each cell = mixed radix over `attributes` (first attr most
            # significant), so groups are ordered lexicographically like the values.
            gid = np.zeros(domain.n_cells, dtype=np.int64)
            weight = 1
            for c in reversed(attributes):
                size = len(domain.domains[c])
                gid += domain.axis_ranks(c) * weight
                weight *= size
            num_groups = int(weight)

            rows = gid
            cols = np.arange(domain.n_cells, dtype=np.int64)
            names = [
                ', '.join(f"{c}={v!r}" for c, v in zip(attributes, combo))
                for combo in product(*(domain.domains[c] for c in attributes))
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
        base = 0
        for resolve in self._resolvers:
            rows, cols, n_rows, names = resolve(domain)
            all_rows.append(rows + base)
            all_cols.append(cols)
            all_names.extend(names)
            base += n_rows

        rows = np.concatenate(all_rows) if all_rows else np.empty(0, dtype=np.int64)
        cols = np.concatenate(all_cols) if all_cols else np.empty(0, dtype=np.int64)
        data = np.ones(len(rows), dtype=np.float64)

        self._query_names = all_names
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

    def __len__(self) -> int:
        return len(self._resolvers)

    def __repr__(self) -> str:
        return f"QueryWorkload({len(self._resolvers)} query spec(s))"
