import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple

from .expression import Expr, col

# A resolver is a closure produced by each query-definition method.
# It captures the query parameters and, when called with the contingency domain,
# returns (list_of_rows, list_of_names).
_Resolver = Callable[[pd.DataFrame], Tuple[List[np.ndarray], List[str]]]


class QueryWorkload:
    """Builds a linear query workload matrix Q for the TopDown algorithm.

    Maps a pandas-like syntax to the linear queries described by McKenna et al.
    in the HDMM framework. Each query becomes one row of Q; the workload answers
    on data vector x are Q @ x.

    Q is built lazily: resolvers capture the query intent at definition time and
    materialise into a matrix when .build(contingency_df) is called.

    Args:
        schema: Optional domain description {attr: [values]}, corresponding to
                dom(R) in McKenna et al. Used for documentation only.

    Example:
        from workload import QueryWorkload, col

        qw = (QueryWorkload(schema={'Sex': ['M', 'F'], 'Age': list(range(1, 6))})
              .value_counts(['Sex'])
              .value_counts(['Sex', 'Age'])
              .range_query('Age', 2, 4)
              .add(col('Income') == 'high', name='count_high_income'))

        Q       = qw.build(contingency_df)     # (n_queries x n_cells)
        answers = qw.answer(x, contingency_df) # Q @ x
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

        Rows are ordered by the sorted unique combinations of the attributes.

        Args:
            attributes: Column names to count over. Must be present in the
                        contingency domain passed to .build().

        Returns:
            self, for chaining.
        """
        def resolve(df: pd.DataFrame) -> Tuple[List[np.ndarray], List[str]]:
            rows, names = [], []
            unique_combos = (
                df[attributes]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            for _, combo_row in unique_combos.iterrows():
                expr: Optional[Expr] = None
                parts: List[str] = []
                for c in attributes:
                    eq = col(c) == combo_row[c]
                    expr = eq if expr is None else expr & eq
                    parts.append(f"{c}={combo_row[c]!r}")
                assert expr is not None
                rows.append(expr.evaluate(df).astype(float).values)
                names.append(', '.join(parts))
            return rows, names

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
        def resolve(df: pd.DataFrame) -> Tuple[List[np.ndarray], List[str]]:
            row = expression.evaluate(df).astype(float).values
            return [row], [name or repr(expression)]

        self._resolvers.append(resolve)
        return self

    def range_query(self, attr: str, start: Any, end: Any) -> 'QueryWorkload':
        """Add a single counting query for attr in [start, end] (inclusive).

        Shorthand for .add((col(attr) >= start) & (col(attr) <= end)).
        Corresponds to a single row of the AllRange matrix from McKenna et al.

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

    def build(self, contingency_df: pd.DataFrame) -> np.ndarray:
        """Materialise Q as a (n_queries x n_cells) float numpy matrix.

        Each row is a 0/1 vector indicating which contingency cells are included
        in that query. Safe to call multiple times (idempotent).

        Args:
            contingency_df: Global contingency domain produced by
                            DataHandler.generate_contingency_dataframe().

        Returns:
            np.ndarray of shape (n_queries, n_cells).

        Raises:
            ValueError: If no queries have been defined.
        """
        all_rows: List[np.ndarray] = []
        all_names: List[str] = []

        for resolve in self._resolvers:
            rows, names = resolve(contingency_df)
            all_rows.extend(rows)
            all_names.extend(names)

        if not all_rows:
            raise ValueError(
                "QueryWorkload has no queries. "
                "Call .value_counts(), .add(), or .range_query() first."
            )

        self._query_names = all_names
        return np.vstack(all_rows)

    def answer(self, x: np.ndarray, contingency_df: pd.DataFrame) -> np.ndarray:
        """Compute Q @ x (workload answers on data vector x).

        Args:
            x:              Data vector, one count per contingency cell.
            contingency_df: The global contingency domain.

        Returns:
            np.ndarray of shape (n_queries,).
        """
        return self.build(contingency_df) @ x

    # ── Introspection ──────────────────────────────────────────────────────────

    @property
    def query_names(self) -> List[str]:
        """Row labels from the last .build() call (empty before first build)."""
        return list(self._query_names)

    def __len__(self) -> int:
        return len(self._resolvers)

    def __repr__(self) -> str:
        return f"QueryWorkload({len(self._resolvers)} query spec(s))"
