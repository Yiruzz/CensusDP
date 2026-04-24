import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from .expression import Expr, col


class WorkloadMatrix:
    """Builds a linear query workload matrix Q for the TopDown algorithm.

    Each query is a row of Q. A query answer on data vector x is Q @ x,
    where each entry Q[i, j] indicates how much cell j contributes to query i.
    For counting queries (the default), entries are 0 or 1.

    Queries are defined lazily and materialised into a numpy matrix by calling
    `.build(contingency_df)` once the contingency domain is known.

    Usage:
        from workload import WorkloadMatrix, col

        wm = WorkloadMatrix()
        wm.add(col('Sex') == 'M', name='count_male')
        wm.add((col('Sex') == 'F') & (col('Age') >= 18), name='adult_female')
        wm.add_marginal(['Sex'])          # one query per unique Sex value
        wm.add_marginal(['Sex', 'Age'])   # one query per (Sex, Age) combination

        Q = wm.build(contingency_df)      # shape: (n_queries, n_cells)
    """

    def __init__(self) -> None:
        self._explicit: List[Tuple[Expr, str]] = []
        self._marginals: List[List[str]] = []

    # ------------------------------------------------------------------ 
    # Query definition API
    # ------------------------------------------------------------------

    def add(self, expression: Expr, name: str = '') -> 'WorkloadMatrix':
        """Add a single counting query defined by a boolean predicate.

        Args:
            expression: A workload expression built with `col(...)`.
            name: Optional label for the query (used in __repr__ only).

        Returns:
            self, for chaining.
        """
        self._explicit.append((expression, name))
        return self

    def add_marginal(self, columns: List[str]) -> 'WorkloadMatrix':
        """Add one counting query per unique combination of the given columns.

        Each generated query counts all contingency cells that match one specific
        combination of `columns`, summing over all other attributes. This is
        equivalent to a sub-table marginal.

        Resolution is lazy — unique values are determined at `.build()` time.

        Args:
            columns: Attribute names to marginalise over. Must be present in
                     the contingency domain.

        Returns:
            self, for chaining.
        """
        self._marginals.append(list(columns))
        return self

    # ------------------------------------------------------------------
    # Matrix construction
    # ------------------------------------------------------------------

    def build(self, contingency_df: pd.DataFrame) -> np.ndarray:
        """Materialise Q as a (n_queries x n_cells) float numpy matrix.

        Args:
            contingency_df: The global contingency domain DataFrame produced
                            by DataHandler.generate_contingency_dataframe().

        Returns:
            np.ndarray of shape (n_queries, n_cells) with 0/1 entries.

        Raises:
            ValueError: If no queries have been defined.
        """
        rows: List[np.ndarray] = []

        # Explicit single-predicate queries
        for expr, _ in self._explicit:
            mask = expr.evaluate(contingency_df)
            rows.append(mask.astype(float).values)

        # Lazy marginal queries - resolve unique combinations now
        for columns in self._marginals:
            unique_combos = contingency_df[columns].drop_duplicates().reset_index(drop=True)
            for _, combo_row in unique_combos.iterrows():
                # Build compound equality expression for this combination
                expr: Optional[Expr] = None
                for c in columns:
                    eq = col(c) == combo_row[c]
                    expr = eq if expr is None else expr & eq
                assert expr is not None
                mask = expr.evaluate(contingency_df)
                rows.append(mask.astype(float).values)

        if not rows:
            raise ValueError(
                "WorkloadMatrix has no queries. "
                "Use .add() or .add_marginal() before calling .build()."
            )

        return np.vstack(rows)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def n_explicit(self) -> int:
        """Number of explicit (non-marginal) queries."""
        return len(self._explicit)

    def n_marginal_specs(self) -> int:
        """Number of marginal specifications (each may expand to many rows)."""
        return len(self._marginals)

    def __repr__(self) -> str:
        return (
            f"WorkloadMatrix("
            f"{len(self._explicit)} explicit queries, "
            f"{len(self._marginals)} marginal spec(s))"
        )
