import numpy as np
import scipy.sparse as sp
from itertools import product
from typing import List, Optional, Tuple

from .expression import Expr, col


class WorkloadMatrix:
    """Builds a linear query workload matrix Q for the TopDown algorithm.

    Each query is a row of Q. A query answer on data vector x is Q @ x,
    where each entry Q[i, j] indicates how much cell j contributes to query i.
    For counting queries (the default), entries are 0 or 1.

    Queries are defined lazily and materialised into a scipy CSR matrix by calling
    `.build(domain)` once the contingency domain is known.

    Usage:
        from queries.workload_matrix import WorkloadMatrix
        from queries import col

        wm = WorkloadMatrix()
        wm.add(col('Sex') == 'M', name='count_male')
        wm.add((col('Sex') == 'F') & (col('Age') >= 18), name='adult_female')
        wm.add_marginal(['Sex'])          # one query per unique Sex value
        wm.add_marginal(['Sex', 'Age'])   # one query per (Sex, Age) combination

        Q = wm.build(domain)              # scipy CSR, shape (n_queries, n_cells)
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

    def build(self, domain) -> sp.csr_matrix:
        """Materialise Q as a sparse (n_queries x n_cells) CSR matrix with 0/1 entries.

        Args:
            domain: ContingencyDomain produced by
                    DataHandler.build_contingency_domain().

        Returns:
            scipy.sparse.csr_matrix of shape (n_queries, n_cells).

        Raises:
            ValueError: If no queries have been defined.
        """
        all_rows: List[np.ndarray] = []
        all_cols: List[np.ndarray] = []
        base = 0

        # Explicit single-predicate queries — one row each.
        for expr, _ in self._explicit:
            cols = domain.select(expr.evaluate(domain)).astype(np.int64)
            all_rows.append(np.full(len(cols), base, dtype=np.int64))
            all_cols.append(cols)
            base += 1

        # Marginal queries — every cell maps to exactly one combination, so the
        # block is the cell→group assignment computed from the mixed-radix structure.
        for columns in self._marginals:
            gid = np.zeros(domain.n_cells, dtype=np.int64)
            weight = 1
            for c in reversed(columns):
                gid += domain.axis_ranks(c) * weight
                weight *= len(domain.domains[c])
            all_rows.append(gid + base)
            all_cols.append(np.arange(domain.n_cells, dtype=np.int64))
            base += int(weight)

        if base == 0:
            raise ValueError(
                "WorkloadMatrix has no queries. "
                "Use .add() or .add_marginal() before calling .build()."
            )

        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        data = np.ones(len(rows), dtype=np.float64)
        return sp.coo_matrix((data, (rows, cols)), shape=(base, domain.n_cells)).tocsr()

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
