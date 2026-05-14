import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from constraints.constraint import Constraint, SparseRow


class LogicalExpression(Constraint, ABC):
    """Base class for all logical expressions.

    Implementations must provide `reduce(contingency_df)` which returns a
    boolean `pd.Series` mask aligned with `contingency_df.index`.

    A standalone logical expression added as a constraint enforces that no row
    where the expression evaluates to False has positive count. That is encoded
    as a SparseRow: sum_{i in negated} x[i] == 0.
    """

    @abstractmethod
    def reduce(self, contingency_df: pd.DataFrame) -> pd.Series:
        """Reduce the expression to a boolean Series aligned with `contingency_df`."""
        raise NotImplementedError()

    def to_sparse_row(self, contingency_df: pd.DataFrame) -> SparseRow:
        # The expression must hold for the produced data. Equivalent to forcing the count
        # of rows that violate the expression (negated mask) to be zero.
        negated = ~self.reduce(contingency_df)
        indices = np.asarray(negated[negated].index, dtype=np.int64)
        coefs = np.ones_like(indices, dtype=np.float64)
        return SparseRow(indices=indices, coefs=coefs, sense='=', rhs=0.0)
