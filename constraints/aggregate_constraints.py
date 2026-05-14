import pandas as pd
import numpy as np

from .logical_expressions import LogicalExpression
from constraints.constraint import Constraint, SparseRow
from abc import ABC


class AggregateConstraint(Constraint, ABC):
    """Base class for all aggregate constraints. Provides the interface."""

    def __init__(self, expression: LogicalExpression, value: int) -> None:
        """Constructor of an AggregateConstraint.

        Args:
            expression (LogicalConstraint): A logical constraint to aggregate over.
            value (int, optional): A static value for the aggregate constraint.
        """
        self.expression = expression
        self.value = value


class SumEqual(AggregateConstraint):
    """Represents a sum equality constraint: Sum(expression) == value"""

    def to_sparse_row(self, contingency_df: pd.DataFrame) -> SparseRow:
        reduced_series = self.expression.reduce(contingency_df)
        indices = np.asarray(reduced_series[reduced_series].index, dtype=np.int64)
        coefs = np.ones_like(indices, dtype=np.float64)
        return SparseRow(indices=indices, coefs=coefs, sense='=', rhs=float(self.value))

# NOTE: Add more aggregate expressions as needed
