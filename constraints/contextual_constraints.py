import pandas as pd
import numpy as np
from typing import Callable

from constraints.logical_expressions.base import LogicalExpression
from constraints.constraint import SparseRow
from .aggregate_constraints import AggregateConstraint

from abc import ABC


class ContextualAggregateConstraint(AggregateConstraint, ABC):
    """Aggregate constraint whose value depends on the node's data subset.

    The numeric `value` is computed at tree-build time by `apply_aggregation_function`
    using the DataFrame that corresponds to the node's hierarchical context. After
    that, the constraint behaves like a regular AggregateConstraint and exposes
    `to_sparse_row`.
    """

    def __init__(self, expression: LogicalExpression, aggregation_function: Callable[[pd.DataFrame], int]) -> None:
        # Placeholder value until apply_aggregation_function runs against the node's data.
        super().__init__(expression=expression, value=-1)
        self.aggregation_function = aggregation_function

    def apply_aggregation_function(self, contextualized_df: pd.DataFrame) -> int:
        if self.aggregation_function is None:
            raise ValueError("No aggregation_function provided to compute the value.")
        self.value = self.aggregation_function(contextualized_df)
        return self.value


class SumEqualRealTotal(ContextualAggregateConstraint):
    """Convenience class for the user to easily set the Real Total constraint."""

    def __init__(self, expression: LogicalExpression) -> None:
        super().__init__(expression=expression, aggregation_function=lambda df: len(df))

    def to_sparse_row(self, contingency_df: pd.DataFrame) -> SparseRow:
        reduced_series = self.expression.reduce(contingency_df)
        indices = np.asarray(reduced_series[reduced_series].index, dtype=np.int64)
        coefs = np.ones_like(indices, dtype=np.float64)
        return SparseRow(indices=indices, coefs=coefs, sense='=', rhs=float(self.value))

# NOTE: Add more aggregate expressions as needed
