import pandas as pd

from typing import Callable

from .logical_expressions import LogicalExpression
from constraints.constraint import Constraint
from abc import ABC

class AggregateConstraint(Constraint, ABC):
    """
    Base class for all aggregate constraints. Provides the interface.
    """
    def __init__(self, expression: LogicalExpression, value: int) -> None:
        """
        Constructor of an AggregateConstraint.
        Args:
            expression (LogicalConstraint): A logical constraint to aggregate over.
            value (int, optional): A static value for the aggregate constraint.
        """
        self.expression = expression
        self.value = value

class SumEqual(AggregateConstraint):
    """Represents a sum equality constraint: Sum(expression) == value"""
    def to_constraint(self, contingency_df: pd.DataFrame) -> Callable:
        # Get reduced series
        reduced_series = self.expression.reduce(contingency_df)
        # Get indices where the expression is True
        indices = reduced_series[reduced_series].index
        # Return a function that checks if the sum of the contingency variable equals the value
        return lambda contingency_var, sum_val=self.value, _indices=indices: contingency_var[_indices].sum() == sum_val

# NOTE: Add more aggregate expressions as needed

