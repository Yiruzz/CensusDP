import pandas as pd
from functools import partial
from typing import Callable, List

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

    @staticmethod
    def check_sum(contingency_var: pd.Series, sum_val: int, indices: List[int]) -> bool:
        """Check that the sum of selected indices in the contingency variable equals a target value.
        
        Args:
            contingency_var (pd.Series): A series containing numeric values.
            sum_val (int): The expected sum of the selected elements.
            indices (List[int]): Indices in contingency_var whose values will be summed.
        Returns:
            bool: True if the sum of the selected elements equals sum_val, otherwise False.
        """

        return sum(contingency_var[i] for i in indices) == sum_val
    
    def to_constraint(self, contingency_df: pd.DataFrame) -> Callable:
        # Get reduced series
        reduced_series = self.expression.reduce(contingency_df)
        # Get indices where the expression is True
        indices = reduced_series[reduced_series].index
        # Return a function that checks if the sum of the contingency variable equals the value
        return partial(SumEqual.check_sum, sum_val=self.value, indices=indices)
    
# NOTE: Add more aggregate expressions as needed

