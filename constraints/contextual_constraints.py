import pandas as pd
from functools import partial
from typing import Callable, List

from constraints.logical_expressions.base import LogicalExpression
from .aggregate_constraints import AggregateConstraint

from abc import ABC

class ContextualAggregateConstraint(AggregateConstraint, ABC):
    """
    Base class for all contextual aggregate constraints. Provides the interface.

    A contextual aggregate constraint is an aggregate constraint that depends on the context
    of the hierarchical node in which it is applied. This means that the constraint may vary
    based on the data subset represented by the node.

    That means that the value of the constraint can be dynamically calculated based on the
    DataFrame associated with the node's context.
    """
    def __init__(self, expression: LogicalExpression, aggregation_function: Callable[[pd.DataFrame], int]) -> None:
        """
        Constructor of a ContextualAggregateConstraint.
        Args:
            expression (LogicalExpression): A logical expression to aggregate over.
            aggregation_function (Callable[[pd.DataFrame], int], optional): A function 
                to calculate the value dynamically using the node's DataFrame. Defaults to None.
        """
        # We initialize the base AggregateConstraint with a placeholder value (-1).
        # It will be changed later when we apply the aggregation function at runtime. 
        super().__init__(expression=expression, value=-1)
        self.aggregation_function = aggregation_function
        
    def apply_aggregation_function(self, contextualized_df: pd.DataFrame) -> int:
        """Calculate the value for the aggregate expression.

        Args:
            contingency_df (pd.DataFrame): The DataFrame to use for calculation.
        Returns:
            int: The calculated value.
        """
        if self.aggregation_function is not None:
            self.value = self.aggregation_function(contextualized_df)
            return self.value
        else:
            raise ValueError("No aggregation_function provided to compute the value.")
        

class SumEqualRealTotal(ContextualAggregateConstraint):
    """Convenience class for the user to easily set the Real Total constraint."""
    def __init__(self, expression: LogicalExpression) -> None:

        # Function to calculate the real total from the DataFrame
        get_real_total = lambda df: len(df)
        
        # The true total is the count of rows in the node's context
        super().__init__(expression=expression, aggregation_function=get_real_total)

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
    
    def to_constraint(self, contingency_df):
        # Get reduced series
        reduced_series = self.expression.reduce(contingency_df)
        # Get indices where the expression is True
        indices = reduced_series[reduced_series].index
        # Return a function that checks if the sum of the contingency variable equals the value
        return partial(SumEqualRealTotal.check_sum, sum_val=self.value, indices=indices)
    
# NOTE: Add more aggregate expressions as needed

