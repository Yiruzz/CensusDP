import pandas as pd

from typing import Callable

from constraints.logical_constraints.base import LogicalConstraint
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
    def __init__(self, expression: LogicalConstraint, aggregation_function: Callable[[pd.DataFrame], int]) -> None:
        """
        Constructor of a ContextualAggregateConstraint.
        Args:
            expression (LogicalConstraint): A logical constraint to aggregate over.
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
    def __init__(self, expression: LogicalConstraint) -> None:

        # Function to calculate the real total from the DataFrame
        get_real_total = lambda df: len(df)
        
        # The true total is the count of rows in the node's context
        super().__init__(expression=expression, aggregation_function=get_real_total)
    
    def to_constraint(self, contingency_df):
        # Get reduced series
        reduced_series = self.expression.reduce(contingency_df)
        # Get indices where the expression is True
        indices = reduced_series[reduced_series].index
        # Return a function that checks if the sum of the contingency variable equals the value
        return lambda contingency_var, sum_val=self.value, _indices=indices: contingency_var[_indices].sum() == sum_val

# NOTE: Add more aggregate expressions as needed

