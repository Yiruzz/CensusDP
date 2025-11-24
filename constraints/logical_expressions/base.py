import pandas as pd
from typing import Callable
from abc import ABC, abstractmethod

from constraints.constraint import Constraint

class LogicalExpression(Constraint, ABC):
    """Base class for all logical expressions.

    Implementations must provide `reduce(contingency_df)` which returns a
    boolean `pd.Series` mask aligned with `contingency_df.index`.
    """

    @abstractmethod
    def reduce(self, contingency_df: pd.DataFrame) -> pd.Series:
        """Reduce the constraint to a boolean Series aligned with `contingency_df`.

        Args:
            contingency_df: Pandas DataFrame used as the domain for evaluation.
        Returns:
            pd.Series: boolean mask where True indicates membership.
        """
        raise NotImplementedError()
    
    def to_constraint(self, contingency_df: pd.DataFrame) -> Callable:
        """Convert the logical expression into a constraint function.

        Args:
            contingency_df: Pandas DataFrame used as the domain for evaluation.
        Returns:
            Callable: A function that takes a pd.Series and returns a boolean Series.
        """
        # Get reduced series
        reduced_series = self.reduce(contingency_df)

        # NOTE: The constraint will ensure that the resultant data holds the constraint as True
        # That means if we have something like A -> B, then the combinations that have A=True
        # must also have B=True. In other words, there should be no cases where A=True and B=False.
        # To enforce this, we can create a constraint that checks that the sum of the values
        # where the constraint is False is zero. Hence, we negate the reduced series.
        negated_series = ~reduced_series

        # Get the indices where the constraint is True
        indices = negated_series[negated_series].index

        # Return a function that checks if there are no True values in the negated indices.
        # This will be the function used as a constraint in the optimizer.
        return lambda contingency_var, _indices=indices: contingency_var[_indices].sum() == 0


