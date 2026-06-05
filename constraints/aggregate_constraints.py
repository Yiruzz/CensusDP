import numpy as np
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
    def check_sum(contingency_var, sum_val: int, indices: List[int]) -> bool:
        """Check that the sum of selected indices in the contingency variable equals a target value.

        Args:
            contingency_var: A Pyomo Var/dict indexed by cell index.
            sum_val (int): The expected sum of the selected elements.
            indices (List[int]): Indices in contingency_var whose values will be summed.
        Returns:
            bool: True if the sum of the selected elements equals sum_val, otherwise False.
        """

        return sum(contingency_var[i] for i in indices) == sum_val

    def to_constraint(self, domain) -> Callable:
        # Reduce to a boolean mask over the cells, then take the selected indices.
        reduced_mask = self.expression.reduce(domain)
        indices = np.flatnonzero(reduced_mask).tolist()
        # Return a function that checks if the sum of the contingency variable equals the value
        return partial(SumEqual.check_sum, sum_val=self.value, indices=indices)
    
# NOTE: Add more aggregate expressions as needed

