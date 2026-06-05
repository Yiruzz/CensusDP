import numpy as np
from functools import partial
from typing import Callable, List
from abc import ABC, abstractmethod

from constraints.constraint import Constraint

class LogicalExpression(Constraint, ABC):
    """Base class for all logical expressions.

    Implementations must provide reduce(domain) which returns a boolean
    np.ndarray of length domain.n_cells (entry j True iff cell j is selected).
    """

    @abstractmethod
    def reduce(self, domain) -> np.ndarray:
        """Reduce the constraint to a boolean mask over the contingency cells.

        Args:
            domain: ContingencyDomain used as the cell space for evaluation.
        Returns:
            np.ndarray: boolean mask (length domain.n_cells) where True indicates membership.
        """
        raise NotImplementedError()

    @staticmethod
    def no_true_constraint(contingency_var, indices: List[int]) -> bool:
        """Check that selected indices in the contingency variable are False.

        Args:
            contingency_var: A Pyomo Var/dict indexed by cell index.
            indices (List[int]): Indices in contingency_var that must all be zero.
        Returns:
            bool: True if all selected indices are zero (False), otherwise False.
        """
        return sum(contingency_var[i] for i in indices) == 0

    def to_constraint(self, domain) -> Callable:
        """Convert the logical expression into a constraint function.

        Args:
            domain: ContingencyDomain used as the cell space for evaluation.
        Returns:
            Callable: A function that takes a contingency variable and returns a
                      Pyomo expression / boolean for the optimizer.
        """
        # Get reduced boolean mask over the cells.
        reduced_mask = self.reduce(domain)

        # NOTE: The constraint will ensure that the resultant data holds the constraint as True
        # That means if we have something like A -> B, then the combinations that have A=True
        # must also have B=True. In other words, there should be no cases where A=True and B=False.
        # To enforce this, we can create a constraint that checks that the sum of the values
        # where the constraint is False is zero. Hence, we negate the reduced mask.
        negated_mask = ~reduced_mask

        # Get the integer cell indices where the (negated) constraint holds.
        # .tolist() yields plain python ints for safe use as Pyomo Var keys.
        indices = np.flatnonzero(negated_mask).tolist()

        # Return a function that checks if there are no True values in the negated indices.
        # This will be the function used as a constraint in the optimizer.
        return partial(LogicalExpression.no_true_constraint, indices=indices)
