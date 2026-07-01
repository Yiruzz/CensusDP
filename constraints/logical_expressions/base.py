import numpy as np
from abc import ABC, abstractmethod

from constraints.constraint import Constraint
from constraints.sparse_constraint import SparseConstraint

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

    def to_sparse_constraint(self, domain) -> SparseConstraint:
        """Convert the logical expression into a sparse linear constraint.

        Returns a SparseConstraint representing: sum(x[indices_where_negated_mask_true]) == 0

        This enforces the logical constraint: if the logical condition is True, then the sum
        of variables at those cells must be non-zero (cells in the support). Conversely, cells
        where the condition is False must all be zero (negated_mask indices sum to 0).

        Args:
            domain: ContingencyDomain used as the cell space for evaluation.

        Returns:
            SparseConstraint: Sparse representation of the logical constraint.
        """
        # Get reduced boolean mask over the cells.
        reduced_mask = self.reduce(domain)

        # NOTE: The constraint will ensure that the resultant data holds the constraint as True
        # That means if we have something like A -> B, then the combinations that have A=True
        # must also have B=True. In other words, there should be no cases where A=True and B=False.
        # To enforce this, we can create a constraint that checks that the sum of the values
        # where the constraint is False is zero. Hence, we negate the reduced mask.
        negated_mask = ~reduced_mask

        # Indices of variables that do not satisfy the condition.
        # Coefficients are all 1.
        # TODO: Check whether the dtypes are appropriate; they default to float64, which consumes a lot of memory.
        indices = np.flatnonzero(negated_mask)

        # Return a constraint that enforces the sum of the variables at the selected domain indices to be zero.
        return SparseConstraint(
            indices=indices,
            sense="=",
            rhs=0.0
        )
