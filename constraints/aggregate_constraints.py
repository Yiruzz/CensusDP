import numpy as np
from typing import FrozenSet

from .logical_expressions import LogicalExpression
from constraints.sparse_constraint import SparseConstraint
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

    def scope(self) -> FrozenSet[str]:
        return self.expression.scope()

class SumEqual(AggregateConstraint):
    """Represents a sum equality constraint: Sum(expression) == value"""

    def to_sparse_constraint(self, domain) -> SparseConstraint:
        '''Convert logical expression to sparse linear constraint.

        Args:
            domain: The domain over which to reduce the logical expression.

        Returns:
            SparseConstraint: Sparse representation of sum(x[indices]) == self.value
        '''
        # Reduce to a boolean mask over the cells, then take the selected indices.
        reduced_mask = self.expression.reduce(domain)
        indices = np.flatnonzero(reduced_mask)
        coefs = np.ones(len(indices))

        # Return a constraint that encapsulates that several values of the contingency variable must sum to the given value.
        return SparseConstraint(
            indices=indices,
            coefs=coefs,
            sense="=",
            rhs=float(self.value)
        )
    
# NOTE: Add more aggregate expressions as needed

