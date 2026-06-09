import pandas as pd
import numpy as np

from .logical_expressions import LogicalExpression
from constraints.constraint import Constraint, SparseConstraint
from abc import ABC


class AggregateConstraint(Constraint, ABC):
    '''Base class for all aggregate constraints.

    An aggregate constraint enforces that the sum of contingency cells
    satisfying a logical expression equals a target value.
    '''

    def __init__(self, expression: LogicalExpression, value: int) -> None:
        '''Initialize an aggregate constraint.

        Args:
            expression (LogicalExpression): A logical expression to aggregate over.
            value (int): The target sum value for cells satisfying the expression.
        '''
        self.expression = expression
        self.value = value


class SumEqual(AggregateConstraint):
    '''Represents a sum equality constraint: Sum(expression) == value'''

    def to_sparse_constraint(self, domain) -> SparseConstraint:
        reduced_mask = self.expression.reduce(domain)
        indices = np.asarray(np.flatnonzero(reduced_mask), dtype=np.uint32)
        coefs = np.ones_like(indices, dtype=np.uint8)
        return SparseConstraint(indices=indices, coefs=coefs, sense='=', rhs=float(self.value))

# NOTE: Add more aggregate expressions as needed
