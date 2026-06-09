import pandas as pd
import numpy as np
from typing import Callable

from constraints.logical_expressions.base import LogicalExpression
from constraints.constraint import SparseConstraint
from .aggregate_constraints import AggregateConstraint

from abc import ABC

class ContextualAggregateConstraint(AggregateConstraint, ABC):
    '''Base class for contextual aggregate constraints.

    A contextual aggregate constraint depends on the data subset (context) of the
    hierarchical node in which it is applied. The constraint value is dynamically
    calculated based on the filtered DataFrame at evaluation time, rather than being
    known in advance.
    '''

    def __init__(self, expression: LogicalExpression, aggregation_function: Callable[[pd.DataFrame], int]) -> None:
        '''Initialize a contextual aggregate constraint.

        Args:
            expression (LogicalExpression): A logical expression to aggregate over.
            aggregation_function (Callable[[pd.DataFrame], int]): A function that takes
                the filtered DataFrame and returns the computed constraint value.
        '''
        # Initialize with placeholder value; will be replaced when aggregation function is applied
        super().__init__(expression=expression, value=-1)
        self.aggregation_function = aggregation_function

    def apply_aggregation_function(self, contextualized_df: pd.DataFrame) -> int:
        '''Compute the constraint value based on the filtered DataFrame.

        Called during data materialization to compute context-dependent constraint values.

        Args:
            contextualized_df (pd.DataFrame): The filtered DataFrame for this node's context.

        Returns:
            int: The computed constraint value.

        Raises:
            ValueError: If no aggregation function was provided.
        '''
        if self.aggregation_function is None:
            raise ValueError("No aggregation_function provided to compute the value.")
        self.value = self.aggregation_function(contextualized_df)
        return self.value


class SumEqualRealTotal(ContextualAggregateConstraint):
    '''Constraint that sums to the total count of records in the node's context.

    Used to enforce that the sum of cells satisfying an expression equals
    the total number of records in the filtered dataset. This is a convenience
    class that pre-defines the aggregation function.
    '''

    def __init__(self, expression: LogicalExpression) -> None:
        '''Initialize the Real Total constraint.

        Args:
            expression (LogicalExpression): The expression whose cells must sum to the total count.
        '''
        super().__init__(expression=expression, aggregation_function=lambda df: len(df))

    def to_sparse_constraint(self, domain) -> SparseConstraint:
        '''Convert to SparseConstraint: sum of selected cells == total record count.

        Args:
            domain: ContingencyDomain used as the cell space for evaluation.

        Returns:
            SparseConstraint: Sparse linear constraint with sense='=' and rhs=self.value,
                      containing indices where the expression is True.
        '''
        reduced_mask = self.expression.reduce(domain)
        indices = np.asarray(np.flatnonzero(reduced_mask), dtype=np.uint32)
        coefs = np.ones_like(indices, dtype=np.uint8)
        return SparseConstraint(indices=indices, coefs=coefs, sense='=', rhs=float(self.value))

# NOTE: Add more aggregate expressions as needed
