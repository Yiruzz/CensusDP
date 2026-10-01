import numpy as np
from typing import Callable, Optional

from constraints.sparse_constraint import SparseConstraint
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
    def __init__(self, expression: LogicalExpression, aggregation_function: Callable[[np.ndarray], int]) -> None:
        """
        Constructor of a ContextualAggregateConstraint.
        Args:
            expression (LogicalExpression): A logical expression to aggregate over.
            aggregation_function (Callable[[np.ndarray], int], optional): A function
                to calculate the value dynamically using the node's counts array. Defaults to None.
        """
        # We initialize the base AggregateConstraint with a placeholder value (-1).
        # It will be changed later when we apply the aggregation function at runtime.
        super().__init__(expression=expression, value=-1)
        self.aggregation_function = aggregation_function
        
    def apply_aggregation_function(self, counts: np.ndarray, filter_dict: Optional[dict] = None) -> int:
        """Calculate the value for the aggregate expression.

        Args:
            counts (np.ndarray): The counts array to use for calculation.
            filter_dict (Optional[dict]): The node's geography, for the subclasses whose value
                depends on which node it is rather than on its counts (SumBoundOfNode).
        Returns:
            int: The calculated value.
        """
        if self.aggregation_function is not None:
            self.value = self.aggregation_function(counts)
            return self.value
        else:
            raise ValueError("No aggregation_function provided to compute the value.")
        

class SumEqualRealTotal(ContextualAggregateConstraint):
    """Convenience class for the user to easily set the Real Total constraint."""

    @staticmethod
    def get_real_total(counts):
        # This is because we can now get a count over all possible combinations for a node using group by and size.
        # So there is no need to use len() over the DataFrame; we can simply sum the Series or list of counts computed previously.
        return counts.sum()
    
    def __init__(self, expression: LogicalExpression) -> None:

        # The true total is the sum of all counts in the node's context
        super().__init__(expression=expression, aggregation_function=SumEqualRealTotal.get_real_total)

    def to_sparse_constraint(self, domain) -> SparseConstraint:
        '''Convert to sparse linear constraint (agnóstico).

        Returns a SparseConstraint representation of the contextual sum constraint.
        '''
        # Reduce to a boolean mask over the cells, then take the selected indices.
        reduced_mask = self.expression.reduce(domain)
        indices = np.flatnonzero(reduced_mask)
        coefs = np.ones(len(indices))

        return SparseConstraint(
            indices=indices,
            coefs=coefs,
            sense="=",
            rhs=float(self.value)
        )
    
class SumBoundOfNode(ContextualAggregateConstraint):
    """sum(x[mask]) <sense> value, where the value is read off the node's geography, not its counts.

    Args:
        expression (LogicalExpression): The cells the sum runs over.
        sense (str): "=", "<=" or ">=". Fixed per constraint: the factored pipeline compiles the
            row once per bag and reuses its sense for every node.
        values (dict): The value per node, keyed by the tuple of its filter_dict values in hierarchy
            order: () for the root, (state,) for a state, and so on.
    """

    def __init__(self, expression: LogicalExpression, sense: str, values: dict) -> None:
        super().__init__(expression=expression, aggregation_function=None)
        self.sense = sense
        self.values = values

    def apply_aggregation_function(self, counts: np.ndarray, filter_dict: Optional[dict] = None) -> int:
        self.value = self.values[tuple(filter_dict.values())]
        return self.value

    def to_sparse_constraint(self, domain) -> SparseConstraint:
        indices = np.flatnonzero(self.expression.reduce(domain))
        return SparseConstraint(indices=indices, coefs=np.ones(len(indices)), sense=self.sense,
                                rhs=float(self.value))


# NOTE: Add more aggregate expressions as needed

