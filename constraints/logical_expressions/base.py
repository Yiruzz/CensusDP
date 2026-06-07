import numpy as np
from abc import ABC, abstractmethod

from constraints.constraint import Constraint, SparseRow

class LogicalExpression(Constraint, ABC):
    '''Base class for all logical expressions.

    Implementations must provide `reduce(domain)` which returns a
    boolean numpy array mask over the given domain.

    A standalone logical expression added as a constraint enforces that no cell
    where the expression evaluates to False has positive count. That is encoded
    as a SparseRow: sum_{i in negated} x[i] == 0.
    '''

    @abstractmethod
    def reduce(self, domain) -> np.ndarray:
        '''Reduce the expression to a boolean mask over the given domain.

        Args:
            domain: ContingencyDomain for evaluation.

        Returns:
            np.ndarray: Boolean array indicating which cells satisfy the expression.
        '''
        raise NotImplementedError()

    def to_sparse_row(self, domain) -> SparseRow:
        '''Convert the logical expression to a SparseRow constraint.

        Enforces that cells violating the expression have zero count.

        Args:
            domain: ContingencyDomain for evaluation.

        Returns:
            SparseRow: Sparse constraint with sense='=' and rhs=0.0.
        '''
        negated = ~self.reduce(domain)
        indices = np.asarray(np.flatnonzero(negated), dtype=np.uint32)
        coefs = np.ones_like(indices, dtype=np.uint8)
        return SparseRow(indices=indices, coefs=coefs, sense='=', rhs=0.0)
