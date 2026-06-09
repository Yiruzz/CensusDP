from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SparseConstraint:
    '''Sparse linear constraint representation for LP/optimization.

    Encodes a linear constraint over contingency cells in a compact format
    suitable for writing to LP files and passing to optimization solvers.

    Attributes:
        indices (np.ndarray): Cell indices involved in the constraint (shape: (n,)).
        coefs (np.ndarray): Numerical coefficients accompanying those indices (shape: (n,)).
        sense (str): Relation operator: "=", "<=", or ">=".
        rhs (float): Right-hand side value (the constraint equals/relates to this value).
    '''
    indices: np.ndarray
    coefs: np.ndarray
    sense: str
    rhs: float

    def offset(self, k: int) -> "SparseConstraint":
        '''Adjust all indices by an offset k (for combining child constraints).

        Used when concatenating contingency vectors from multiple children.
        Each child's constraint indices are shifted by the cumulative offset.

        Args:
            k (int): Offset to add to all indices.

        Returns:
            SparseConstraint: New SparseConstraint with adjusted indices (indices + k).
        '''
        return SparseConstraint(indices=self.indices + int(k), coefs=self.coefs, sense=self.sense, rhs=self.rhs)


class Constraint(ABC):
    '''Base interface for all constraints.

    Implementations must provide `to_sparse_constraint(domain)` which returns a
    SparseConstraint consumed by the optimizer when it writes the LP file.
    '''

    @abstractmethod
    def to_sparse_constraint(self, domain) -> SparseConstraint:
        '''Convert the constraint into a SparseConstraint over the given domain.

        Args:
            domain: ContingencyDomain used as the cell space for evaluation.

        Returns:
            SparseConstraint: Sparse linear representation consumed by the optimizer.
        '''
        raise NotImplementedError()