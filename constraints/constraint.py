from abc import ABC, abstractmethod
from constraints.sparse_constraint import SparseConstraint

class Constraint(ABC):
    """Base interface for all constraints.

    Implementations must provide to_constraint(domain), which returns a SparseConstraint.
    It encapsulates the selected indices so that, when the optimizer receives a contingency variable,
    it selects the corresponding variables and uses them to evaluate the expressed condition.
    """

    @abstractmethod
    def to_sparse_constraint(self, domain) -> SparseConstraint:
        """Convert the constraint into a SparseConstraint over the given domain.

        Args:
            domain: ContingencyDomain used as the cell space for evaluation.
        Returns:
            SparseConstraint: Sparse linear representation consumed by the optimizer.
        """
        raise NotImplementedError()