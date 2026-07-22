from abc import ABC, abstractmethod
from typing import FrozenSet
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

    @abstractmethod
    def scope(self) -> FrozenSet[str]:
        """Return the set of attribute columns this constraint references.

        Used to seed the junction tree: a constraint can only be enforced within
        a bag (marginal) whose columns contain its whole scope, so each scope
        becomes a mandatory clique of the interaction graph.

        Returns:
            frozenset[str]: The referenced query columns (possibly empty).
        """
        raise NotImplementedError()