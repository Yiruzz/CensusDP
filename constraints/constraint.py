from abc import ABC, abstractmethod
from typing import Callable

class Constraint(ABC):
    """Base interface for all constraints.

    Implementations must provide to_constraint(domain) which returns a callable
    constraint function that receives a contingency variable and returns a boolean
    expression that will be used by the optimizer.
    """

    @abstractmethod
    def to_constraint(self, domain) -> Callable:
        """Convert the constraint into a callable function.

        Args:
            domain: ContingencyDomain used as the cell space for evaluation.
        Returns:
            Callable: A function that takes a contingency variable and returns a boolean
                      expression that will be used by the optimizer.
        """
        raise NotImplementedError()