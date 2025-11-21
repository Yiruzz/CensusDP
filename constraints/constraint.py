from abc import ABC, abstractmethod
import pandas as pd
from typing import Callable

class Constraint(ABC):
    """Base interface for all constraints.

    Implementations must provide `to_constraint(contingency_df)` which returns a
    callable constraint function.
    """

    @abstractmethod
    def to_constraint(self, contingency_df: pd.DataFrame) -> Callable:
        """Convert the constraint into a callable function.

        Args:
            contingency_df (pd.DataFrame): Pandas DataFrame used as the domain for evaluation.
        Returns:
            Callable: A function that takes a pd.Series and returns a boolean Series.
        """
        raise NotImplementedError()