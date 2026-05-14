from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Callable


@dataclass(frozen=True)
class SparseRow:
    """Sparse representation of a single linear constraint over a contingency vector.

    The constraint is:
        sum_k coefs[k] * x[indices[k]]   <sense>   rhs

    where <sense> is one of '=', '<=', '>='. Variables are referenced by integer
    index into the local contingency vector; the optimizer is responsible for
    mapping those indices to its own variable namespace (e.g. shifting by an
    offset when several children share a joint problem).
    """
    indices: np.ndarray
    coefs: np.ndarray
    sense: str
    rhs: float

    def offset(self, k: int) -> "SparseRow":
        """Return a new row with all indices shifted by k. Used to translate per-child
        rows into the joint contingency vector that combines several children."""
        return SparseRow(indices=self.indices + int(k), coefs=self.coefs, sense=self.sense, rhs=self.rhs)


class Constraint(ABC):
    """Base interface for all constraints.

    Implementations must provide `to_sparse_row(contingency_df)` which returns a
    SparseRow consumed by the optimizer when it writes the LP file.

    `to_constraint(contingency_df)` is kept for backwards compatibility with the
    previous Pyomo-callable contract and can be implemented in terms of
    `to_sparse_row` by subclasses that need it.
    """

    @abstractmethod
    def to_sparse_row(self, contingency_df: pd.DataFrame) -> SparseRow:
        """Convert the constraint into a SparseRow over the given domain.

        Args:
            contingency_df (pd.DataFrame): DataFrame used as the domain for evaluation.
        Returns:
            SparseRow: Sparse linear representation consumed by the optimizer.
        """
        raise NotImplementedError()

    def to_constraint(self, contingency_df: pd.DataFrame) -> Callable:
        """Legacy callable interface kept so older code paths keep importing.
        New code should consume `to_sparse_row` directly."""
        raise NotImplementedError("Callable-based constraints are no longer supported; use to_sparse_row.")
