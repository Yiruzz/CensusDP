from dataclasses import dataclass
import numpy as np
from typing import Literal, Optional


@dataclass(frozen=True)
class SparseConstraint:
    '''Sparse linear constraint representation for LP/optimization.

    Encodes a linear constraint over contingency cells in a compact format,
    suitable for writing to LP files and passing to optimization solvers.
    Only non-zero terms are stored.

    Attributes:
        indices (np.ndarray): Cell indices involved in the constraint (shape: (n,)).
        sense (str): Relation operator: "=" (equality), "<=" (less-equal), ">=" (greater-equal).
        rhs (float): Right-hand side value (the constraint equals/relates to this value).
    '''
    indices: np.ndarray
    sense: Literal["=", "<=", ">=", "<", ">"]
    rhs: float

    def __post_init__(self):
        '''Validate shapes after initialization.'''
        # Normalize "==" to "=" for internal consistency
        if self.sense == "==":
            object.__setattr__(self, 'sense', "=")
        if self.sense not in ("=", "<=", ">=", "<", ">"):
            raise ValueError(f"sense must be '=', '<=', '>=', '<' or '>'. Got '{self.sense}'")

    def prune_to_active_space(self, offset: int, active_indices_set: set) -> Optional["SparseConstraint"]:
        """
        Shift local constraint indices into global space and prune inactive variables.
        Args:
            offset (int): Global index offset applied to local indices.
            active_indices_set (set): Set of active global indices allowed in the current optimization. Any index not in this set is discarded.

        Returns:
            Optional[SparseConstraint]: A new SparseConstraint containing only active global indices. Returns None if all indices are pruned.
        """

        if active_indices_set is None:
            return SparseConstraint(
                indices=self.indices+offset,
                sense=self.sense,
                rhs=self.rhs
            )

        global_indices = self.indices + offset
        mask = np.array([i in active_indices_set for i in global_indices])

        pruned_indices = global_indices[mask]

        if len(pruned_indices) == 0:
            return None

        return SparseConstraint(
            indices=pruned_indices,
            sense=self.sense,
            rhs=self.rhs
        )

    def __repr__(self) -> str:
        '''Human-readable representation for debugging.'''
        terms = [f"x[{i}]" for i in self.indices]
        constraint_str = " + ".join(terms)
        return f"{constraint_str} {self.sense} {self.rhs}"