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
        coefs (np.ndarray): Numerical coefficients accompanying those indices (shape: (n,)).
            For sum constraints, all coefs are typically 1.0.
        sense (str): Relation operator: "=" (equality), "<=" (less-equal), ">=" (greater-equal).
        rhs (float): Right-hand side value (the constraint equals/relates to this value).
    '''
    indices: np.ndarray
    coefs: np.ndarray
    sense: Literal["=", "<=", ">="]
    rhs: float

    def __post_init__(self):
        '''Validate shapes after initialization.'''
        if self.indices.shape[0] != self.coefs.shape[0]:
            raise ValueError(f"indices and coefs must have same length. Got {self.indices.shape[0]} vs {self.coefs.shape[0]}")
        # Normalize "==" to "=" for internal consistency
        if self.sense == "==":
            object.__setattr__(self, 'sense', "=")
        if self.sense not in ("=", "<=", ">="):
            raise ValueError(f"sense must be '=', '<=', or '>='. Got '{self.sense}'")

    def prune_to_active_space(self, offset: int, active_mask: np.ndarray) -> Optional["SparseConstraint"]:
        """
        Shift local constraint indices into global space and prune inactive variables.

        The mask is indexed by the local position, not the global one, and that is what makes
        this cheap. The active set is always {k * width + p : p in support} - the same support
        for every child block k - so whether a position survives depends only on p, never on
        k.

        Args:
            offset (int): Global index offset applied to the surviving local indices.
            active_mask (np.ndarray): Boolean array of length width (the per-node space).
                Entry p is True when position p is active. Positions where it is False are
                dropped from the row.

        Returns:
            Optional[SparseConstraint]: A new SparseConstraint containing only active global
                indices. Returns None if all indices are pruned.
        """
        keep = active_mask[self.indices]
        if not keep.any():
            return None

        return SparseConstraint(
            indices=self.indices[keep] + offset,
            coefs=self.coefs[keep],
            sense=self.sense,
            rhs=self.rhs
        )

    def __repr__(self) -> str:
        '''Human-readable representation for debugging.'''
        terms = [f"{c:.4g}*x[{i}]" for i, c in zip(self.indices, self.coefs)]
        constraint_str = " + ".join(terms)
        return f"{constraint_str} {self.sense} {self.rhs}"