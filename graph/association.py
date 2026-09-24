"""Pairwise association between attribute columns.

The marginal-selection heuristics weight the interaction graph by pairwise
association. The standard, type-agnostic choice is mutual information.

Association is estimated from privately measured 2-way marginals: the caller
supplies a measure_pair callable that returns the (noisy) joint count table
for a pair of columns, having spent privacy budget via the configured mechanism.
This keeps the graph package free of any data / DuckDB / privacy imports while
ensuring selection never touches raw data.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

# Given two column names, return their noisy joint
# count table as a 2D array of shape (|dom(a)|, |dom(b)|).
PairCounts = Callable[[str, str], np.ndarray]


def mutual_information(joint_counts: np.ndarray) -> float:
    """Mutual information (nats) of a joint count table.

    Args:
        joint_counts: 2D array of non-negative counts (|A| x |B|).

    Returns:
        I(A; B) >= 0 in nats; 0 if the table is empty.
    """
    counts = np.asarray(joint_counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0: # Empty table or all-zero counts case
        return 0.0

    # Joint probability
    p = counts / total
    # Probability of just A and just B (1-way marginals)
    p_a = p.sum(axis=1, keepdims=True)
    p_b = p.sum(axis=0, keepdims=True)

    # non-zero mask
    nz = p > 0
    outer = p_a @ p_b  # p(a) p(b)
    # Compute mutual information
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p[nz] * np.log(p[nz] / outer[nz])
    return float(max(0.0, terms.sum()))


class PairwiseAssociation:
    """Builds a symmetric pairwise mutual-information matrix over columns."""

    def compute(self, columns: Sequence[str], measure_pair: PairCounts) -> np.ndarray:
        """Return the symmetric MI matrix for columns.

        Args:
            columns: Attribute columns to score.
            measure_pair: Callable returning the (private) joint count table for
                a pair of columns. Counts are clamped to be non-negative before
                the MI is computed (discrete-Gaussian noise can make them
                negative).

        Returns:
            (n x n) symmetric float matrix; the diagonal is 0.
        """
        columns = list(columns)
        n = len(columns)
        mi = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                # Clip negative counts to 0 before computing MI. Can happend because of DP noise.
                counts = np.clip(np.asarray(measure_pair(columns[i], columns[j])), 0, None)
                value = mutual_information(counts)
                mi[i, j] = mi[j, i] = value
        return mi