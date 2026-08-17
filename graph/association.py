"""Pairwise association between attribute columns.

The marginal-selection heuristics weight the interaction graph by pairwise
association. Two statistics are offered, and which one a heuristic wants is
declared by MarginalSelectionStrategy.pair_statistic:

  mutual_information          I(A; B) in nats. Type-agnostic and scale-free, which
                              is also its weakness for selection: it says nothing
                              about how many records the pair misplaces, so it
                              cannot be traded against a cost measured in records.
  independence_residual_l1    ||x - N p_a (x) p_b||_1, in RECORDS. The L1 error the
                              independent model already commits on that pair, i.e.
                              exactly what measuring the pair would buy back. This
                              is the criterion McKenna's MST and AIM use, and it is
                              in the same units as the published utility metric.

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

# A per-pair statistic: joint count table -> scalar weight.
PairStatistic = Callable[[np.ndarray], float]


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


def independence_residual_l1(joint_counts: np.ndarray) -> float:
    """L1 distance between a joint count table and its independent product, in records.

    ``||x - N p_a (x) p_b||_1``: how many records the independent model misplaces on this
    pair, and therefore how many measuring the pair could put back. It is the same
    quantity McKenna scores edges by (mst.py's ``norm(x - xhat, 1)``, with the reference
    model fitted to the 1-way marginals only), and it lands in the units of the published
    utility metric - unlike mutual information, which is in nats and cannot be traded
    against a cost counted in records.

    No bias correction is applied here: on a noisy table the value is inflated by roughly
    E|noise| per cell, and subtracting that needs the noise scale, which this package
    deliberately does not know. The caller (a cost-aware heuristic) does the correction.

    Args:
        joint_counts: 2D array of non-negative counts (|A| x |B|).

    Returns:
        The L1 residual in records; 0 if the table is empty.
    """
    counts = np.asarray(joint_counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:  # Empty table or all-zero counts case
        return 0.0

    # Independent model with the SAME total, so the residual is a distance between two
    # tables of equal mass and not a mixture of shape and scale error.
    row = counts.sum(axis=1, keepdims=True)
    col = counts.sum(axis=0, keepdims=True)
    independent = (row @ col) / total
    return float(np.abs(counts - independent).sum())


class PairwiseAssociation:
    """Builds a symmetric pairwise association matrix over columns.

    Attributes:
        statistic: The per-pair statistic applied to each noisy joint table. Defaults to
            mutual_information, which is what the two spanning-tree heuristics want; the
            cost-aware heuristic asks for independence_residual_l1 instead.
    """

    def __init__(self, statistic: PairStatistic = mutual_information) -> None:
        self.statistic: PairStatistic = statistic

    def compute(self, columns: Sequence[str], measure_pair: PairCounts) -> np.ndarray:
        """Return the symmetric association matrix for columns.

        Args:
            columns: Attribute columns to score.
            measure_pair: Callable returning the (private) joint count table for
                a pair of columns. Counts are clamped to be non-negative before
                the statistic is computed (discrete-Gaussian noise can make them
                negative).

        Returns:
            (n x n) symmetric float matrix; the diagonal is 0.
        """
        columns = list(columns)
        n = len(columns)
        weights = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                # Clip negative counts to 0 first. Can happen because of DP noise.
                counts = np.clip(np.asarray(measure_pair(columns[i], columns[j])), 0, None)
                value = self.statistic(counts)
                weights[i, j] = weights[j, i] = value
        return weights