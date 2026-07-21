"""Marginal-selection heuristics.

A heuristic decides which marginals (cliques of attributes) to measure. It
returns a list of cliques that are then embedded in the interaction graph and
triangulated into bags by :class:`~graph.junction_tree.JunctionTree`.

The first, simplest strategy (heuristic 1) is a 2-way maximum spanning tree
weighted by mutual information, unioned with the mandatory cliques coming from
the constraints. The interface is intentionally small so richer strategies
(density growth, expandable marginals) can drop in later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import FrozenSet, Iterable, List, Sequence

import numpy as np


class _UnionFind:
    """Disjoint-set over integer ids (path-compressed)."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            # Path compression for more efficient future finds.
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Merge the sets of a and b; return True iff they were different."""
        ra, rb = self.find(a), self.find(b) # Roots
        if ra == rb: # Same root case, already connected components
            return False
        self._parent[ra] = rb # Merge the two components (root of a points to root of b)
        return True


class MarginalSelectionStrategy(ABC):
    """Selects the cliques (marginals) to embed in the interaction graph."""

    @abstractmethod
    def select(self, columns: Sequence[str], association: np.ndarray,
               mandatory_cliques: Iterable[Iterable[str]]) -> List[FrozenSet[str]]:
        """Return the cliques to embed.

        Args:
            columns: All attribute columns.
            association: Symmetric pairwise association matrix aligned to
                columns (e.g. mutual information).
            mandatory_cliques: Constraint scopes that must each be contained in
                some bag.

        Returns:
            List of cliques (column subsets). Always includes the mandatory ones.
        """
        raise NotImplementedError


class MaxSpanningTreeMI(MarginalSelectionStrategy):
    """Heuristic 1: 2-way marginals from a max spanning tree over MI that
    contains the mandatory constraint cliques.

    The constraint cliques are forced into the structure first (their columns are
    pre-merged into connected components). A maximum spanning tree is then grown
    with Kruskal over the mutual-information edges, adding a 2-way marginal only
    when it connects two columns not already joined — by the constraints or by a
    previously chosen edge. This means an MI edge is never added inside a
    component the constraints already connect, so no redundant marginals are
    selected. The result is a maximum-weight spanning tree constrained to contain
    the constraint cliques.
    """

    def select(self, columns: Sequence[str], association: np.ndarray,
               mandatory_cliques: Iterable[Iterable[str]]) -> List[FrozenSet[str]]:
        columns = list(columns)
        index = {c: i for i, c in enumerate(columns)}

        mandatory = [frozenset(mc) for mc in mandatory_cliques if mc]
        cliques: List[FrozenSet[str]] = list(mandatory)

        components = _UnionFind(len(columns))
        # Force the constraint cliques: all columns of a clique share a component,
        # so MI edges will never be added within an already-constrained group.
        for clique in mandatory:
            members = [index[c] for c in clique if c in index]
            for other in members[1:]:
                components.union(members[0], other)

        # Kruskal over MI edges, heaviest first; keep only inter-component edges.
        edges = [
            (float(association[i, j]), i, j)
            for i in range(len(columns))
            for j in range(i + 1, len(columns))
        ]
        edges.sort(reverse=True)
        for _weight, i, j in edges:
            if components.union(i, j):
                cliques.append(frozenset((columns[i], columns[j])))

        return cliques


class UnconstrainedMaxSpanningTreeMI(MarginalSelectionStrategy):
    """Baseline heuristic: MST over MI computed independently of the constraints.

    Builds the full maximum spanning tree over all columns by mutual information
    (n-1 edges), then unions the mandatory constraint cliques on top. Unlike
    MaxSpanningTreeMI, the spanning tree does not know about the constraints, 
    so it may pick 2-way edges that duplicate or cross constraint cliques.

    Triangulation later absorbs any edge that ends up contained in a clique, but
    an edge that adds a new chord across a constraint cycle can still enlarge a
    bag (raise the treewidth), so this baseline is expected to be no better — and
    sometimes worse — than the constraint-aware version. Kept for empirical
    comparison of the two selection strategies.
    """

    def select(self, columns: Sequence[str], association: np.ndarray,
               mandatory_cliques: Iterable[Iterable[str]]) -> List[FrozenSet[str]]:
        columns = list(columns)
        cliques: List[FrozenSet[str]] = [frozenset(mc) for mc in mandatory_cliques if mc]

        # Plain Kruskal maximum spanning tree; the constraints do not steer it.
        components = _UnionFind(len(columns))
        edges = [
            (float(association[i, j]), i, j)
            for i in range(len(columns))
            for j in range(i + 1, len(columns))
        ]
        edges.sort(reverse=True)
        for _weight, i, j in edges:
            if components.union(i, j):
                cliques.append(frozenset((columns[i], columns[j])))

        return cliques
