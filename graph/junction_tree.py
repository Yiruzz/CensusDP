"""Junction-tree construction over attribute columns.

Pure structural logic (networkx-backed), independent of the DP pipeline, the
solver, and the data. The flow mirrors the classic recipe:

    interaction graph  ->  triangulation (chordal completion)
                       ->  maximal cliques  = bags (the marginals to measure)
                       ->  maximum-weight spanning tree over bags (weight =
                           separator size) which guarantees the running-
                           intersection property (RIP).

The resulting JunctionTree exposes the bags, the tree adjacency, the separators, 
and a root-to-leaf traversal order used later by the measurement, estimation, 
and microdata phases.
"""
from __future__ import annotations

from collections import deque
from math import prod
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx

# A bag / clique is a tuple of column names in canonical (global) order.
Bag = Tuple[str, ...]


def _min_weight_chordal(graph: nx.Graph, weights: Mapping[str, int]) -> nx.Graph:
    """Triangulate a graph with a cardinality aware elimination order.

    A junction tree needs a chordal graph.
    It functions as follows: pick an order and, for each vertex, connect its still-present 
    neighbours into a clique before removing it. Any order yields a chordal graph the order
    only decides how big the resulting cliques are.

    networkx's default order (``complete_to_chordal_graph``) is blind to domain sizes, so
    it can eliminate a high-cardinality column early and fuse it into a huge clique. The
    *min-weight* heuristic here instead removes, at each step, the vertex whose clique
    (itself + its neighbours) has the smallest product of cardinalities - the smallest
    marginal to build. It minimises the bags cell count, not the edge count, and can cut
    the width by orders of magnitude with no change to which constraints are enforceable.

    Args:
        graph: The interaction graph (a copy is triangulated; the original is untouched).
        weights: Column -> domain cardinality.

    Returns:
        A chordal supergraph of ``graph`` (its edges plus the fill edges added).
    """
    # remaining is the shrinking elimination graph 
    # chordal keeps every vertex and accumulates the needed fill edges.
    remaining = graph.copy()
    chordal = graph.copy()
    while remaining.number_of_nodes():
        # Cheapest clique right now = vertex with the smallest (itself + neighbours) product.
        victim = min(
            remaining.nodes(),
            key=lambda v: prod([weights[v]] + [weights[u] for u in remaining.neighbors(v)]),
        )
        # Make the victim's neighbours a clique: each missing pair is a fill edge (a chord)
        neighbours = list(remaining.neighbors(victim))
        for a in range(len(neighbours)):
            for b in range(a + 1, len(neighbours)):
                if not remaining.has_edge(neighbours[a], neighbours[b]):
                    remaining.add_edge(neighbours[a], neighbours[b])
                    chordal.add_edge(neighbours[a], neighbours[b])
        # Its neighbours are now connected, so the rest stays a valid graph.
        remaining.remove_node(victim)
    return chordal


def _max_weight_spanning_tree(bag_sets: List[frozenset]) -> Dict[int, List[int]]:
    """Maximum-weight spanning tree over the clique-intersection graph.

    Nodes are bag indices; every pair is connected with weight = |Ci ∩ Cj|
    (zero-weight edges included so the result is a single connected tree even
    when the interaction graph is disconnected). A maximum-weight spanning tree
    with separator-size weights satisfies the running-intersection property.
    """
    n = len(bag_sets)
    tree_adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    if n <= 1:
        return tree_adj

    # Build the complete graph over bag indices, weighted by separator size.
    cg = nx.Graph()
    cg.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            cg.add_edge(i, j, weight=len(bag_sets[i] & bag_sets[j]))

    # Compute the maximum-weight spanning tree
    tree = nx.maximum_spanning_tree(cg, weight="weight")
    for i, j in tree.edges():
        tree_adj[i].append(j)
        tree_adj[j].append(i)
    return tree_adj


class JunctionTree:
    """A junction tree of attribute marginals (bags) with separators.

    Attributes:
        columns: All attribute columns, in canonical significance order.
        bags: List of bags (each a tuple of columns in canonical order). These
            are the marginals to measure.
        tree_adj: Undirected adjacency over bag indices.
        root: Index of the root bag (largest bag).
        order: BFS order of bag indices from the root.
        parent: Bag index -> parent bag index (None for the root).
        parent_separator: Bag index -> shared columns with its parent.
    """

    def __init__(self, columns: Sequence[str], bags: List[Iterable[str]],
                 tree_adj: Dict[int, List[int]], root: int) -> None:
        self.columns: List[str] = list(columns)
        self._col_rank: Dict[str, int] = {c: i for i, c in enumerate(self.columns)}
        self.bags: List[Bag] = [self._canon(b) for b in bags]
        self.tree_adj: Dict[int, List[int]] = {
            i: list(tree_adj.get(i, [])) for i in range(len(self.bags))
        }
        self.root: int = root
        self._build_traversal()

    def _canon(self, cols: Iterable[str]) -> Bag:
        """Order a column set by the global column significance order."""
        return tuple(sorted(cols, key=lambda c: self._col_rank[c]))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def build(cls, columns: Sequence[str], cliques: Iterable[Iterable[str]],
              weights: Optional[Mapping[str, int]] = None) -> "JunctionTree":
        """Build a junction tree from a set of cliques over columns.

        Each clique (a mandatory constraint scope or a heuristic-selected
        marginal) is embedded as a connected subgraph of the interaction
        graph. The graph is then triangulated and its maximal cliques become the
        bags.

        Args:
            columns: All attribute columns.
            cliques: Iterable of column subsets to embed as cliques.
            weights: Optional column -> domain cardinality. When given, the graph is
                triangulated with a cardinality-aware (min-weight) elimination order so
                the bags stay small in CELLS, not just in column count. Without it the
                default fill-minimising triangulation is used.

        Returns:
            The constructed JunctionTree.
        """
        columns = list(columns)
        col_set = set(columns)

        graph = nx.Graph()
        graph.add_nodes_from(columns)
        # Add edges for each clique (fully connect the clique)
        for clique in cliques:
            nodes = [c for c in clique if c in col_set]
            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    graph.add_edge(nodes[a], nodes[b])

        # Triangulate the graph and extract the maximal cliques as bags. 
        if weights is not None:
            # The min-weight order keeps the bags small in cells when cardinalities are known.
            chordal = _min_weight_chordal(graph, weights)
        else:
            chordal, _ = nx.complete_to_chordal_graph(graph)
        bags: List[Iterable[str]] = [tuple(cl) for cl in nx.chordal_graph_cliques(chordal)]

        # Every column must appear in at least one bag; isolated columns (no
        # clique, no edge) become their own singleton bag.
        covered = set().union(*[set(b) for b in bags]) if bags else set()
        for c in columns:
            if c not in covered:
                bags.append((c,))

        bag_sets = [frozenset(b) for b in bags]
        tree_adj = _max_weight_spanning_tree(bag_sets)
        root = max(range(len(bags)), key=lambda i: len(bags[i])) if bags else 0
        return cls(columns, bags, tree_adj, root)

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------
    def _build_traversal(self) -> None:
        '''Traverse the tree from the root, recording each bag's parent and the
        shared columns with its parent.'''
        n = len(self.bags)
        self.parent: Dict[int, Optional[int]] = {i: None for i in range(n)}
        self.parent_separator: Dict[int, Bag] = {i: () for i in range(n)}
        self.order: List[int] = []
        if n == 0:
            return

        seen = {self.root}
        queue = deque([self.root])
        while queue:
            i = queue.popleft()
            self.order.append(i)
            for j in self.tree_adj[i]:
                if j not in seen:
                    seen.add(j)
                    self.parent[j] = i
                    self.parent_separator[j] = self._canon(
                        set(self.bags[i]) & set(self.bags[j])
                    )
                    queue.append(j)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def n_bags(self) -> int:
        return len(self.bags)

    def separator(self, i: int, j: int) -> Bag:
        """Shared columns between bags i and j (canonical order)."""
        return self._canon(set(self.bags[i]) & set(self.bags[j]))

    def bag_of_scope(self, scope: Iterable[str]) -> Optional[int]:
        """Index of the first bag whose columns contain scope (or None)."""
        s = set(scope)
        for i, bag in enumerate(self.bags):
            if s <= set(bag):
                return i
        return None

    def edges(self) -> Iterable[Tuple[int, int]]:
        """Yield each tree edge once as an ordered (i, j) pair with i < j."""
        seen = set()
        for i in range(len(self.bags)):
            for j in self.tree_adj[i]:
                edge = (i, j) if i < j else (j, i)
                if edge not in seen:
                    seen.add(edge)
                    yield edge

    def __repr__(self) -> str:
        return f"JunctionTree(bags={self.bags})"
