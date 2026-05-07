from typing import Dict, List

import gcol
import networkx as nx
import numpy as np


class QueryGraph:
    """Conflict graph of a binary query matrix Q.

    Two queries (rows of Q) conflict iff Q[i] * Q[j] > 0, i.e. they share at
    least one cell. A proper coloring partitions queries into pairwise-disjoint
    color classes; within a class queries have disjoint supports, so the
    differential-privacy budget can be reused across queries by parallel
    composition. Across classes we compose sequentially.

    Notes:
    - gcol.node_coloring is a heuristic upper bound on chi(G). If it returns
      more colors than the true chromatic number, each query gets a smaller
      budget slice -> more noise. This is privacy-safe (never under-pays);
      only utility is affected.
    - In production callers should prefer the static helper
      QueryGraph.compute_num_colors(Q) so the networkx Graph and the
      O(n_queries^2) overlap matrix are released as soon as the call returns.
    """

    def __init__(self, Q: np.ndarray) -> None:
        if Q.ndim != 2:
            raise ValueError(f"Q must be 2-D, got shape {Q.shape}")
        self.Q: np.ndarray = Q
        self.n_queries: int = Q.shape[0]
        self.graph: nx.Graph = self._build_graph(Q)
        
        # strategy in {'random','welsh-powell','dsatur','rlf'}, opt_alg in {None,1..5} for extra reduction.                                                                                
        # Defaults are fine for small Q; tune for larger workloads. See gcol docs. 
        # node_coloring(G, strategy='dsatur', opt_alg=None, it_limit=0)                          
        self.colors: Dict[int, int] = gcol.node_coloring(self.graph) if self.n_queries > 0 else {}
        self.color_groups: List[List[int]] = self._build_color_groups(self.colors, self.n_queries)
        self.num_colors: int = len(self.color_groups)

    @staticmethod
    def _build_graph(Q: np.ndarray) -> nx.Graph:
        '''Build the conflict graph of Q using networkx.'''
        G = nx.Graph()
        n = Q.shape[0]
        G.add_nodes_from(range(n))
        if n <= 1:
            return G
        Qb = (Q != 0).astype(np.int8)
        # Qb @ Qb.T does all operations needed to find disjoint and overlapping query pairs.
        # We only need the upper triangle of the matrix to build the graph, since it's simmetric.
        overlap = Qb @ Qb.T
        iu, ju = np.triu_indices(n, k=1)
        edges = [(int(i), int(j)) for i, j in zip(iu, ju) if overlap[i, j] > 0]
        G.add_edges_from(edges)
        return G

    @staticmethod
    def _build_color_groups(colors: Dict[int, int], n_queries: int) -> List[List[int]]:
        '''Group query indices by color, sorted by minimum query index in each color class.'''
        groups: Dict[int, List[int]] = {}
        for q in range(n_queries):
            groups.setdefault(colors[q], []).append(q)
        return [groups[c] for c in sorted(groups.keys(), key=lambda k: min(groups[k]))]

    def split_budget(self, total_budget: float) -> float:
        """Per-query budget after even split across color classes."""
        if self.num_colors == 0:
            raise ValueError("QueryGraph has zero colors (empty Q?)")
        return total_budget / self.num_colors

    @staticmethod
    def compute_num_colors(Q: np.ndarray) -> int:
        """Build the conflict graph, color it, return num_colors and discard the rest.

        Memory-conscious entry point. The QueryGraph instance and its underlying
        networkx Graph, overlap matrix, colors dict, and color_groups list all
        become unreferenced when this method returns, so the GC can reclaim them.
        Callers retain only the integer.
        """
        return QueryGraph(Q).num_colors

    def __repr__(self) -> str:
        return (
            f"QueryGraph(n_queries={self.n_queries}, "
            f"n_edges={self.graph.number_of_edges()}, "
            f"num_colors={self.num_colors})"
        )
