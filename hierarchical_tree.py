import numpy as np

from hierarchical_node import HierarchicalNode

from collections import deque
from typing import List, Callable, Tuple, Generator

class HierarchicalTree:
    r'''Represents a hierarchical tree structure. Each node is a HierarchicalNode.

    This class is responsible for tree structure management, traversal,
    and operations that need to be applied across multiple nodes.

    Example of a hierarchical tree:
    
    C = Country
    S = State

        C
       / \
      S1  S2
     /|\   \
     
    '''
    def __init__(self, root_id: int = 0, level: int = 0, constraints: list[Callable] = []) -> None:
        """
        Initialize the hierarchical tree with a root node.
        
        Args:
            root_id (int): ID for the root node.
            level (int): Level of the root node.
            constraints (int): Optional constraints for the root node.

        Attributes:
            nodes (List[HierarchicalNode]): The nodes of the tree.
            _node_count (int): Number of nodes in the tree.
            _levels (List[int]): List where each index represents a level, and the value indicates the node index where that level starts.
            _contingency_vectors_shm (Optional[str]): Name of the shared memory buffer where contingency vectors are stored.
        """
        self.nodes = [HierarchicalNode(geo_id=root_id, level=level, constraints=constraints)]
        self._node_count = 1
        self._levels = [level]
        self._contingency_vectors_shm = None

        # Logical sizes of the per-node contingency vector across pipeline stages.
        # _contingency_vectors rows are physically sized to vector_length = max(n_queries, n_cells),
        # so a single allocation accommodates both the noisy measurement y (length n_queries) and
        # the estimated cell counts x_hat (length n_cells). Set by DataHandler.build_hierarchical_tree.
        self.n_queries: int = 0
        self.n_cells: int = 0
        self.vector_length: int = 0

    
    def iterate_by_levels(self) -> Generator[Tuple[int, List[HierarchicalNode]], None, None]:
        """
        Iterate over the tree level by level using BFS.
        
        Yields:
            Tuples of (level, list of nodes at that level)
        """
        queue = deque([(self.nodes[0], 0)])
        current_level = 0
        level_nodes: List[HierarchicalNode] = []

        while queue:
            node, level = queue.popleft()

            # When we reach a new level, yield the previous level's nodes
            if level != current_level:
                yield current_level, level_nodes
                current_level = level
                level_nodes = []

            level_nodes.append(node)

            for child in node.children:
                queue.append((child, level + 1))

        if level_nodes:
            yield current_level, level_nodes

    def apply(self, operation: Callable[[HierarchicalNode], None], level: int = None) -> None:
        """
        Apply a function to each node in the tree considering BFS traversal.
        
        Args:
            operation: Function that takes a HierarchicalNode as input and returns None
            level: 
        """
        for node in self.nodes:
            operation(node)
    
