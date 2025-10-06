from hierarchical_node import HierarchicalNode

from collections import deque
from typing import List, Callable, Optional, Tuple, Generator



class HierarchicalTree:
    '''Represents a hierarchical tree structure. Each node is a HierarchicalNode.

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
    def __init__(self, root_id: int = 0, constraints: Optional[List] = None) -> None:
        """
        Initialize the hierarchical tree with a root node.
        
        Args:
            root_id: ID for the root node
            constraints: Optional constraints for the root node

        Attributes:
            root (HierarchicalNode): The root node of the tree
            _node_count (int): Internal counter to keep track of the number of nodes in the tree
        """
        self.root = HierarchicalNode(root_id, constraints)
        self._node_count = 1
    
    def iterate_by_levels(self) -> Generator[Tuple[int, List[HierarchicalNode]], None, None]:
        """
        Iterate over the tree level by level using BFS.
        
        Yields:
            Tuples of (level, list of nodes at that level)
        """
        if not self.root:
            raise ValueError("The tree has no root node.")

        queue = deque([(self.root, 0)])
        current_level = 0
        level_nodes = []

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

    def apply(self, operation: Callable[[HierarchicalNode], None]) -> None:
        """
        Apply a function to each node in the tree considering BFS traversal.
        
        Args:
            operation: Function that takes a HierarchicalNode as input and returns None
        """
        bfs_traversal = self.iterate_by_levels()
        for _, nodes in bfs_traversal:
            for node in nodes:
                operation(node)
    
