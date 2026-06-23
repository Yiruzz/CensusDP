from collections import deque
from hierarchical_node import HierarchicalNode

from typing import Generator

class HierarchicalTree:
    r'''
    Represents a hierarchical tree structure. Each node is a HierarchicalNode.

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
    def __init__(self) -> None:
        '''
        Initialize the hierarchical tree with a root node.

        Attributes:
            root (HierarchicalNode): Reference to the root node.
            _node_count (int): Total number of nodes in the tree.
            _levels (int): Total number of levels in the tree.
        '''
        self.root = HierarchicalNode(level=0, filter_dict={})
        self._node_count = 1
        self._levels = 1

    def __str__(self) -> str:
        '''
        Returns a string representation of the tree with statistics.

        Returns:
            str: Information about total nodes, levels, and nodes per level.
        '''
        nodes_per_level = self._count_nodes_per_level()

        result = "--- HierarchicalTree ---\n"
        result += f"Total nodes: {self._node_count}\n"
        result += f"Total levels: {self._levels}\n"
        result += "Nodes per level:\n"

        for level, count in sorted(nodes_per_level.items()):
            result += f"  Level {level}: {count} nodes\n"

        return result.strip()

    def _count_nodes_per_level(self) -> dict:
        '''Count the number of nodes at each level of the tree using BFS.

        Returns:
            dict: Dictionary with level as key and node count as value.
        '''
        nodes_by_level = {i: 0 for i in range(self._levels)}
        queue = deque([self.root])

        while queue:
            node = queue.popleft()
            nodes_by_level[node.level] += 1

            if not node.is_leaf():
                for child in node.children:
                    queue.append(child)

        return nodes_by_level

    def _index_nodes(self) -> None:
        '''Assign unique incremental IDs to all nodes via BFS traversal.

        Each node receives an id starting from 0 at the root, incrementing sequentially
        through breadth-first order.
        '''
        node_id = 0
        queue = deque([self.root])

        while queue:
            node = queue.popleft()
            node.id = node_id
            node_id += 1

            if not node.is_leaf():
                for child in node.children:
                    queue.append(child)

    def iter_nodes_with_levels(self) -> Generator[tuple[int, int], None, None]:
        '''Traverse the tree using BFS and yield node ID and level pairs.

        Yields:
            tuple[int, int]: A tuple of (node_id, node_level) for each node.
        '''
        queue = deque([self.root])

        while queue:
            node = queue.popleft()
            yield (node.id, node.level)

            if not node.is_leaf():
                for child in node.children:
                    queue.append(child)

    def print_all_nodes(self) -> None:
        '''Print all nodes in the hierarchical tree using BFS traversal.'''
        print("\n--- Tree Nodes ---\n")
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            print(node)
            print()
            for child in node.children:
                queue.append(child)