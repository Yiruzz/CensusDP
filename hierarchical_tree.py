from collections import deque
from hierarchical_node import HierarchicalNode

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
    def __init__(self, root_id: int = 0, level: int = 0) -> None:
        '''
        Initialize the hierarchical tree with a root node.

        Args:
            root_id (int): ID for the root node.
            level (int): Level of the root node.

        Attributes:
            root (HierarchicalNode): Reference to the root node.
            _node_count (int): Total number of nodes in the tree.
            _levels (int): Total number of levels in the tree.
        '''
        self.root = HierarchicalNode(geo_id=root_id, level=level)
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
        result += f"Cantidad de nodos: {self._node_count}\n"
        result += f"Cantidad de niveles: {self._levels}\n"
        result += "Nodos por nivel:\n"

        for level, count in sorted(nodes_per_level.items()):
            result += f"  Nivel {level}: {count} nodos\n"

        return result.strip()

    def _count_nodes_per_level(self) -> dict:
        '''
        Count the number of nodes at each level of the tree using BFS.

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

    def print_all_nodes(self) -> None:
        '''Print all nodes in the hierarchical tree using BFS traversal.'''
        print("\n--- Nodos del árbol ---\n")
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            print(node)
            print()
            for child in node.children:
                queue.append(child)