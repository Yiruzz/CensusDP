import numpy as np

from typing import Callable, List, Optional, Any

class HierarchicalNode:
    '''Represents a node in a hierarchical tree structure.

    Each node contains filter information for accessing the subset of data it represents,
    and a list of its children. The actual contingency vectors and constraints are computed
    lazily when needed, not at tree construction time.

    This class focuses solely on node specific data and operations,
    without any tree traversal or tree-wide operation logic.
    '''
    def __init__(self, geo_id: int, level: int) -> None:
        """
        Initialize a hierarchical node.

        Args:
            geo_id (int): Identifier related to geography.
            level (int): Level where the node is located.

        Attributes:
            geo_id (int): Identifier related to geography.

            children (List[HierarchicalNode]): List of child nodes.
            parent (HierarchicalNode): Reference to the parent node.

            hierarchical_path (List[Any]): List of hierarchical geo_ids visited to reach this node from the root.
                                           Example: [0] for root, [0, 'A'] for region A, [0, 'A', 'A1'] for region A + comuna A1.
                                           Used to filter the dataframe to get this node's data subset.
            level (int): Level where the node is located.

            contingency_vector (Optional[np.ndarray]): Node's contingency vector, None when not materialized or freed.
            constraints (List[Callable]): List of constraints for this node.
        """
        self.geo_id: int = geo_id

        self.children: List[HierarchicalNode] = []
        self.parent: Optional[HierarchicalNode] = None

        self.hierarchical_path: List[int] = [0]
        self.level: int = level

        self.contingency_vector: Optional[np.ndarray] = None
        self.constraints: List[Callable] = []

    def add_child(self, child_node: 'HierarchicalNode') -> None:
        '''Add a child node to this node.

        The child's hierarchical_path is built by appending its own geo_id to parent's path.

        Args:
            child_node (HierarchicalNode): The child node to add.
        '''
        child_node.parent = self
        child_node.hierarchical_path = self.hierarchical_path + [child_node.geo_id]
        self.children.append(child_node)

    def is_root(self) -> bool:
        '''Check if the node is the root (no parent).
        
        Returns:
            bool: True if the node is the root, False otherwise.
        '''
        return self.parent is None
    
    def is_leaf(self) -> bool:
        '''Check if the node is a leaf (no children).

        Returns:
            bool: True if the node is a leaf, False otherwise.
        '''
        return len(self.children) == 0
    
    def __str__(self) -> str:
        """Return a detailed string representation of the node with key attributes."""
        has_contingency = self.contingency_vector is not None and len(self.contingency_vector) > 0
        has_constraints = len(self.constraints) > 0
        path = " -> ".join(str(x) for x in self.hierarchical_path)

        result = "--- HierarchicalNode ---\n"
        result += f"Geo ID: {self.geo_id}\n"
        result += f"Nivel: {self.level}\n"
        result += f"Ruta jerárquica: {path}\n"
        result += f"Cantidad de hijos: {len(self.children)}\n"
        result += f"Vector de contingencia: {has_contingency}\n"
        result += f"Constraints: {has_constraints}\n"

        return result.strip()