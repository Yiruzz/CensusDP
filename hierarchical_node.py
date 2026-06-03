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
    def __init__(self, geo_id: int, level: int,  constraints: List[Callable] = None) -> None:
        """
        Initialize a hierarchical node.

        Args:
            geo_id (int): Identifier related to geography.
            level (int): Level where the node is located.
            constraints (List[Callable]): Optional list of constraints for this node.

        Attributes:
            id (Optional[int]): Unique identifier for the node.
            geo_id (int): Identifier related to geography.

            children (List[HierarchicalNode]): List of child nodes.
            parent (HierarchicalNode): Reference to the parent node.

            hierarchical_path (List[Any]): List of hierarchical geo_ids visited to reach this node from the root.
                Example: [] for root, ['A'] for region A, ['A', 'A1'] for region A + comuna A1.
                Used to filter the dataframe to get this node's data subset.
            level (int): Level where the node is located.

            contingency_vector (np.ndarray): Single slot holding the node's current data (computed lazily).
            constraints (List[Callable]): List of constraints for this node.
            comparative_vector (np.ndarray): Optional vector for this node. Used to compare distributions.
        """
        self.id: Optional[int] = None
        self.geo_id: int = geo_id

        self.children: List[HierarchicalNode] = []
        self.parent: Optional[HierarchicalNode] = None

        self.hierarchical_path: List[Any] = []
        self.level: int = level

        self.contingency_vector: np.ndarray = np.array([])
        self.constraints: List[Callable] = constraints if constraints is not None else []
        self.comparative_vector: Optional[np.ndarray] = None

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
    
    def __repr__(self) -> str:
        '''String representation of the node.'''
        return f"HierarchicalNode(id={self.id}, level={self.level}, children={len(self.children)})"