import numpy as np

from typing import List, Optional, Any

class HierarchicalNode:
    '''Represents a node in a hierarchical tree structure.
    
    Each node contains an array with the data and a list of its children.

    This class focuses solely on node specific data and operations,
    without any tree traversal or tree-wide operation logic.
    '''
    def __init__(self, node_id: int, constraints: Optional[List] = None) -> None:
        """
        Initialize a hierarchical node.
        
        Args:
            node_id (int): Unique identifier for this hierarchical_path entity
            constraints (List[Callable]): Optional list of constraints for this node
        
        Attributes:
            id (int): Unique identifier for the node
            children (List[HierarchicalNode]): List of child nodes
            parent (HierarchicalNode): Reference to the parent node
            hierarchical_path (List[Any]): List of hierarchical nodes visited to reach this node from the root

            contingency_vector (np.ndarray): Contingency vector for this node.
            constraints (List[Callable]): List of constraints for this node.
                                          The constraints are functions that take a contingency vector as input
                                          and return a boolean indicating whether the constraint is satisfied.
            
            comparative_vector (np.ndarray): Optional vector for this node. Used to compare distributions.
        """
        self.id: int = node_id

        self.children: List['HierarchicalNode'] = []
        self.parent: Optional['HierarchicalNode'] = None

        # The path is needed to save runtime when generating data from a specific node
        self.hierarchical_path: List[Any] = []
        
        # Data containers
        self.contingency_vector: Optional[np.ndarray] = None
        self.constraints = constraints or []
        
        # NOTE: This value is only used to compare distributions between different states of the data
        # e.g., original data vs noisy data
        # It is only relevant when a distance metric is defined by the user
        self.comparative_vector: Optional[np.ndarray] = None

    def add_child(self, child_node: 'HierarchicalNode') -> None:
        """
        Add a child node to this node.
        
        Args:
            child_node (HierarchicalNode): The child node to add.
        """
        child_node.parent = self
        self.children.append(child_node)

    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf (no children).
        
        Returns:
            bool: True if the node is a leaf, False otherwise.
        """
        return len(self.children) == 0
    
    def get_level(self) -> int:
        """
        Get the level of this node in the tree.
        
        Returns:
            int: The level of the node (0 for root, 1 for children of root, ...).
        """
        return len(self.hierarchical_path)
    
    def is_root(self) -> bool:
        """
        Check if the node is the root (no parent).
        
        Returns:
            bool: True if the node is the root, False otherwise.
        """
        return self.parent is None
    
    def __repr__(self) -> str:
        '''String representation of the node.'''
        return f"HierarchicalNode(id={self.id}, level={self.get_level()}, children={len(self.children)})"