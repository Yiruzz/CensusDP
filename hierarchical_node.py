import numpy as np

from typing import Callable, List, Optional, Any
from constraints.callable_constraints import AggregateConstraintFunction, SubarrayConstraintFunction

class HierarchicalNode:
    '''Represents a node in a hierarchical tree structure.
    
    Each node contains an array with the data and a list of its children.

    This class focuses solely on node specific data and operations,
    without any tree traversal or tree-wide operation logic.
    '''
    def __init__(self, node_id: int, geo_id: int, constraints: List[Callable]) -> None:
        """
        Initialize a hierarchical node.
        
        Args:
            node_id (int): Unique identifier for this hierarchical node
            geo_id (int): Identifier related to geography
            constraints (List[Callable]): Optional list of constraints for this node
        
        Attributes:
            node_id (int): Unique identifier for the node at its level
            geo_id (int): Identifier related to geography
            children (List[HierarchicalNode]): List of child nodes
            parent (HierarchicalNode): Reference to the parent node
            hierarchical_path (List[Any]): List of hierarchical nodes visited to reach this node from the root

            contingency_vector (np.ndarray): Contingency vector for this node.
            constraints (List[Callable]): List of constraints for this node.
                                          The constraints are functions that take a contingency vector as input
                                          and return a boolean indicating whether the constraint is satisfied.
            
            comparative_vector (np.ndarray): Optional vector for this node. Used to compare distributions.
        """
        self.node_id: int = node_id
        self.geo_id: int = geo_id

        self.children: List['HierarchicalNode'] = []
        self.parent: Optional['HierarchicalNode'] = None

        # The path is needed to save runtime when generating data from a specific node
        self.hierarchical_path: List[Any] = []
        
        # Data containers
        self.contingency_vector: np.ndarray = np.array([])
        self.constraints: List[Callable] = constraints
        
        # This value is only used to compare distributions between different states of the data
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
        child_node.hierarchical_path = self.hierarchical_path + [self.node_id]
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
    
    def combine_child_vectors(self) -> np.ndarray[int]:
        """
        Retrieve the contingency vectors of the children and combine them
        into a single vector.

        Returns:
            np.ndarray: A 1D NumPy array containing the concatenated values.
        """
        if self.is_leaf():
            return np.array([])
        
        childs_contingency_vectors = [child.contingency_vector for child in self.children]
        joint_contingency_vector = np.concatenate(childs_contingency_vectors)
        return joint_contingency_vector
    
    def combine_child_constraints(self, joint_contingency_vector: np.ndarray[int]) -> List[Callable]:
        """
        Retrieve the constraints of the children and store them in a list, 
        adjusting the indices to match the new joint contingency vector that will be applied.

        Returns:
            List[Callable]: A list of callable objects with fixed parameters.
        """
        joint_constraints = []

        if not self.is_leaf():
            # All vectors have the same length
            vectors_length = len(self.contingency_vector)

            # Publication constraints defined by the user
            start = 0
            for child in self.children:
                end = start + vectors_length
                for constraint in child.constraints:
                    # NOTE: We use an object with a __call__ method, which acts like a function, replacing a lambda function.
                    # Build a sub-dict with keys 0..(e-s-1) so the constraint's indices still match
                    joint_constraints.append(SubarrayConstraintFunction(start=start, end=end, constraint=constraint))
                start = end

            # Consistency constraint: sum of children = parent
            for index in range(vectors_length):
                # Parent's contingency vector value at 'index' must equal sum of children's values at 'index'
                # Precompute the indices to sum to avoid slice notation incompatible with Pyomo vars
                indices_to_sum = list(range(index, len(joint_contingency_vector), vectors_length))
                joint_constraints.append(AggregateConstraintFunction(indices=indices_to_sum, value=self.contingency_vector[index]))
        
        return joint_constraints
    
    def update_child_vectors(self, joint_solution: np.ndarray[int]) -> None:
        """
        Update the child vectors with the solution from the estimation phase. 
        The provided list will have sufficient size for all children of the node and will respect the order of the children.
        """
        if not self.is_leaf():
            vectors_length = len(self.contingency_vector)
            start = 0
            for child in self.children:
                end = start + vectors_length
                child.contingency_vector = joint_solution[start:end]
                start = end
        return None
    
    def __repr__(self) -> str:
        '''String representation of the node.'''
        return f"HierarchicalNode(id={self.node_id}, level={self.get_level()}, children={len(self.children)})"