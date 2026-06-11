import numpy as np

from typing import Any, Callable, Dict, List, Optional

class HierarchicalNode:
    '''Represents a node in a hierarchical tree structure.

    Each node contains filter information for accessing the subset of data it represents,
    and a list of its children. The actual contingency vectors and constraints are computed
    lazily when needed, not at tree construction time.

    This class focuses solely on node specific data and operations,
    without any tree traversal or tree-wide operation logic.
    '''
    def __init__(self, level: int, filter_dict: Dict[str, Any]) -> None:
        '''
        Initialize a hierarchical node.

        Args:
            level (int): Level where the node is located.
            filter_dict (Dict[str, Any]): Dictionary mapping column names to their filter values.
                Default is an empty dictionary. Example: {} for root, {'Region': 'A'} for region A,
                {'Region': 'A', 'Comuna': 'A1'} for region A + comuna A1.

        Attributes:
            id (Optional[int]): Unique incremental ID assigned via BFS traversal after tree construction.

            children (List[HierarchicalNode]): List of child nodes.
            parent (HierarchicalNode): Reference to the parent node.

            filter_dict (Dict[str, Any]): Dictionary mapping column names to their filter values. Used to filter the data to get this node's data subset.
            level (int): Level where the node is located.

            contingency_vector: Node's contingency vector, None when not materialized or freed.
                Has two lifecycle states: a dense np.ndarray noisy measurement (query space)
                right after materialization, then a sparse scipy CSC column of estimated cell
                counts (cell space) after the node is solved. 
            constraints (Optional[List[Callable]]): List of constraints for this node.
        '''
        self.id: Optional[int] = None

        self.children: List[HierarchicalNode] = []
        self.parent: Optional[HierarchicalNode] = None

        self.filter_dict: Dict[str, Any] = filter_dict 
        self.level: int = level

        self.contingency_vector: Optional[np.ndarray] = None
        self.constraints: Optional[List[Callable]] = None

    def add_child(self, child_node: 'HierarchicalNode') -> None:
        '''Add a child node to this node.

        Args:
            child_node (HierarchicalNode): The child node to add.
        '''
        child_node.parent = self
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
        '''Return a detailed string representation of the node with key attributes.'''
        cv = self.contingency_vector
        if cv is None:
            has_contingency = False
        elif hasattr(cv, 'nnz'):  # sparse cell-count vector (estimated state)
            has_contingency = cv.nnz > 0
        else:  # dense noisy measurement vector (pre-estimation state)
            has_contingency = cv.size > 0
        has_constraints = self.constraints is not None and len(self.constraints) > 0
        filter_str = ", ".join(f"{k}={v}" for k, v in self.filter_dict.items()) if self.filter_dict else "root"

        result = "--- HierarchicalNode ---\n"
        result += f"ID: {self.id}\n"
        result += f"Nivel: {self.level}\n"
        result += f"Filtros: {filter_str}\n"
        result += f"Cantidad de hijos: {len(self.children)}\n"
        result += f"Vector de contingencia: {has_contingency}\n"
        result += f"Constraints: {has_constraints}\n"

        return result.strip()