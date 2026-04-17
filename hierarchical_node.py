import numpy as np
from typing import Callable, List, Optional, Tuple, Iterator

class HierarchicalNode:
    ''' Represents a node in a hierarchical tree class, containing information about its ID, parent ID, level,
    DataFrame range, children range, contingency vector path, and constraints. 
    A node can be either a leaf node (if it has no children) or an internal node (if it has children).

    Example:
        Node 0 (Parent: None) in level 0 [Internal]
          DataFrame range: (0, 1000)
          Children range: (1, 10) - Total: 9 children
          Contingency vector: Setted
          Constraints: 1 constraints
        
        Node 1 (Parent: 0) in level 1 [Leaf]
          DataFrame range: (0, 100)
          Children range: Empty - Total: 0 children
          Contingency vector: Not set
          Constraints: 0 constraints

    When the tree is first built or loaded, the node contains only the ID, parent ID, level, DataFrame range,
    and children range. The contingency vector and constraints are set later depending on the specific
    queries and constraints defined in the TopDown class.

    '''

    def __init__(self, node_id: int, parent_id: Optional[int], geo_value: int, level: int,
                 df_range: Tuple[int, int], children_range: Optional[Tuple[int, int]]) -> None:
        '''Initializes a hierarchical node with its ID, parent ID, level, dataframe range, and children range.

        Args:
            node_id (int): An integer representing the unique identifier of the node.
            parent_id (Optional[int]): An optional integer representing the identifier of the parent node. None if the node is the root.
            geo_value (int): An integer that represents the geographic value of the node (i.e., the value at which it is divided within its level)
            level (int): An integer representing the level of the node in the hierarchical tree considering hierarchical columns.
            df_range (Tuple[int, int]): A tuple of two integers representing the start and end indices of the sorted dataframe rows corresponding to this node.
            children_range (Optional[Tuple[int, int]]): A tuple, or None if the node is a leaf, representing the start and end indices (corresponding with their IDs) of the child nodes.

        Attributes:
            id (int): An integer representing the unique identifier of the node.
            parent (Optional[int]): An optional integer representing the identifier of the parent node. None if the node is the root.
            geo (int): An integer that represents the geographic value of the node (i.e., the value at which it is divided within its level)
            level (int): An integer representing the level of the node in the hierarchical tree.
            df_range (Tuple[int, int]): A tuple of two integers representing the start and end indices of the sorted dataframe rows corresponding to this node.
            children_range (Optional[Tuple[int, int]]): A tuple, or None if the node is a leaf, representing the start and end indices (corresponding with their IDs) of the child nodes.
            contingency_vector (np.array[int]): A numpy array containing the contingency frequencies for the node.
            constraints (List[Callable]): A list of callables representing the constraints associated with this node associated with its contingency vector. Initially empty.
        '''
        
        self.id: int = node_id
        self.parent: Optional[int] = parent_id
        self.geo: int = geo_value

        self.level: int = level

        self.df_range: Tuple[int, int] = df_range
        self.children_range: Optional[Tuple[int, int]] = children_range

        self.contingency_vector: np.array[int] = None
        self.constraints: List[Callable] = []

    def is_root(self) -> bool:
        '''Determine if the node is the root node.
        
        Returns:
            bool: True if the node is root (i.e., has no parent node), False otherwise.
        '''
        return self.parent is None

    def is_leaf(self) -> bool:
        '''Determine if the node is a leaf node (i.e., has no children) based on the presence of the children_range attribute.

        Returns:
            bool: True if the node is a leaf node, False otherwise.

        '''
        return self.children_range is None
    
    def get_children_iter(self) -> Iterator[int]:
        '''Return an iterator over the range of children of the node.

        Returns:
            Iterator[int]: An iterator yielding child indices from start to end (inclusive).
        '''
        start, end = self.children_range
        return iter(range(start, end + 1))
    
    def __repr__(self) -> str:
        '''Create a string representation of the hierarchical node,
        showing its ID, parent ID, level, dataframe range, children range, contingency vector path, and constraints.

        Returns:
            str: A string representation of the hierarchical node.

        '''
        node = f"\nNode {self.id} (Parent: {self.parent}) in level {self.level} {'[Leaf]' if self.is_leaf() else '[Internal]'}\n"
        node += f"  DataFrame range: {self.df_range} \n"
        node += f"  Children range: {self.children_range if self.children_range else 'Empty'}"

        if self.children_range is not None:
            node += f" - Total: {1 + self.children_range[1] - self.children_range[0]} children"

        node += f"\n  Contingency vector: {'Setted' if not self.contingency_vector is None else 'Not set'} \n"
        node += f"  Constraints: {len(self.constraints)} constraints \n"
        
        return node