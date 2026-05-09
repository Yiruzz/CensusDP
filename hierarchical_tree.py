import numpy as np

from hierarchical_node import HierarchicalNode

from collections import deque
from typing import List, Callable, Tuple, Generator

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
            _node_count (int): Internal counter to keep track of the number of nodes in the tree.
            _contingency_vectors_shm (Optional[str]): Name of the shared memory buffer where contingency vectors are stored.
        """
        self.nodes = [HierarchicalNode(geo_id=root_id, level=level, constraints=constraints)]
        self._node_count = 1
        self._levels = [level]
        self._contingency_vectors = None
        self._contingency_vectors_shm = None
    
    def combine_vectors(self, node: HierarchicalNode) -> np.ndarray:
        '''Retrieve the contingency vectors of the children of node and combine them into a single vector.

        Returns:
            np.ndarray: A 1D NumPy array containing the concatenated values.
        '''

        if node.is_leaf():
            return np.array([])
        
        childs_contingency_vectors = [self._contingency_vectors[child.id] for child in node.children]
        joint_contingency_vector = np.concatenate(childs_contingency_vectors, dtype='int64')
        return joint_contingency_vector

    def update_vectors(self, node: HierarchicalNode, joint_solution: np.ndarray) -> None:
        """
        Update the child vectors with the solution from the estimation phase. 
        The provided list will have sufficient size for all children of the node and will respect the order of the children.
        """
        if not node.is_leaf():
            vectors_length = self._contingency_vectors.shape[1]
            start = 0
            for child in node.children:
                end = start + vectors_length
                self._contingency_vectors[child.id, :] = joint_solution[start:end]
                start = end
        
    def combine_child_constraints(self, node: HierarchicalNode, joint_contingency_vector: np.ndarray) -> list[Callable]:
        """
        Retrieve the constraints of the children and store them in a list, 
        adjusting the indices to match the new joint contingency vector that will be applied.

        Returns:
            node: 
            list[Callable]: A list of callable objects with fixed parameters.
        """
        joint_constraints = []

        if not node.is_leaf():
            # All vectors have the same length
            vectors_length = self._contingency_vectors.shape[1]

            # Publication constraints defined by the user
            start = 0
            for child in node.children:
                end = start + vectors_length
                for constraint in child.constraints:
                    # NOTE: We use an object with a __call__ method, which acts like a function, replacing a lambda function.
                    # Build a sub-dict with keys 0..(e-s-1) so the constraint's indices still match
                    joint_constraints.append(lambda joint_array, s=start, e=end, c=constraint: c({i - s: joint_array[i] for i in range(s, e)}))
                start = end

            # Consistency constraint: sum of children = parent
            for index in range(vectors_length):
                # Parent's contingency vector value at 'index' must equal sum of children's values at 'index'
                # Precompute the indices to sum to avoid slice notation incompatible with Pyomo vars
                indices_to_sum = list(range(index, len(joint_contingency_vector), vectors_length))
                joint_constraints.append(lambda joint_array, idxs=indices_to_sum, value=self._contingency_vectors[node.id, index]:
                                             sum(joint_array[j] for j in idxs) == value)
        
        return joint_constraints
    
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
    
