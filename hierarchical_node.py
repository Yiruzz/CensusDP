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
        '''
        Initialize a hierarchical node.

        Args:
            geo_id (int): Identifier related to geography.
            level (int): Level where the node is located.

        Attributes:
            id (Optional[int]): Unique incremental identifier assigned during tree construction via BFS.
            geo_id (int): Identifier related to geography.

            children (List[HierarchicalNode]): List of child nodes.
            parent (HierarchicalNode): Reference to the parent node.

            hierarchical_path (List[Any]): List of hierarchical geo_ids visited to reach this node from the root.
                                           Example: [0] for root, [0, 'A'] for region A, [0, 'A', 'A1'] for region A + comuna A1.
                                           Used to filter the dataframe to get this node's data subset.
            level (int): Level where the node is located.

            contingency_vector (Optional[np.ndarray]): Node's contingency vector, None when not materialized or freed.
            constraints (Optional[List[Callable]]): List of constraints for this node.
        '''
        self.id: Optional[int] = None
        self.geo_id: int = geo_id

        self.children: List[HierarchicalNode] = []
        self.parent: Optional[HierarchicalNode] = None

        self.hierarchical_path: List[int] = [0]
        self.level: int = level

        self.contingency_vector: Optional[np.ndarray] = None
        self.constraints: Optional[List[Callable]] = None

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
    
    def combine_child_vectors(self) -> np.ndarray:
        '''Concatenate the contingency vectors of all children into a single vector.

        Returns:
            np.ndarray: A 1D array with the concatenated child vectors.
                       Empty array if this node is a leaf.
        '''
        if self.is_leaf():
            return np.array([], dtype=int)

        return np.concatenate([child.contingency_vector for child in self.children])
    
    def combine_child_constraints(self) -> List[Callable]:
        '''Combine all child constraints into a single list with adjusted indices.

        Each child constraint is adapted to work with the flattened joint vector.
        Consistency constraints ensure parent value = sum of child values at each index.

        Returns:
            List[Callable]: Constraints callable with all indices adjusted to joint vector.
                           Empty list if this node is a leaf.
        '''
        joint_constraints = []

        if not self.is_leaf():
            vectors_length = len(self.contingency_vector)
            num_children = len(self.children)

            # Wrap child publication constraints with adjusted indices
            start = 0
            for child in self.children:
                end = start + vectors_length
                for constraint in child.constraints:
                    joint_constraints.append(
                        lambda joint_array, s=start, e=end, c=constraint:
                            c({i - s: joint_array[i] for i in range(s, e)})
                    )
                start = end

            # Add consistency constraints: parent value at each index = sum of child values at that index
            for index in range(vectors_length):
                indices_to_sum = [index + i * vectors_length for i in range(num_children)]
                joint_constraints.append(
                    lambda joint_array, idxs=indices_to_sum, value=self.contingency_vector[index]:
                        sum(joint_array[j] for j in idxs) == value
                )

        return joint_constraints
    
    def update_child_vectors(self, joint_solution: np.ndarray) -> None:
        '''Distribute the joint solution back to individual child contingency vectors.

        Args:
            joint_solution (np.ndarray): Concatenated solution from optimization,
                                        with one child's vector after another.
        '''
        if not self.is_leaf():
            vectors_length = len(self.contingency_vector)
            start = 0
            for child in self.children:
                end = start + vectors_length
                child.contingency_vector = joint_solution[start:end]
                start = end
    
    def __str__(self) -> str:
        '''Return a detailed string representation of the node with key attributes.'''
        has_contingency = self.contingency_vector is not None and len(self.contingency_vector) > 0
        has_constraints = self.constraints is not None and len(self.constraints) > 0
        path = " -> ".join(str(x) for x in self.hierarchical_path)

        result = "--- HierarchicalNode ---\n"
        result += f"ID: {self.id}\n"
        result += f"Geo ID: {self.geo_id}\n"
        result += f"Nivel: {self.level}\n"
        result += f"Ruta jerárquica: {path}\n"
        result += f"Cantidad de hijos: {len(self.children)}\n"
        result += f"Vector de contingencia: {has_contingency}\n"
        result += f"Constraints: {has_constraints}\n"

        return result.strip()