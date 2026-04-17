import numpy as np

from callable_wrappers import (SerializableAggregateFunction,
                               SerializableSubarrayFunction)

from hierarchical_node import HierarchicalNode
from typing import List, Tuple, Callable

class HierarchicalTree:
    '''Represents a hierarchical tree structure composed of multiple HierarchicalNode objects.
    The tree is represented as a list of nodes.

    Example:
        [<HierarchicalNode>, <HierarchicalNode>, ...]

    Where each node id corresponds to its index in the list
    and the father_id attribute of each node indicates the index of its parent node in the list.
    Also is possible indentify the levels of the tree by after get the nodes of each level using ranges of indices.
    '''

    def __init__(self, nodes: List["HierarchicalNode"], node_ranges_by_level: List[Tuple[int, int]]) -> None:
        '''Initializes the hierarchical tree structure with a list of nodes and their corresponding index ranges for each level.
        
        Args:
            nodes (List["HierarchicalNode"]): A list of HierarchicalNode objects representing the nodes in the tree.
            node_ranges_by_level (List[Tuple[int, int]]): A list of tuples, where each tuple contains the start and end index of nodes for a specific level in the tree.
        
        Atributes:
            nodes (List["HierarchicalNode"]): A list of HierarchicalNode objects representing the nodes in the tree.
            node_ranges_by_level (List[Tuple[int, int]]): A list of tuples, where each tuple contains the start and end index of nodes for a specific level in the tree.
            _node_count (int): The total number of nodes in the tree.
            _level_count (int): The total number of levels in the tree.
        '''
        self.nodes: List["HierarchicalNode"] = nodes
        self.node_ranges_by_level: List[Tuple[int, int]] = node_ranges_by_level
        self._node_count: int = len(self.nodes)
        self._level_count: int = len(self.node_ranges_by_level)

    def get_combined_contingency_vector(self, idx_node: int) -> np.ndarray:
        '''Retrieve the contingency vectors from the child nodes within the
        specified range and combine them into a single vector.

        Args:
            idx_node (int): Index of the node whose children vectors will be combined.

        Returns:
            np.ndarray: Concatenated array of all contingency vectors in the specified children range.
        '''
        start_idx, end_idx = self.nodes[idx_node].children_range
        contingency_vectors = [node.contingency_vector for node in self.nodes[start_idx:end_idx+1]]
        joint_contingency_vector = np.concatenate(contingency_vectors)
        return joint_contingency_vector
    
    def build_joint_constraints(self, idx_node: int, joint_contingency_vector: np.ndarray[int]) -> List[Callable]:
        '''Build the set of constraints associated with a node over a joint contingency vector.

        This function combines the constraints of multiple nodes into a single representation
        defined over a joint contingency vector (previously constructed by combining individual
        node vectors).

        The resulting constraints include:
        - Publication constraints: ensure that the original constraints of each node are preserved
        when expressed in the joint vector space.
        - Consistency constraints: ensure that the values corresponding to a specific node match
        the aggregation (sum) of the relevant entries in the joint vector.

        Args:
            idx_node (int): Index of the node whose constraints are being enforced within the joint structure.
            joint_contingency_vector (np.ndarray): Combined contingency vector representing multiple nodes.

        Returns:
            List[Callable]: A list of constraint functions defined over the joint contingency vector.
        '''
        joint_constraints = []

        # All vectors have the same length
        vectors_length = len(self.nodes[idx_node].contingency_vector)
        joint_contingency_vector_length = len(joint_contingency_vector)

        # Publication constraints defined by the user
        i = 0
        start_idx, end_idx = self.nodes[idx_node].children_range
        for node in self.nodes[start_idx:end_idx+1]:
            j = i + vectors_length
            for constraint in node.constraints:
                # NOTE: We use an object with a __call__ method, which acts like a function, replacing a lambda function.
                # Build a sub-dict with keys 0..(e-s-1) so the constraint's indices still match
                joint_constraints.append(SerializableSubarrayFunction(start=i, end=j, func=constraint))
            i = j

        # Consistency constraint: sum of children = parent
        for idx in range(vectors_length):
            # Parent's contingency vector value at 'index' must equal sum of children's values at 'index'
            # Precompute the indices to sum to avoid slice notation incompatible with Pyomo vars
            indices_to_sum = list(range(idx, joint_contingency_vector_length, vectors_length))
            joint_constraints.append(SerializableAggregateFunction(indices=indices_to_sum, value=self.nodes[idx_node].contingency_vector[idx]))
    
        return joint_constraints
    
    def set_contingency_vectors(self, idx_node: int, joint_solution: np.ndarray) -> None:
        ''' Update the contingency vectors of the child nodes within the range defined
        by the given node, slicing the provided joint vector according to each
        child vector length.

        Args:
            idx_node (int): Index of the node whose children will be updated.
            joint_solution (np.ndarray): A concatenated array containing all
                contingency vectors of the child nodes.
        '''
        start_idx, end_idx = self.nodes[idx_node].children_range
        vectors_length = len(self.nodes[start_idx].contingency_vector)
        
        i = 0
        for node in self.nodes[start_idx:end_idx+1]:
            j = i + vectors_length
            node.contingency_vector = joint_solution[i:j]
            i = j
        return None

    def __repr__(self) -> str:
        '''Create a string representation of the hierarchical tree structure.
        Show the number of nodes and levels, and the range of node indices for each level.

        Returns:
            str: A string representation of the hierarchical tree structure.
        '''
        tree = f"HierarchicalTree with {self._node_count} nodes and {self._level_count} levels. \n"

        for level, (start_idx, end_idx) in enumerate(self.node_ranges_by_level):
            tree += f"  Level {level}: Nodes {start_idx} to {end_idx-1} - {end_idx - start_idx} nodes \n"
        return tree