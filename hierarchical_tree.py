from hierarchical_node import HierarchicalNode
from typing import List, Tuple

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
            noisy_contingency_vectors (bool): Specify if the contingency vectors of nodes have noisy or not, by default, False. Also if they don't exist, we assume it. 
        '''
        self.nodes: List["HierarchicalNode"] = nodes
        self.node_ranges_by_level: List[Tuple[int, int]] = node_ranges_by_level
        self._node_count: int = len(self.nodes)
        self._level_count: int = len(self.node_ranges_by_level)

        self.noisy_contingency_vectors: bool = False

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