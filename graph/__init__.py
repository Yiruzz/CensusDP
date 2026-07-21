"""Graph & junction-tree construction for the factored-marginals pipeline.

Pure structural logic, independent of the DP pipeline, the solver, and the data.
"""
from .junction_tree import JunctionTree, Bag
from .association import PairwiseAssociation, mutual_information, PairCounts
from .heuristics import (
    MarginalSelectionStrategy,
    MaxSpanningTreeMI,
    UnconstrainedMaxSpanningTreeMI,
)

__all__ = [
    "JunctionTree",
    "Bag",
    "PairwiseAssociation",
    "mutual_information",
    "PairCounts",
    "MarginalSelectionStrategy",
    "MaxSpanningTreeMI",
    "UnconstrainedMaxSpanningTreeMI",
]
