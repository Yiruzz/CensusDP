"""Graph & junction-tree construction for the factored-marginals pipeline.

Pure structural logic, independent of the DP pipeline, the solver, and the data.
A cost-aware heuristic needs cardinalities and noise scales, which live outside; they
arrive as plain numbers and callables inside a CostModel, built by selection_cost.py.
"""
from .junction_tree import JunctionTree, Bag
from .association import (
    PairwiseAssociation,
    mutual_information,
    independence_residual_l1,
    PairCounts,
    PairStatistic,
)
from .cost_model import CostModel
from .heuristics import (
    MarginalSelectionStrategy,
    MaxSpanningTreeMI,
    UnconstrainedMaxSpanningTreeMI,
    CostAwareGreedySelection,
)

__all__ = [
    "JunctionTree",
    "Bag",
    "PairwiseAssociation",
    "mutual_information",
    "independence_residual_l1",
    "PairCounts",
    "PairStatistic",
    "CostModel",
    "MarginalSelectionStrategy",
    "MaxSpanningTreeMI",
    "UnconstrainedMaxSpanningTreeMI",
    "CostAwareGreedySelection",
]
