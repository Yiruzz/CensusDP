"""What a clique costs, in the currency the pipeline actually pays.

A marginal-selection heuristic that trades utility against size needs three things the
association matrix does not carry: how many cells a bag really has, how much noise the
tree puts on each cell, and how much noise the selection measurement itself carried. All
three are cheap to compute, but they live outside this package - the cell count needs the
declared domain and the structural zeros, the noise scale needs the privacy mechanism.

CostModel is the seam. It holds only plain numbers and callables, so ``graph`` stays free
of any data / DuckDB / privacy / domain import, exactly as before. ``selection_cost.py``
at the project root is the single place that builds one; both TopDown and the experiment
harness go through it, for the same reason ``tree_wide`` has to be single: two derivations
of the cell space do not raise when they disagree, they mislay indices.

Everything in here is derived from DECLARATIONS - column cardinalities, constraint scopes,
the privacy parameters - never from the data. That is what lets a heuristic consult the
cost for free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class CostModel:
    """The public, data-independent cost of a candidate structure.

    Attributes:
        cardinalities: Column -> domain size. Also what JunctionTree.build wants as its
            cardinality-aware triangulation weights.
        bag_cells: Bag (tuple of columns) -> its RESTRICTED cell count, i.e. the product
            space minus the cells the declared edit constraints make impossible. Expected
            to be memoised by the builder: a greedy search re-evaluates the same bags
            thousands of times, and materialising a restriction is the expensive step.
        tree_noise: Number of bags -> expected |noise| per cell of the tree measurement,
            in records. The number of bags IS the sensitivity in the factored pipeline, so
            this is how a candidate that adds a bag pays for the extra noise it puts on
            every cell of the model, not just its own.
        selection_noise: Expected |noise| per cell of the pairwise selection measurement,
            in records. Used to de-bias the association matrix: a table of n cells carries
            about ``selection_noise * n`` of L1 that is noise rather than signal. Zero when
            the association came from raw data (the non-DP oracle arms).
        max_width: Hard cap on the marginal width W. A candidate whose junction tree
            exceeds it is dropped outright rather than penalised. No metric of problem size
            predicts the rounding MIP's runtime, so this is a coarse filter - but it is the
            only cheap one there is, and it is decided in seconds before anything is solved.
    """

    cardinalities: Mapping[str, int]
    bag_cells: Callable[[Tuple[str, ...]], int]
    tree_noise: Callable[[int], float]
    selection_noise: float = 0.0
    max_width: Optional[int] = None

    def width(self, bags) -> int:
        """Marginal width W of a set of bags: the concatenated measurement's length."""
        return sum(self.bag_cells(tuple(bag)) for bag in bags)

    def noise_burden(self, bags) -> float:
        """Expected L1 of the noise over a whole node measurement, in records.

        ``E|noise| per cell x number of cells``, with the per-cell scale read at the bag
        count those bags imply. Both factors move when a clique is added - a new bag raises
        the sensitivity, hence the noise on cells that were already there - so this is the
        quantity to difference, not W alone.
        """
        bags = list(bags)
        return self.tree_noise(len(bags)) * self.width(bags)
