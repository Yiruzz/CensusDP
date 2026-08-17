"""The single place that builds a CostModel for marginal selection.

A cost-aware heuristic asks three questions the association matrix cannot answer: how many
cells a bag really has once the declared edit rules remove the impossible ones, how much
noise the tree puts on a cell at a given number of bags, and how much noise the pairwise
selection measurement itself carried. The answers need the domain, the constraints and the
privacy mechanism - none of which ``graph`` is allowed to import. This module is the join,
and it lives at the project root next to the other pipeline modules for that reason.

It has to stay the ONLY definition, for the same reason ``tree_wide`` does: the experiment
harness derives the cost independently of TopDown (the oracle arms select before any
TopDown exists), and two derivations of a cell space do not raise when they disagree - they
mislay indices.

Nothing here reads the data. Column cardinalities, constraint scopes and the privacy
parameters are all declarations, so consulting the cost is free and the released width does
not become data-dependent.
"""

from typing import Iterable, Optional, Sequence, Tuple

from math import prod

from constraints.constraint import Constraint
from constraints.domain_restriction import build_restriction
from domain import ContingencyDomain
from graph import CostModel
from privacy.variants import PrivacyMechanism

# Above this many cells a bag's restriction is not materialised. build_restriction allocates
# one bool per cell of the UNRESTRICTED space, so a candidate bag of a few hundred million
# cells would cost hundreds of megabytes to reject. The product is returned instead, which
# is a safe upper bound - restricting only ever shrinks - and such a bag is over any sane
# width cap anyway.
DEFAULT_MATERIALIZE_CAP = 20_000_000


def build_cost_model(contingency_domain: ContingencyDomain,
                     structural: Optional[Iterable[Constraint]] = None,
                     mechanism: Optional[PrivacyMechanism] = None,
                     tree_level: int = 0,
                     selection_level: Optional[int] = None,
                     selection_sensitivity: Optional[int] = None,
                     max_width: Optional[int] = None,
                     materialize_cap: int = DEFAULT_MATERIALIZE_CAP) -> CostModel:
    '''Assemble the cost model a cost-aware selection strategy needs.

    Args:
        contingency_domain (ContingencyDomain): The global domain, as
            DataHandler.build_contingency_domain leaves it. Only its per-column value sets
            are used, through subdomain(); the joint size is never forced.
        structural (Optional[Iterable[Constraint]]): The tree-wide constraints, as
            tree_wide() selects them. Those declaring structural zeros shrink each candidate
            bag, which is exactly why the cost cannot be estimated as a product of
            cardinalities.
        mechanism (Optional[PrivacyMechanism]): The mechanism whose budget the run will
            spend. None means no noise is modelled at all: tree_noise returns 1.0, so the
            cost term degenerates to counting cells, and selection_noise is 0.
        tree_level (int): Which level's noise scale represents the tree. Under zCDP the
            scale is proportional to sqrt(number of bags) at EVERY level, so the ranking
            between candidates does not depend on this choice - only the scale of the
            heuristic's lam does. The root is the conservative reading.
        selection_level (Optional[int]): Level index of the pairwise selection measurement
            (_reserve_selection_budget appends it last). None means the association came
            from raw data (a non-DP oracle arm) and carries no noise to correct for.
        selection_sensitivity (Optional[int]): Sensitivity of that measurement, i.e. the
            number of pairs measured. Required together with selection_level.
        max_width (Optional[int]): Hard cap on the marginal width W, or None for no cap.
        materialize_cap (int): See DEFAULT_MATERIALIZE_CAP.

    Returns:
        CostModel: With bag_cells already memoised - a greedy search re-evaluates the same
            bags thousands of times, and the restriction is the expensive step.

    Raises:
        ValueError: If exactly one of selection_level / selection_sensitivity is given.
    '''
    rules = list(structural or [])
    cardinalities = {column: int(size) for column, size
                     in zip(contingency_domain.columns, contingency_domain.sizes)}

    cache = {}

    def bag_cells(bag: Tuple[str, ...]) -> int:
        key = tuple(bag)
        cached = cache.get(key)
        if cached is not None:
            return cached

        product = prod(cardinalities[column] for column in key)
        if product > materialize_cap:
            # Upper bound rather than an exception: the caller is scoring a candidate, and
            # a bag this size loses on cells alone.
            cache[key] = product
            return product

        cells = build_restriction(contingency_domain.subdomain(key), rules).n_cells
        cache[key] = cells
        return cells

    if (selection_level is None) != (selection_sensitivity is None):
        raise ValueError(
            "selection_level and selection_sensitivity go together: the noise of the "
            "pairwise measurement is not determined by one of them alone."
        )

    if mechanism is None:
        # No mechanism: the cost term counts cells and nothing else, and there is no
        # measurement noise to de-bias the association with.
        tree_noise = lambda n_bags: 1.0                                   # noqa: E731
        selection_noise = 0.0
    else:
        tree_noise = lambda n_bags: mechanism.expected_abs_noise(         # noqa: E731
            tree_level, int(n_bags))
        selection_noise = (0.0 if selection_level is None else
                           mechanism.expected_abs_noise(selection_level,
                                                        int(selection_sensitivity)))

    return CostModel(cardinalities=cardinalities, bag_cells=bag_cells,
                     tree_noise=tree_noise, selection_noise=selection_noise,
                     max_width=max_width)


def domain_from_declared(columns: Sequence[str], domains) -> ContingencyDomain:
    '''A ContingencyDomain straight from declared per-column value sets.

    The pipeline builds its domain inside initialize(), after a DuckDB view exists, because
    an undeclared column has to be inferred. Anything that wants to score a structure
    WITHOUT running the pipeline (the oracle arms, the lam sweep) already has the value sets
    in hand and needs the domain earlier than that. This is that shortcut, in one place, so
    the probe and the run shape the same cell space.

    Args:
        columns (Sequence[str]): Query columns, in significance order.
        domains: Column -> sorted value array, as ContingencyDomain holds them.

    Returns:
        ContingencyDomain: Unrestricted; build_cost_model applies the rules per bag.
    '''
    columns = list(columns)
    return ContingencyDomain(columns=columns, domains={c: domains[c] for c in columns})
