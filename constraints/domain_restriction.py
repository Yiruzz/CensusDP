"""Turn declared skip-logic rules into a smaller cell space instead of optimizer rows.

A rule like `Implies(Equal('P08', 1), Equal('P19', 98))` ("men are not asked how many children
they had") is a structural zero, it is a definition based on the domain, not on the data or on the
geographic node. Emitting it as `sum(x[forbidden cells]) == 0` keeps those cells alive through
measurement, noise, spill and the root's variables just to pin them to zero. This module removes
them from the cell space instead, so the rule disappears rather than being enforced.

Everything here is derived from declarations, never from the data, which is what makes it
DP-safe.
"""

import numpy as np

from typing import Iterable, List, Mapping, Sequence

from constraints.constraint import Constraint
from constraints.logical_expressions.base import LogicalExpression
from domain import ContingencyDomain, RestrictedDomain


def tree_wide(constraints_by_level: "Mapping[int, Sequence[Constraint]]") -> List[Constraint]:
    """The constraints registered at every level of the hierarchy.

    Only these may shrink the cell space, because it is shared by every node of the tree.

    Args:
        constraints_by_level: Level -> constraints, as TopDown.constraints holds it.

    Returns:
        List[Constraint]: In registration order.
    """
    levels = list(constraints_by_level.values())
    if not levels:
        return []
    # Identity, not equality, two constraints can compare equal and still be distinct
    # registrations, and Constraint does not define __eq__ anyway.
    common = set.intersection(*({id(c) for c in level} for level in levels))
    return [c for c in levels[0] if id(c) in common]


def build_restriction(domain: ContingencyDomain, constraints: Iterable[Constraint]) -> ContingencyDomain:
    """Restrict ``domain`` to the cells its declared logical constraints allow.

    Every logical constraint restricts - declaring one is declaring that the combination is
    impossible, so there is no reason to keep paying for cells it forbids. The consequence is
    that a record contradicting one stops the run (RestrictedDomain.encode raises). Cleaning
    the data, and declaring only the constraints that must actually hold, is the caller's
    responsibility.

    Aggregate constraints (SumEqual, SumEqualRealTotal) are not logical and never restrict,
    "these cells sum to N" says nothing about which cells are possible.

    A constraint only applies when the domain holds every column it mentions. That is what
    lets each bag derive its restriction independently. A bag that cannot evaluate a
    constraint simply does not apply it, and the cells it would have removed are driven to
    zero through the separator rows instead.

    Args:
        domain: The unrestricted product space (a bag sub-domain, or the joint domain of the
            full-joint pipeline).
        constraints: Candidate constraints. Aggregates, and those whose scope is not inside
            this domain, are ignored.

    Returns:
        A RestrictedDomain, or ``domain`` unchanged when there is nothing to restrict.

    Raises:
        ValueError: If the constraints leave no cell at all, which means they contradict
            each other.
    """
    columns = set(domain.columns)
    rules = [c for c in constraints
             if isinstance(c, LogicalExpression) and c.scope() <= columns]

    if not rules:
        return domain

    # reduce() is already "the cells where the constraint holds", i.e. the ones that survive
    # it. A cell must satisfy every constraint, so the masks are ANDed.
    mask = np.ones(domain.n_cells, dtype=bool)
    for rule in rules:
        mask &= rule.reduce(domain)

    valid_cells = np.flatnonzero(mask)
    if len(valid_cells) == len(mask):
        return domain
    if len(valid_cells) == 0:
        raise ValueError(
            f"The declared rules over {sorted(domain.columns)} forbid every cell of the "
            f"domain. They contradict each other."
        )
    return RestrictedDomain(domain, valid_cells)
