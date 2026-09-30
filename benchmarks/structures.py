"""Families of marginal structures derived mechanically from a declared one, for experiment 6.

Experiment 6 needs several structures per dataset, and hand-declaring them is what blocked it:
three of the five datasets declare only one, and the alternatives that once existed for SINASC and
Spain were the only copy and were deleted. These two rules generate a spread around the
declaration instead, in both directions:

    coarsen  merge the two bags sharing an attribute whose union has the fewest cells
    refine   from a bag of three or more attributes, drop the pair whose cardinality product is
             largest and keep the two subsets that omit one of them each

Both are width-extremal, which is what makes them deterministic with no tuning constant, and
neither reads the data, so no budget is spent on selection and the released width stays
independent of the microdata. `WIDTH_CAP` only stops a merge from producing a bag nothing can
solve; it does not choose between merges.

Three cells per bag move at once, in two directions, which is the point of the experiment rather
than a flaw in it: coarsening raises the coverage (the fraction of attribute pairs inside some bag)
and lowers Delta = 2 x bags, but multiplies the width; refining does the reverse.

**What a structure requests is not what it gets.** The junction tree embeds every declared bag and
every constraint scope as a clique, triangulates, and takes the maximal cliques, so a refinement
can be re-fused and a dataset whose scopes already dominate barely moves. Nothing here asserts
otherwise: `runner` records the bags the tree actually returned, and the analysis plots those. See
`exp6` in `experiments.py` for the measured per-dataset outcome.
"""

import itertools

# A merge past this many cells is refused. Not a choice between merges, only a floor on solvability.
WIDTH_CAP = 1_000_000


def cells(bag, sizes):
    total = 1
    for column in bag:
        total *= sizes[column]
    return total


def coarsen(bags, sizes):
    """Merge the cheapest pair of bags sharing an attribute. None when no merge fits the cap."""
    best = None
    for left, right in itertools.combinations(bags, 2):
        if not set(left) & set(right):
            continue
        union = tuple(sorted(set(left) | set(right)))
        if cells(union, sizes) > WIDTH_CAP:
            continue
        if best is None or cells(union, sizes) < cells(best[0], sizes):
            best = (union, left, right)
    if best is None:
        return None
    union, left, right = best
    return [union] + [bag for bag in bags if bag not in (left, right)]


def refine(bags, sizes):
    """Split every bag of three or more attributes in two. None when every bag is already a pair."""
    if all(len(bag) < 3 for bag in bags):
        return None
    out = []
    for bag in bags:
        if len(bag) < 3:
            out.append(bag)
            continue
        first, second = max(itertools.combinations(bag, 2),
                            key=lambda pair: sizes[pair[0]] * sizes[pair[1]])
        out += [tuple(a for a in bag if a != first), tuple(a for a in bag if a != second)]
    return out


def family(bags, sizes, steps=4):
    """{'m1': bags, ..., 's1': bags, ...} around `bags`: `steps` rungs of each rule at most.

    'm' walks toward the joint distribution and 's' away from it. A rung that changes nothing, or
    that no longer fits the cap, ends that direction.
    """
    start = sorted({tuple(sorted(set(bag))) for bag in bags})
    derived = {}
    for prefix, rule in (('m', coarsen), ('s', refine)):
        current = start
        for step in range(1, steps + 1):
            following = rule(current, sizes)
            if following is None:
                break
            following = sorted({tuple(sorted(set(bag))) for bag in following})
            if following == current:
                break
            current = following
            derived[f'{prefix}{step}'] = current
    return derived


def register_family(structures, default, domains, steps=4):
    """Add the derived family of `structures[default]` to `structures`, and return its names.

    Called from each dataset's `marginals.py` so that the names exist in the module itself: the
    orchestrator needs them to build the queue and the driver, which is a separate process, needs
    them to resolve `--structure`. Registering from `experiments.py` would only reach the first.
    """
    sizes = {column: len(values) for column, values in domains.items()}
    declared = [[column for column in bag if column in sizes] for bag in structures[default]]
    names = []
    for suffix, bags in family([bag for bag in declared if bag], sizes, steps).items():
        name = f'{default}_{suffix}'
        structures[name] = [list(bag) for bag in bags]
        names.append(name)
    return sorted(names)
