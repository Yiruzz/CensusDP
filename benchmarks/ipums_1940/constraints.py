"""Constraints for IPUMS 1940: no edit rules, and the DAS's unit bounds on request.

The DAS 1940 configuration (configs/Census1940/DDP2010_Update/ipums_1940.ini) declares no edit rule.
It declares invariants: the total per state, which the runner already holds (set_constraint_to_level
covers the nation and the states), and `gqhh_vect`, the occupied households and group-quarters
facilities of each type per enumeration district, read from the H records and published exactly.
From those it bounds the persons of each hhgq type (`hhgq_total_lb/ub`, querybase_constraints.py):

    persons of type g >= units of type g      for every group-quarters type (households: >= 0,
                                              since a housing unit may be vacant)
    persons of type g <= units of type g * 99999, i.e. none at all where there is no such unit

The DAS computes them at the enumeration districts and aggregates them upwards, so they hold at every
level. At the levels where the total is also invariant its formula adds `<= total - the minimum of
the other types`, which the exact total and the lower bounds of the other types already imply, so it
is left out. `unit_bounds` returns them; benchmarks/runner.py applies them with --unit-bounds, which
only the comparison with the DAS uses.
"""

import duckdb
import numpy as np

from constraints.contextual_constraints import SumBoundOfNode
from constraints.logical_expressions.atomic import Equal

from .domains import DOMAINS
from .prepare import units_path

# hhgq_cap in Constraints_DHCP.py: it only ever binds where there are no units.
CAP = 99999


def constraints(columns):
    return []


def unit_bounds(hierarchy, sample):
    """hhgq_total_lb and hhgq_total_ub of the DAS, one SumBoundOfNode per hhgq type and sense."""
    path = units_path(sample)
    con = duckdb.connect()
    units = {}
    for depth in range(len(hierarchy) + 1):
        keys = list(hierarchy[:depth])
        rows = con.execute(f"SELECT {', '.join(keys + ['hhgq', 'SUM(units)'])} "
                           f"FROM read_parquet('{path}') GROUP BY {', '.join(keys + ['hhgq'])}").fetchall()
        for row in rows:
            units.setdefault(tuple(row[:depth]), np.zeros(len(DOMAINS['hhgq']), dtype=np.int64))[
                row[depth]] += row[depth + 1]
    rules = []
    for g in DOMAINS['hhgq']:
        of_type = {node: int(counts[g]) for node, counts in units.items()}
        if g > 0:
            rules.append(SumBoundOfNode(Equal('hhgq', g), '>=', of_type))
        rules.append(SumBoundOfNode(Equal('hhgq', g), '<=', {node: n * CAP for node, n in of_type.items()}))
    return rules
