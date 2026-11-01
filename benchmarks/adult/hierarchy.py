"""Fabricated hierarchy for Adult: nested bands of the age code, 20, 10 and 5 codes wide.

Adult has no geography, so the tree is built from age. The age code (1..74) is read only to build
these columns and is not an attribute (domains.py), so the leaves partition the records like a
geography and a record cannot contradict its node.
"""

WIDTHS = {'AGE_20': 20, 'AGE_10': 10, 'AGE_5': 5}

HIERARCHY = list(WIDTHS)


def sql_columns():
    """The SELECT expressions that add the hierarchy columns, as VARCHAR."""
    return ', '.join(f'CAST(age // {width} AS VARCHAR) AS {name}' for name, width in WIDTHS.items())
