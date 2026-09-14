"""Edit constraints for IPUMS 1940: none.

The DAS 1940 configuration (configs/Census1940/DDP2010_Update/ipums_1940.ini) declares only
invariants (the total per state, set in the driver) and hhgq_total_lb/ub, bounds that depend on
household counts and cannot be written as rules over the person histogram.
"""


def constraints(columns):
    return []
