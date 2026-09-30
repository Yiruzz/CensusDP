"""Declared marginal structures for IPUMS 1940. --full-joint ignores them.

The DAS 1940 workload (configs/Census1940/DDP2010_Update/ipums_1940.ini, dpqueries) measures
hhgq, age x hispanic x race x citizen and age x sex, plus age-group recodes of age x sex.
'das' takes its two cross-tabs as bags, with hhgq joined to age x sex.
"""

from benchmarks.structures import register_family
from .domains import DOMAINS
STRUCTURES = {
    'das': [['age', 'hispanic', 'race', 'citizen'], ['age', 'sex', 'hhgq']],
    'das_sex': [['age', 'sex', 'hispanic', 'race', 'citizen'], ['age', 'sex', 'hhgq']],
    'joint': [['hhgq', 'sex', 'age', 'hispanic', 'race', 'citizen']],
}

DEFAULT = 'das'

# The derived family of DEFAULT, for experiment 6: benchmarks/structures.py.
# das_m1 is the full joint in one bag, which is also the declared 'joint'.
DERIVED = register_family(STRUCTURES, DEFAULT, DOMAINS)

# The DAS queries without the age-group recodes, reported on their own by benchmarks/metrics.py.
WORKLOAD = [['hhgq'], ['age', 'hispanic', 'race', 'citizen'], ['age', 'sex']]
