"""Declared marginal structures for the Chilean 2017 census. The constraint scopes are added as
bags automatically.
"""

STRUCTURES = {
    'baseline': [
        ['P07', 'P08', 'P09'],  # relationship x sex x age
        ['P08', 'P09', 'P13'],  # sex x age x attends education
        ['P13', 'P14', 'P15'],  # education detail
        ['P08', 'P09', 'P16'],  # sex x age x indigenous
        ['P08', 'P09', 'P17'],  # sex x age x worked last week
        ['P09', 'P12'],         # age x place of birth
        ['P10', 'P11'],         # residence now x five years ago
        ['P21M', 'P21A'],       # month x year of the last child, which no rule connects
    ],
}

DEFAULT = 'baseline'
