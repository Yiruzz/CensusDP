"""Query workloads for the full-joint IPUMS 1940 runs, for the comparison with the DAS.

'das' is the workload the 2020 DAS release measures on this schema, transcribed from
configs/Census1940/DDP2010_Update/ipums_1940.ini:

    dpqueries   = hhgq1940,
                  age1940 * hispanic1940 * cenrace1940 * citizen1940,
                  age1940 * sex1940,
                  ageGroups4 * sex1940,
                  ageGroups16 * sex1940,
                  ageGroups64 * sex1940,
                  detailed
    queriesprop = .2, .5, .05, .05, .05, .05, .1

The age recodes are the three levels of the hierarchical-by-4 tree over the 116 age values
(29, 8 and 2 groups), defined in
das_decennial/programs/schema/attributes/age1940.py::recodeAgeGroups{4,16,64}.

Each entry is (workload builder, per-block proportions). The proportions divide each level's
budget between the seven blocks - they do not add to it - which is what makes our noise match
theirs query by query rather than only on average. See TopDown.set_query_budget.
"""

from queries import QueryWorkload, Recode

# 116 age values, 0..115, where 115 means "115 or more".
AGE_TOP = 116


def age_groups(width, name):
    """The DAS's ageGroups<width> recode: consecutive bins of `width` over 0..AGE_TOP-1."""
    groups = {f'{start} to {min(start + width, AGE_TOP) - 1}':
              list(range(start, min(start + width, AGE_TOP)))
              for start in range(0, AGE_TOP, width)}
    return Recode('age', groups, name=name)


def das(columns):
    """The seven DAS query groups. `columns` is the full column list, for `detailed`."""
    return (QueryWorkload()
            .value_counts(['hhgq'])
            .value_counts(['age', 'hispanic', 'race', 'citizen'])
            .value_counts(['age', 'sex'])
            .value_counts([age_groups(4, 'ageGroups4'), 'sex'])
            .value_counts([age_groups(16, 'ageGroups16'), 'sex'])
            .value_counts([age_groups(64, 'ageGroups64'), 'sex'])
            .value_counts(list(columns)))


def detailed(columns):
    """Only the full joint, i.e. what --full-joint measures on its own. One block."""
    return QueryWorkload().value_counts(list(columns))


WORKLOADS = {
    'das': (das, [.2, .5, .05, .05, .05, .05, .1]),
    # The same seven queries with the budget split evenly. Not a DAS configuration: it is the
    # control that isolates what their allocation buys, since a uniform split over marginal
    # blocks is exactly what a single stacked measurement already does.
    'das_uniform': (das, [1 / 7] * 7),
    'detailed': (detailed, [1.0]),
}
