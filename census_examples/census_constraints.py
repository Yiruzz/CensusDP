"""Edit constraints for the Chilean 2017 census microdata (data/constraints.txt).

These encode the questionnaire's skip logic: when a question does not apply, the field
carries a "no aplica" sentinel. They are structural zeros - combinations that cannot exist
- and the point of enforcing them is that the released microdata stays internally coherent.

THE SENTINELS ARE NOT UNIFORM, and the documentation disagrees with the data. Verified
against the 17.5M-row personas file:

  * COMUNA fields  (P10COMUNA, P11COMUNA, P12COMUNA)  -> 98    (documentation says 998)
  * PAIS fields    (P10PAIS, P11PAIS, P12PAIS)        -> 998
  * YEAR fields    (P12A_LLEGADA, P21A)               -> 9998  (they hold years)
  * P18            -> the STRING '98'  (P18 is VARCHAR: occupation codes include letters)
  * everything else                                   -> 98

DECOMPOSITION - the reason this is usable at all.

    "if P09 < 15 -> P17=98 & P18=98 & P19=98 & P20=98 & P21M=98 & P21A=9998"

as a single constraint has a scope of SEVEN columns. A constraint can only be enforced
inside a bag containing its whole scope, so that one clique would force a bag of
101*10*24*26*26*14*93 = 21,334,884,480 cells - larger than the full joint of the entire
viviendas dataset, and hopeless to measure or solve.

But  P -> (A & B)  is logically identical to  (P -> A) & (P -> B), so the same rule can be
written as six separate constraints of TWO columns each. The largest becomes
{P09, P21A} = 101*93 = 9,393 cells. Same semantics, same structural zeros, treewidth
reduced by nine orders of magnitude. Every rule below is therefore stored decomposed.
"""

from constraints.logical_expressions.atomic import Equal, LessThan, NotEqual
from constraints.logical_expressions.compound import And, Implies, Not, Or

# Sentinel used by each column when its question does not apply.
NOT_APPLICABLE = {
    "P10COMUNA": 98, "P11COMUNA": 98, "P12COMUNA": 98,
    "P10PAIS": 998, "P11PAIS": 998, "P12PAIS": 998,
    "P12A_LLEGADA": 9998, "P21A": 9998,
    "P18": "98",
}
DEFAULT_SENTINEL = 98


def _sentinel(column):
    return NOT_APPLICABLE.get(column, DEFAULT_SENTINEL)


def _skip_rules():
    """The questionnaire's skip logic as (antecedent, consequent columns) pairs.

    Each pair expands to one Implies per consequent column - see the module docstring on
    why the conjunctions are split rather than kept whole.
    """
    return [
        # Residencia habitual: extra detail only when living in another comuna or country.
        (Or(Equal("P10", 1), Equal("P10", 2)), ["P10COMUNA", "P10PAIS"]),
        (Equal("P10", 3), ["P10PAIS"]),
        (Equal("P10", 4), ["P10COMUNA"]),

        # Residencia hace 5 anios.
        (Or(*[Equal("P11", v) for v in (1, 2, 4, 5, 6, 7, 8)]), ["P11COMUNA", "P11PAIS"]),
        (Equal("P11", 3), ["P11PAIS"]),
        (Equal("P11", 9), ["P11COMUNA"]),

        # Lugar de nacimiento.
        (Equal("P12", 1), ["P12COMUNA", "P12PAIS", "P12A_LLEGADA", "P12A_TRAMO"]),
        (Equal("P12", 2), ["P12PAIS", "P12A_LLEGADA", "P12A_TRAMO"]),
        (Equal("P12", 8), ["P12COMUNA"]),
        (Or(*[Equal("P12", v) for v in (3, 4, 5, 6, 7)]), ["P12COMUNA", "P12PAIS"]),

        # Nunca estudio: no education follow-up.
        (Equal("P13", 3), ["P14", "P15", "P15A"]),

        # Pueblo originario. The second rule needs "and not the 99 = no responde code":
        # when P16A is 99, P16A_OTRO is 99 too, not 98 (497,927 rows in the data).
        (Equal("P16", 2), ["P16A", "P16A_OTRO"]),
        (And(NotEqual("P16A", 10), NotEqual("P16A", 99)), ["P16A_OTRO"]),

        # Under 15: no occupation or fertility questions.
        (LessThan("P09", 15), ["P17", "P18", "P19", "P20", "P21M", "P21A"]),

        # Did not work last week for these reasons: no occupation detail.
        (Or(*[Equal("P17", v) for v in (4, 5, 6, 7, 8)]), ["P18"]),

        # Men are not asked about children.
        (Equal("P08", 1), ["P19", "P20", "P21M", "P21A"]),

        # No children: no follow-up.
        (Equal("P19", 0), ["P20", "P21M", "P21A"]),
    ]


def personas_constraints(available_columns):
    """Build the personas constraints that the given column selection can express.

    A constraint is only included when EVERY column it mentions is being queried -
    otherwise its cells do not exist in the cell space at all.

    Args:
        available_columns (Iterable[str]): The query columns of this run.

    Returns:
        List[Implies]: Decomposed constraints, each over antecedent + one consequent column.
    """
    available = set(available_columns)
    constraints = []

    for antecedent, consequent_columns in _skip_rules():
        if not antecedent.scope() <= available:
            continue
        for column in consequent_columns:
            if column in available:
                constraints.append(Implies(antecedent, Equal(column, _sentinel(column))))

    return constraints


def viviendas_constraints(available_columns):
    """The viviendas rule: an unoccupied dwelling has no occupant characteristics.

    Args:
        available_columns (Iterable[str]): The query columns of this run.

    Returns:
        List[Implies]: Decomposed constraints.
    """
    available = set(available_columns)
    antecedent = NotEqual("P02", 1)
    if not antecedent.scope() <= available:
        return []

    consequents = {"P03A": 98, "P03B": 98, "P03C": 98, "P04": 98, "P05": 98,
                   "CANT_HOG": 0, "CANT_PER": 0}
    return [Implies(antecedent, Equal(column, value))
            for column, value in consequents.items() if column in available]
