"""Edit constraints for the Chilean 2017 census, personas: the questionnaire's skip logic.

Source: docs/manual_de_usuario_censo_2017_16r.pdf (chapter VI) and INE's label files, which give
each column's "No aplica" code. `P -> (A & B)` is split into one rule per column so no rule
forces a wide bag. P21A has no rules: the manual gives 98 as its "No aplica", the file uses 9998.
"""

from collections import defaultdict

from constraints.logical_expressions.atomic import Equal, LessThan, NotEqual
from constraints.logical_expressions.compound import And, Equivalent, Implies, Or

NOT_APPLICABLE = {
    'P10COMUNA': 98, 'P11COMUNA': 98, 'P12COMUNA': 98,  # etiquetas_persona_comuna_16r.csv
    'P10PAIS': 998, 'P11PAIS': 998, 'P12PAIS': 998,     # etiquetas_persona_pais.csv
    'P12A_LLEGADA': 9998,                                # manual, chapter VI
    'P18': '98',                                         # etiquetas_persona_p18.csv, a string
}

SKIP_RULES = [
    # Usual residence: comuna and country only for someone living elsewhere.
    (Or(Equal('P10', 1), Equal('P10', 2)), ['P10COMUNA', 'P10PAIS']),
    (Equal('P10', 3), ['P10PAIS']),
    (Equal('P10', 4), ['P10COMUNA']),
    # Residence five years ago.
    (Or(*[Equal('P11', v) for v in (1, 2, 4, 5, 6, 7, 8)]), ['P11COMUNA', 'P11PAIS']),
    (Equal('P11', 3), ['P11PAIS']),
    (Equal('P11', 9), ['P11COMUNA']),
    # Place of birth.
    (Equal('P12', 1), ['P12COMUNA', 'P12PAIS', 'P12A_LLEGADA', 'P12A_TRAMO']),
    (Equal('P12', 2), ['P12PAIS', 'P12A_LLEGADA', 'P12A_TRAMO']),
    (Equal('P12', 8), ['P12COMUNA']),
    (Or(*[Equal('P12', v) for v in (3, 4, 5, 6, 7)]), ['P12COMUNA', 'P12PAIS']),
    # Never attended education.
    (Equal('P13', 3), ['P14', 'P15', 'P15A']),
    # Indigenous people. P16A_OTRO details P16A = 10; a missing P16A (99) is left out of the rule.
    (Equal('P16', 2), ['P16A', 'P16A_OTRO']),
    (And(NotEqual('P16A', 10), NotEqual('P16A', 99)), ['P16A_OTRO']),
    # Under 15: no work or fertility questions.
    (LessThan('P09', 15), ['P17', 'P18', 'P19', 'P20', 'P21M']),
    # Did not work last week for these reasons: no branch of activity.
    (Or(*[Equal('P17', v) for v in (4, 5, 6, 7, 8)]), ['P18']),
    # Men are not asked about children.
    (Equal('P08', 1), ['P19', 'P20', 'P21M']),
    # No children: no follow-up.
    (Equal('P19', 0), ['P20', 'P21M']),
]


def constraints(columns):
    """Every skip rule, plus its converse: "No aplica" appears only where some skip forces it."""
    available = set(columns)
    rules = []
    causes = defaultdict(list)
    for antecedent, targets in SKIP_RULES:
        for column in targets:
            causes[column].append(antecedent)
            if column in available and antecedent.scope() <= available:
                rules.append(Implies(antecedent, Equal(column, NOT_APPLICABLE.get(column, 98))))
    for column, antecedents in causes.items():
        if {column}.union(*(a.scope() for a in antecedents)) <= available:
            sentinel = Equal(column, NOT_APPLICABLE.get(column, 98))
            rules.append(Equivalent(antecedents[0], sentinel) if len(antecedents) == 1
                         else Implies(sentinel, Or(*antecedents)))
    return rules
