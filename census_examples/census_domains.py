"""Declared per-column domains for the Chilean 2017 census microdata.

Source: docs/manual_de_usuario_censo_2017_16r.pdf, chapter VI "Diccionario de variables",
which lists for every variable its response categories, its range and its type. That is the
questionnaire, not the data - which is the whole point. Inferring the domain with
SELECT DISTINCT (what DataHandler falls back to) makes the shape of the cell space depend on
the data and not DP safe.

Columns absent from the tables below are simply not returned; the caller then gets
DataHandler's inference and its warning, scoped to what is actually still missing.
"""

from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from .census_constraints import NOT_APPLICABLE, _sentinel

# A declared domain is (values, sentinels). Ranges come straight from the manual's
# "(min - max)" column; the sentinel pair is what follows it.
_Domain = Tuple[Union[range, Sequence], Tuple]

_LETTERS = tuple(chr(c) for c in range(ord("A"), ord("Z") + 1))

# ---------------------------------------------------------------------------------------
# Vivienda: questionnaire items 1 to 5.
# ---------------------------------------------------------------------------------------
_VIVIENDAS: Dict[str, _Domain] = {
    "P01":  (range(1, 11), ()),          # Dwelling type               (1-10), no sentinel
    "P02":  (range(1, 5),  ()),          # Dwelling occupancy          (1-4),  no sentinel
    "P03A": (range(1, 7),  (98, 99)),    # Exterior wall material
    "P03B": (range(1, 8),  (98, 99)),    # Roof covering material
    "P03C": (range(1, 6),  (98, 99)),    # Floor material
    "P04":  (range(0, 7),  (98, 99)),    # Rooms used as bedrooms
    "P05":  (range(1, 5),  (98, 99)),    # Water source
}

# ---------------------------------------------------------------------------------------
# Persona: questionnaire items 7 to 21.
# ---------------------------------------------------------------------------------------
_PERSONAS: Dict[str, _Domain] = {
    "P07":          (range(1, 20),     ()),            # Relationship to head of household (1-19)
    "P08":          (range(1, 3),      ()),            # Sex
    "P09":          (range(0, 101),    ()),            # Age; never "no aplica"
    "P10":          (range(1, 5),      (98, 99)),      # Usual residence
    "P10PAIS":      (range(0, 998),    (998, 999)),    # Country of usual residence
    "P11":          (range(1, 10),     (98, 99)),      # Residence 5 years ago
    "P11PAIS":      (range(0, 998),    (998, 999)),    # Country of residence 5 years ago
    "P12":          (range(1, 9),      (98, 99)),      # Place of birth
    "P12PAIS":      (range(0, 998),    (998, 999)),    # Country of birth
    "P12A_LLEGADA": (range(1950, 2018), (9998, 9999)),  # Year of arrival in the country
    "P12A_TRAMO":   (range(1, 5),      (98, 99)),      # Period of arrival in the country
    "P13":          (range(1, 4),      (98, 99)),      # Attends formal education
    "P14":          (range(0, 9),      (98, 99)),      # Highest grade or year completed
    "P15":          (range(1, 15),     (98, 99)),      # Level of the highest course completed
    "P15A":         (range(1, 3),      (98, 99)),      # Completed the specified level
    "P16":          (range(1, 3),      (98, 99)),      # Self-identifies as indigenous
    "P16A":         (range(1, 11),     (98, 99)),      # Indigenous people (listed)
    "P16A_OTRO":    (range(1, 98),     (98, 99)),      # Indigenous people (other)
    "P17":          (range(1, 9),      (98, 99)),      # Worked last week
    # P18 is CHARACTER: economic activity branches A..Z, and its sentinels are STRINGS.
    # With an integer domain, mask_compare('==', '98') silently selects 0 cells.
    "P18":          (_LETTERS,          ("98", "99")),  # Branch of economic activity
    "P19":          (range(0, 24),     (98, 99)),      # Total children ever born
    "P20":          (range(0, 24),     (98, 99)),      # Total children currently alive
    "P21M":         (range(1, 13),     (98, 99)),      # Month of birth of the last child
    # Deliberate union: the manual says 98/99, census_constraints.py declares 9998.
    "P21A":         (range(1890, 2018), (98, 99, 9998, 9999)),
}


def _build(table: Dict[str, _Domain], columns: Optional[Iterable[str]]) -> Dict[str, np.ndarray]:
    """Materialise the requested columns as sorted, de-duplicated value arrays.

    Sorting is not cosmetic. ContingencyDomain stores what it is handed verbatim
    (domain.py: `np.asarray(domains[c])`, no sort) while encode() resolves values with
    np.searchsorted, so an unsorted domain rejects VALID values with a message claiming the
    opposite: "Column 'X' contains values outside its declared domain: [4]".

    Args:
        table: One of _VIVIENDAS / _PERSONAS.
        columns: Columns of this run; None means every declared column. Columns absent from
            the table are skipped, so the caller falls back to DataHandler's inference for
            exactly those and gets its warning.

    Returns:
        Dict[str, np.ndarray]: Sorted value array per declared column.
    """
    requested = list(columns) if columns is not None else list(table)
    domain: Dict[str, np.ndarray] = {}
    for column in requested:
        if column not in table:
            continue
        values, sentinels = table[column]
        domain[column] = np.array(sorted(set(list(values) + list(sentinels))))
    return domain


# Columns that appear as the CONSEQUENT of some skip rule and therefore need their sentinel
# in the domain, even when they use the DEFAULT_SENTINEL and are not in NOT_APPLICABLE.
_CONSEQUENTS = frozenset({
    "P03A", "P03B", "P03C", "P04", "P05",                       # viviendas
    "P10COMUNA", "P10PAIS", "P11COMUNA", "P11PAIS", "P12COMUNA", "P12PAIS",
    "P12A_LLEGADA", "P12A_TRAMO", "P14", "P15", "P15A", "P16A", "P16A_OTRO",
    "P17", "P18", "P19", "P20", "P21M", "P21A",
})


def check_sentinels(domain: Dict[str, np.ndarray]) -> None:
    """Assert that every sentinel the edit constraints assert on is IN the declared domain.

    This is the coupling that motivates keeping domains in Python next to the constraints.
    A rule reads `Implies(antecedent, Equal(column, sentinel))`; if `sentinel` is not a value
    of `column`, the Equal selects no cell and the rule collapses into "force every cell of
    the antecedent to zero" - infeasible against SumEqualRealTotal, or silently destructive
    without it. Catching it here turns that into an error at configuration time.

    Args:
        domain: The dict returned by personas_domain / viviendas_domain.

    Raises:
        ValueError: If a declared column is missing the sentinel its constraints use.
    """
    missing = []
    for column, values in domain.items():
        if column not in NOT_APPLICABLE and column not in _CONSEQUENTS:
            continue
        sentinel = _sentinel(column)
        if not np.isin(np.array([sentinel], dtype=values.dtype), values).all():
            missing.append((column, sentinel))
    if missing:
        detail = ", ".join(f"{c}: missing {s!r}" for c, s in missing)
        raise ValueError(
            f"The declared domain does not contain the sentinel the edit constraints assert "
            f"on ({detail}). The skip rules on those columns would be unsatisfiable."
        )


def viviendas_domain(columns: Optional[Iterable[str]] = None) -> Dict[str, np.ndarray]:
    """Declared domain for the viviendas file, restricted to `columns`.

    Mirrors viviendas_constraints(available_columns): pass the run's query columns and get
    back only those, so the same call works for any --queries subset.

    Args:
        columns: Query columns of this run. None returns every declared column.

    Returns:
        Dict[str, np.ndarray]: Sorted values per column, ready for TopDown(domain=...).
    """
    domain = _build(_VIVIENDAS, columns)
    check_sentinels(domain)
    return domain


def personas_domain(columns: Optional[Iterable[str]] = None) -> Dict[str, np.ndarray]:
    """Declared domain for the personas file, restricted to `columns`.

    Mirrors personas_constraints(available_columns).

    P10COMUNA / P11COMUNA / P12COMUNA are NOT returned - see the module docstring. They stay
    on DataHandler's inference path until their codebook range is confirmed.

    Args:
        columns: Query columns of this run. None returns every declared column.

    Returns:
        Dict[str, np.ndarray]: Sorted values per column, ready for TopDown(domain=...).
    """
    domain = _build(_PERSONAS, columns)
    check_sentinels(domain)
    return domain
