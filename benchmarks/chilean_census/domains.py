"""Declared domains for the Chilean 2017 census, personas.

Source: docs/manual_de_usuario_censo_2017_16r.pdf, chapter VI "Diccionario de variables" (range
and "No aplica"/"Missing" codes per question), and INE's label file
etiquetas_persona_comuna_16r.csv for the comuna codes.
"""

import csv

COMUNA_LABELS = 'data/csv-personas-censo-2017/etiquetas_persona_comuna_16r.csv'


def _values(values, sentinels=(98, 99)):
    return sorted(set(values) | set(sentinels))


with open(COMUNA_LABELS, encoding='utf-8-sig') as handle:
    COMUNAS = sorted(int(row['valor']) for row in csv.DictReader(handle, delimiter=';'))

LETTERS = [chr(code) for code in range(ord('A'), ord('Z') + 1)]

DOMAINS = {
    'P07': _values(range(1, 20), ()),                      # relationship to the household head
    'P08': _values(range(1, 3), ()),                       # sex
    'P09': _values(range(0, 101), ()),                     # age
    'P10': _values(range(1, 5)),                           # usual residence
    'P10COMUNA': COMUNAS,
    'P10PAIS': _values(range(0, 998), (998, 999)),
    'P11': _values(range(1, 10)),                          # residence five years ago
    'P11COMUNA': COMUNAS,
    'P11PAIS': _values(range(0, 998), (998, 999)),
    'P12': _values(range(1, 9)),                           # place of birth
    'P12COMUNA': COMUNAS,
    'P12PAIS': _values(range(0, 998), (998, 999)),
    'P12A_LLEGADA': _values(range(1950, 2018), (9998, 9999)),
    'P12A_TRAMO': _values(range(1, 5)),
    'P13': _values(range(1, 4)),                           # attends formal education
    'P14': _values(range(0, 9)),                           # highest grade completed
    'P15': _values(range(1, 15)),                          # level of that grade
    'P15A': _values(range(1, 3)),
    'P16': _values(range(1, 3)),                           # indigenous
    'P16A': _values(range(1, 11)),
    'P16A_OTRO': _values(range(1, 98)),
    'P17': _values(range(1, 9)),                           # worked last week
    'P18': _values(LETTERS, ('98', '99')),                 # branch of activity, a string column
    'P19': _values(range(0, 24)),                          # children born alive
    'P20': _values(range(0, 24)),                          # children alive
    'P21M': _values(range(1, 13)),                         # month of birth of the last child
    'P21A': _values(range(1890, 2018), (98, 99, 9998, 9999)),  # manual: 98/99, file: 9998/9999
}

COLUMNS = list(DOMAINS)
