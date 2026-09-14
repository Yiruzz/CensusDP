"""Declared domains for the Spanish 2021 census (INE), 78 question columns.

Source: INE's record layout dr_CensoPersonas_2021.json, whose codelists give most domains.
Exceptions: codelists that only list their top category are written out (MANUAL), shared
codelists are narrowed to what the column can take (NARROW), and two values the codelists omit
but the file carries are added: the empty string for questions that do not apply, and the
arrival years 2016-2020 that T_ANOP / T_ANOL do not band.
"""

import json

LAYOUT = 'data/spanish_census/dr_CensoPersonas_2021.json'

COLUMNS = [
    # Demography
    'SEXO', 'VAREDAD', 'MNAC', 'ECIVIL', 'TIPO_MUN_DEGURBA',
    # Birth and nationality
    'PNACIO', 'PNACIM', 'CPRO_NAC', 'RESI_NACIM',
    # Migration
    'RESI_ANT', 'RESI_UNANO', 'RESI_DANO',
    'PAIS_ANT', 'PAIS_PR', 'PAIS_UNANO', 'PAIS_DANO',
    'CPRO_ANT', 'CPRO_UNANO', 'CPRO_DANO', 'PROV_PR',
    'VARANORES', 'VARANOM', 'ANOP', 'VARANOC', 'VARANOE',
    # Education
    'ESREAL_CNEDA', 'ESCUR', 'ESCUR2', 'TESCUR', 'LEST', 'CPRO_EST',
    # Work
    'RELA', 'SITU', 'OCU63', 'ACT89', 'LTRAB', 'CPRO_TRAB',
    # Dwelling
    'TIPO_EDIF_VIV', 'SUP_VIV', 'TENEN_VIV', 'SUP_OCU_VIV',
    'NPLANTAS_SOBRE_EDIF', 'NPLANTAS_BAJO_EDIF', 'ANO_CONS',
    # Household and nucleus (FAM_HOG is left out: it is a function of TIPO_HOG)
    'TIPOPER', 'TAM_HOG', 'NUC_HOG', 'ESTRUC_HOG', 'TIPO_HOG',
    'TIPO_NUC', 'TAM_NUC', 'NHIJOS_NUC', 'TIPO_PAR_NUC1', 'TIPO_PAR_NUC2',
    # Spouse
    'SEXO_CON', 'VAREDAD_CON', 'ECIVL_CON', 'RELA_CON', 'SITU_CON',
    'ESREAL_CON_GR5', 'PNACIM_CON_GR9', 'NACIO_CON_GR10',
    # Parent 1
    'SEXO_MAD', 'VAREDAD_MAD', 'ECIVL_MAD', 'RELA_MAD', 'SITU_MAD',
    'ESREAL_MAD_GR5', 'PNACIM_MAD_GR9', 'NACIO_MAD_GR10',
    # Parent 2
    'SEXO_PAD', 'VAREDAD_PAD', 'ECIVL_PAD', 'RELA_PAD', 'SITU_PAD',
    'ESREAL_PAD_GR5', 'PNACIM_PAD_GR9', 'NACIO_PAD_GR10',
]

# T_EDAD lists only '100' and 'N'. The Parquet stores ages unpadded: '0', '1', ... '100'.
AGES = [str(age) for age in range(101)]
MANUAL = {
    'VAREDAD': AGES, 'VAREDAD_CON': AGES + ['N'], 'VAREDAD_MAD': AGES + ['N'],
    'VAREDAD_PAD': AGES + ['N'],
    # Integers in INE's Parquet; T_TAMHN, T_NNUCL and T_NHIJO list only the top value and 9.
    'TAM_HOG': [1, 2, 3, 4, 5, 9], 'TAM_NUC': [1, 2, 3, 4, 5, 9],
    'NUC_HOG': [0, 1, 2, 3, 9], 'NHIJOS_NUC': [0, 1, 2, 3, 4, 9],
}

# 'N' ("does not live with parent 1/2 or spouse") only applies to the relatives' columns.
NARROW = {'SEXO': ['1', '6'], 'ECIVIL': ['0', '1', '2', '3', '4'],
          'RELA': ['1', '2', '4', '5', '6', '7']}

EMPTY = {
    'ESREAL_CNEDA', 'ESCUR', 'ESCUR2', 'TESCUR', 'RELA', 'SITU', 'OCU63', 'ACT89', 'LTRAB',
    'VARANORES', 'CPRO_ANT', 'CPRO_DANO', 'CPRO_EST', 'CPRO_TRAB', 'CPRO_UNANO', 'PROV_PR',
    'PAIS_ANT', 'SITU_CON', 'SITU_MAD', 'SITU_PAD',
}

RECENT_YEARS = [str(year) for year in range(2016, 2021)]


def _values(column, layout):
    if column in MANUAL:
        return sorted(MANUAL[column])
    ref = next(field['codelist_ref'] for field in layout['layout'] if field['name'] == column)
    values = list(NARROW.get(column, layout['codelists'][ref]))
    if ref in ('T_ANOP', 'T_ANOL'):
        values += RECENT_YEARS
    if column in EMPTY:
        values.append('')
    return sorted(set(values))


with open(LAYOUT, encoding='utf-8-sig') as handle:
    _layout = json.load(handle)

DOMAINS = {column: _values(column, _layout) for column in COLUMNS}
