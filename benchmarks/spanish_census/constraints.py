"""Edit constraints for the Spanish 2021 census: the two blocks INE's record layout documents.

Source: dr_CensoPersonas_2021.json. T_TIPPE gives TIPOPER 'C' = "Vive con cónyuge" and
'N' = "No pertenece a ningún núcleo familiar"; the spouse columns carry 'N' = "No convive con el
progenitor 1 ó 2 o cónyuge" and the nucleus columns '9' = "No es aplicable".
Rules over the empty string (not working, not studying, under 15) are not declared: the layout
never documents that value.
"""

from constraints.logical_expressions.atomic import Equal, NotEqual
from constraints.logical_expressions.compound import Equivalent

NO_SPOUSE = {'SEXO_CON': 'N', 'ECIVL_CON': 'N', 'RELA_CON': 'N', 'ESREAL_CON_GR5': 'N'}
# TAM_NUC and NHIJOS_NUC are integers in INE's Parquet.
NO_NUCLEUS = {'TIPO_NUC': '9', 'TIPO_PAR_NUC1': '9', 'TIPO_PAR_NUC2': '9',
              'TAM_NUC': 9, 'NHIJOS_NUC': 9}

RULES = [(NotEqual('TIPOPER', 'C'), NO_SPOUSE), (Equal('TIPOPER', 'N'), NO_NUCLEUS)]


def constraints(columns):
    """Each rule both ways: the sentinel appears exactly when the antecedent holds."""
    available = set(columns)
    if 'TIPOPER' not in available:
        return []
    return [Equivalent(antecedent, Equal(column, sentinel))
            for antecedent, block in RULES
            for column, sentinel in block.items() if column in available]
