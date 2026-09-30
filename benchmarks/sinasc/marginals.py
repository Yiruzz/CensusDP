"""Declared marginal structure for SINASC 2022.

'blocks' groups delivery, pregnancy, prenatal care and the mother by measured pairwise
association. At full depth it matched larger bags in TVD at a fifth of the time, because the
small municipalities make wide bags noisy.
"""

from benchmarks.structures import register_family
from .domains import DOMAINS
STRUCTURES = {
    'blocks': [
        ['PARTO', 'STCESPARTO', 'STTRABPART', 'TPAPRESENT', 'TPNASCASSI'],
        ['LOCNASC', 'TPNASCASSI', 'PARTO'],
        ['PARTO', 'GESTACAO', 'PESOGR', 'GRAVIDEZ'],
        ['SEXO', 'PESOGR', 'GESTACAO', 'IDANOMAL'],
        ['GESTACAO', 'SEMAGESTAC', 'PESOGR'],
        ['GESTACAO', 'CONSULTAS', 'IDADEMAEG', 'PARIDADE'],
        ['CONSULTAS', 'CONSPRENAT'],
        ['IDADEMAEG', 'PARIDADE', 'ESCMAE', 'ESTCIVMAE'],
        ['RACACOR', 'RACACORMAE', 'ESCMAE'],
    ],
}

DEFAULT = 'blocks'

# The derived family of DEFAULT, for experiment 6: benchmarks/structures.py.
DERIVED = register_family(STRUCTURES, DEFAULT, DOMAINS)
