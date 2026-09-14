"""Declared domains for SINASC 2022 (Brazilian live births), 20 columns.

Source: DATASUS record layout "Estrutura do SINASC" (the DN form), transcribed by hand.
Every value is a string, and the blank '' is declared for every column because any dBase field
can be left empty. PESOGR and IDADEMAEG are the bins written by prepare.py.
"""

LAYOUT = {
    # Birth and delivery
    'LOCNASC': ['1', '2', '3', '4', '5', '9'],  # hospital, other facility, home, other, village, unknown
    'PARTO': ['1', '2', '9'],                   # vaginal, cesarean, unknown
    'STTRABPART': ['1', '2', '3', '9'],         # induced labour; 3 = não se aplica
    'STCESPARTO': ['1', '2', '3', '9'],         # cesarean before labour; 3 = não se aplica
    'TPAPRESENT': ['1', '2', '3', '9'],         # cephalic, breech, transverse, unknown
    'TPNASCASSI': ['1', '2', '3', '4', '9'],    # physician, nurse, midwife, other, unknown
    # The newborn
    'SEXO': ['0', '1', '2'],                    # unknown, male, female
    'RACACOR': ['1', '2', '3', '4', '5'],
    'PESOGR': ['1', '2', '3', '4'],             # <1500 g, <2500 g, <4000 g, 4000 g and more
    'IDANOMAL': ['1', '2', '9'],                # congenital anomaly detected
    # The pregnancy
    'GESTACAO': ['1', '2', '3', '4', '5', '6', '9'],  # weeks: <22, 22-27, 28-31, 32-36, 37-41, 42+
    'GRAVIDEZ': ['1', '2', '3', '9'],           # single, twin, triplet or more
    'CONSULTAS': ['1', '2', '3', '4', '9'],     # prenatal visits: none, 1-3, 4-6, 7+
    'PARIDADE': ['0', '1'],                     # nulliparous, multiparous
    # The mother
    'IDADEMAEG': ['1', '2', '3', '4'],          # <15, 15-19, 20-34, 35 and more
    'ESTCIVMAE': ['1', '2', '3', '4', '5', '9'],
    'ESCMAE': ['1', '2', '3', '4', '5', '9'],   # years of schooling, banded
    'RACACORMAE': ['1', '2', '3', '4', '5'],
    # The fine fields behind GESTACAO and CONSULTAS: two digits, no stated range
    'SEMAGESTAC': [f'{i:02d}' for i in range(100)],
    'CONSPRENAT': [f'{i:02d}' for i in range(100)],
}

DOMAINS = {column: sorted(set(values) | {''}) for column, values in LAYOUT.items()}

COLUMNS = list(DOMAINS)
