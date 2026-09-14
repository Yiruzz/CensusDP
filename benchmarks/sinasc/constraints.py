"""Edit constraints for SINASC 2022, from the DN form and the DATASUS record layout.

- STCESPARTO ("cesarean before labour?") is 3 = não se aplica for a vaginal birth, and a
  cesarean cannot carry that code.
- "Nenhuma consulta" means the visit count is 00, or the field was left blank.
- GESTACAO and CONSULTAS are the layout's binnings of SEMAGESTAC and CONSPRENAT; the coarse
  field may be blank.
"""

from constraints.logical_expressions.atomic import Equal, NotEqual
from constraints.logical_expressions.compound import Implies, Or

GESTACAO_WEEKS = {'1': range(0, 22), '2': range(22, 28), '3': range(28, 32),
                  '4': range(32, 37), '5': range(37, 42), '6': range(42, 99)}
CONSULTAS_VISITS = {'1': range(0, 1), '2': range(1, 4), '3': range(4, 7), '4': range(7, 99)}
RECODINGS = [('GESTACAO', 'SEMAGESTAC', GESTACAO_WEEKS), ('CONSULTAS', 'CONSPRENAT', CONSULTAS_VISITS)]


def constraints(columns):
    rules = [
        Implies(Equal('PARTO', '1'), Equal('STCESPARTO', '3')),
        Implies(Equal('PARTO', '2'), NotEqual('STCESPARTO', '3')),
        Implies(Equal('CONSULTAS', '1'), Or(Equal('CONSPRENAT', '00'), Equal('CONSPRENAT', ''))),
    ]
    for coarse, fine, bins in RECODINGS:
        rules += [Implies(Equal(fine, f'{value:02d}'), Or(Equal(coarse, code), Equal(coarse, '')))
                  for code, values in bins.items() for value in values]
    return [rule for rule in rules if rule.scope() <= set(columns)]
