"""Declared marginal structures for Adult. Runs use DEFAULT unless --structure says otherwise.
"""

from .domains import COLUMNS

STRUCTURES = {
    # Every column paired with the label.
    'baseline': [[column, 'income'] for column in COLUMNS if column != 'income'],
    'blocks': [
        ['marital_status', 'relationship', 'sex', 'income'],
        ['workclass', 'occupation', 'education_num', 'income'],
        ['capital_gain', 'capital_loss', 'income'],
        ['race', 'native_country', 'income'],
        ['age', 'marital_status'],
        ['hours_per_week', 'occupation'],
        ['fnlwgt', 'income'],
    ],
    'large': [
        ['age', 'marital_status', 'relationship', 'sex', 'income'],
        ['workclass', 'occupation', 'education_num', 'income'],
        ['occupation', 'hours_per_week', 'income'],
        ['capital_gain', 'capital_loss', 'income'],
        ['race', 'native_country', 'income'],
        ['fnlwgt', 'race'],
    ],
}

DEFAULT = 'blocks'
