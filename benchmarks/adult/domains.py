"""Declared domains for Adult.
"""

SIZES = {
    'age': 85, 'workclass': 9, 'fnlwgt': 100, 'education_num': 16, 'marital_status': 7,
    'occupation': 15, 'relationship': 6, 'race': 5, 'sex': 2, 'capital_gain': 100,
    'capital_loss': 100, 'hours_per_week': 99, 'native_country': 42, 'income': 2,
}

DOMAINS = {column: list(range(size)) for column, size in SIZES.items()}

COLUMNS = list(DOMAINS)
