"""Declared domains for IPUMS 1940: the DAS 1940 schema, 8*2*116*2*6*2 = 44,544 cells.

Source: das_decennial/programs/schema/attributes/{hhgq,sex,age,hisp,race,citizen}1940.py and
the `.legal` ranges in configs/Census1940/DDP2010_Update/ipums_1940.ini.
"""

DOMAINS = {
    'hhgq': list(range(8)),     # household, then 7 group-quarters types
    'sex': [0, 1],
    'age': list(range(116)),    # 115 means 115 or more
    'hispanic': [0, 1],
    'race': list(range(6)),     # white, black, AIAN, chinese, japanese, other API
    'citizen': [0, 1],
}

COLUMNS = list(DOMAINS)
