"""Build the SINASC 2022 Parquet from the 27 state .dbc files of DATASUS.

Sources: ftp://ftp.datasus.gov.br/dissemin/publicos/SINASC/NOV/DNRES/DN<UF>2022.dbc (records
by the mother's municipality of residence) and
ftp://ftp.datasus.gov.br/territorio/tabelas/2023/base_territorial_2023.zip (municipality to
health region). REGIAO and UF are the first one and two digits of the municipality code.
Records whose municipality is not in the territorial base cannot be placed and are dropped.
Birth weight and mother's age are binned at the WHO and Ministério da Saúde cut points.
"""

import csv
import io
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

from benchmarks import common
from . import dbc
from .domains import COLUMNS

DATASET = 'sinasc'
FOLDER = Path('data/sinasc')
UFS = ['RO', 'AC', 'AM', 'RR', 'PA', 'AP', 'TO', 'MA', 'PI', 'CE', 'RN', 'PB', 'PE', 'AL',
       'SE', 'BA', 'MG', 'ES', 'RJ', 'SP', 'PR', 'SC', 'RS', 'MS', 'MT', 'GO', 'DF']
SAMPLE = "UF = '14'"  # Roraima

WEIGHT_BINS = [('1', 0, 1500), ('2', 1500, 2500), ('3', 2500, 4000), ('4', 4000, 10000)]
AGE_BINS = [('1', 0, 15), ('2', 15, 20), ('3', 20, 35), ('4', 35, 200)]


def health_regions():
    with zipfile.ZipFile(FOLDER / 'base_territorial_2023.zip') as archive:
        text = archive.read('rl_municip_regsaud.csv').decode('latin-1')
    unknown = {'', '0000', '00000', '000000'}
    return {row['CO_MUNICIP']: row['CO_REGSAUD'] for row in csv.DictReader(io.StringIO(text))
            if row['CO_MUNICIP'] not in unknown and row['CO_REGSAUD'] not in unknown}


def binned(column, bins):
    value = f'TRY_CAST({column} AS INTEGER)'
    cases = ' '.join(f"WHEN {value} >= {low} AND {value} < {high} THEN '{code}'"
                     for code, low, high in bins)
    return f"CASE {cases} ELSE '' END"


if __name__ == '__main__':
    regions = health_regions()
    raw = [c for c in COLUMNS if c not in ('PESOGR', 'IDADEMAEG')]
    selection = ', '.join(['substr(CODMUNRES, 1, 1) AS REGIAO', 'substr(CODMUNRES, 1, 2) AS UF',
                           'REGSAUD', 'CODMUNRES'] + raw +
                          [f'{binned("PESO", WEIGHT_BINS)} AS PESOGR',
                           f'{binned("IDADEMAE", AGE_BINS)} AS IDADEMAEG'])
    con = duckdb.connect()
    for index, uf in enumerate(UFS):
        names, rows = dbc.read_dbc(FOLDER / f'DN{uf}2022.dbc')
        frame = pd.DataFrame(rows, columns=names, dtype=str)
        frame = frame[frame['CODMUNRES'].isin(regions.keys())]
        frame['REGSAUD'] = frame['CODMUNRES'].map(regions)
        verb = 'CREATE TABLE sinasc AS' if index == 0 else 'INSERT INTO sinasc'
        con.execute(f'{verb} SELECT {selection} FROM frame')
        print(f'{uf}: {len(frame):,} records')
    con.execute(f"COPY sinasc TO '{common.parquet(DATASET)}' (FORMAT PARQUET)")
    common.write_sample(DATASET, SAMPLE)
