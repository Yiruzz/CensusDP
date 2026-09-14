"""Build the Adult Parquet from Private-PGM's encoded copy (Research/mbi/data/adult.csv).

Column names are made SQL-safe, since the pipeline interpolates them unquoted.
"""

import duckdb

from benchmarks import common
from .hierarchy import sql_columns

DATASET = 'adult'
SOURCE = 'data/adult/adult.csv'
RENAME = {
    'education-num': 'education_num', 'marital-status': 'marital_status',
    'capital-gain': 'capital_gain', 'capital-loss': 'capital_loss',
    'hours-per-week': 'hours_per_week', 'native-country': 'native_country',
    'income>50K': 'income',
}
SAMPLE = "AGE_20 = '2'"

if __name__ == '__main__':
    renames = ', '.join(f'"{old}" AS {new}' for old, new in RENAME.items())
    duckdb.connect().execute(
        f"COPY (SELECT {sql_columns()}, * RENAME ({renames}) FROM read_csv('{SOURCE}')) "
        f"TO '{common.parquet(DATASET)}' (FORMAT PARQUET)")
    common.write_sample(DATASET, SAMPLE)
