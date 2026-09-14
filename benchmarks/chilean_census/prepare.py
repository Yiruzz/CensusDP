"""Build the Chilean 2017 census Parquet: the geography plus the 27 question columns of personas.

Source: INE's personas microdata (data/csv-personas-censo-2017), already converted to Parquet.
"""

from benchmarks import common
from .domains import COLUMNS
from .driver import DATASET, HIERARCHY

SOURCE = ('data/csv-personas-censo-2017/microdato_censo2017-personas/'
          'Microdato_Censo2017-Personas.parquet')
SAMPLE = 'REGION = 11'  # Aysén

if __name__ == '__main__':
    common.write_full(DATASET, SOURCE, HIERARCHY + COLUMNS)
    common.write_sample(DATASET, SAMPLE)
