"""Build the Spanish 2021 census Parquet from INE's zips: the geography plus the 78 columns.

Source: https://www.ine.es/ftp/microdatos/censopv/cen21/CensoPersonas_2021.zip (INE ships the
Parquet inside) and dr_CensoPersonas_2021.zip (the record layout JSON that domains.py reads).
"""

import zipfile
from pathlib import Path

from benchmarks import common

DATASET = 'spanish_census'
FOLDER = Path('data/spanish_census')
MEMBERS = {'CensoPersonas_2021.zip': 'md_CensoPersonas_2021.parquet',
           'dr_CensoPersonas_2021.zip': 'dr_CensoPersonas_2021.json'}
SAMPLE = "CPRO = '42'"  # Soria

if __name__ == '__main__':
    for archive, member in MEMBERS.items():
        if not (FOLDER / member).exists():
            with zipfile.ZipFile(FOLDER / archive) as handle:
                handle.extract(member, FOLDER)

    from .domains import COLUMNS
    from .driver import HIERARCHY
    common.write_full(DATASET, (FOLDER / MEMBERS['CensoPersonas_2021.zip']).as_posix(),
                      HIERARCHY + COLUMNS)
    common.write_sample(DATASET, SAMPLE)
