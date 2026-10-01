"""Build the IPUMS 1940 Parquet with the recodes of the DAS 1940 reader.

Field positions: das_decennial/programs/reader/ipums_1940/ipums_1940_classes.py.
Recodes: person_recoder and map_to_hhgq in ipums_1940_reader.py. The DAS keeps RACE as 1..6
while declaring 0..5 legal; here it is shifted to 0..5.

    python -m benchmarks.ipums_1940.prepare          # sample only: Alaska
    python -m benchmarks.ipums_1940.prepare --full   # also the 132M-person full count
    python -m benchmarks.ipums_1940.prepare --units [--full]   # only the unit counts

Besides the persons it writes the unit counts the DAS holds invariant (`gqhh_vect`): occupied
households and group-quarters facilities per enumeration district and hhgq, one per H record, with
the same recode as the persons. benchmarks/ipums_1940/constraints.py turns them into bounds.
"""

import sys

import duckdb

from benchmarks import common

DATASET = 'ipums_1940'
RAW = {True: 'data/ipums_1940/EXT1940USCB_AK.dat', False: 'data/ipums_1940/EXT1940USCB.dat.gz'}

# map_to_hhgq in ipums_1940_reader.py, applied by the DAS to persons and units alike.
HHGQ = """CASE WHEN gqtype = 0 THEN 0
            WHEN gqtype IN (2, 3, 4) THEN gqtype - 1
            WHEN gqtype IN (6, 7, 8) THEN gqtype - 2
            WHEN gqtype = 9 AND gq IN (1, 2, 5) THEN 0
            WHEN gqtype = 9 THEN 7 END"""

QUERY = """
WITH lines AS (
    SELECT line FROM read_csv('{raw}', columns = {{'line': 'VARCHAR'}}, header = false,
                              delim = chr(1), quote = '', escape = '')
), households AS (
    SELECT substr(line, 8, 8) AS serial,
           substr(line, 54, 2) AS STATEFIP, substr(line, 56, 4) AS COUNTY,
           substr(line, 129, 3) AS SUPDIST, substr(line, 125, 4) AS ENUMDIST,
           CAST(substr(line, 94, 1) AS INTEGER) AS gq, CAST(substr(line, 95, 1) AS INTEGER) AS gqtype
    FROM lines WHERE substr(line, 1, 1) = 'H'
), persons AS (
    SELECT substr(line, 8, 8) AS serial,
           CAST(substr(line, 73, 1) AS INTEGER) AS sex, CAST(substr(line, 74, 3) AS INTEGER) AS age,
           CAST(substr(line, 85, 1) AS INTEGER) AS race, CAST(substr(line, 89, 1) AS INTEGER) AS hispan,
           CAST(substr(line, 118, 1) AS INTEGER) AS citizen
    FROM lines WHERE substr(line, 1, 1) = 'P'
)
SELECT STATEFIP, COUNTY, SUPDIST, ENUMDIST, {hhgq} AS hhgq,
       sex - 1 AS sex,
       LEAST(age, 115) AS age,
       CASE WHEN hispan = 0 THEN 0 ELSE 1 END AS hispanic,
       race - 1 AS race,
       CASE WHEN citizen IN (0, 1, 2) THEN 1 ELSE 0 END AS citizen
FROM persons JOIN households USING (serial)
"""


# The households CTE of QUERY, alone: the DAS reads every H record into its unit table.
UNITS = QUERY[:QUERY.index('), persons AS')] + """)
SELECT STATEFIP, COUNTY, SUPDIST, ENUMDIST, {hhgq} AS hhgq, count(*) AS units
FROM households GROUP BY ALL
"""


def units_path(sample):
    return common.parquet(DATASET, sample)[:-len('.parquet')] + '_units.parquet'


def build(sample):
    target = common.parquet(DATASET, sample)
    con = duckdb.connect()
    con.execute(f"COPY ({QUERY.format(raw=RAW[sample], hhgq=HHGQ)}) TO '{target}' (FORMAT PARQUET)")
    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{target}')").fetchone()[0]
    print(f'{target}: {rows:,} persons')


def build_units(sample):
    target = units_path(sample)
    con = duckdb.connect()
    con.execute(f"COPY ({UNITS.format(raw=RAW[sample], hhgq=HHGQ)}) TO '{target}' (FORMAT PARQUET)")
    rows = con.execute(f"SELECT hhgq, SUM(units) FROM read_parquet('{target}') GROUP BY 1 ORDER BY 1").fetchall()
    print(f'{target}: units per hhgq {rows}')


if __name__ == '__main__':
    steps = [build_units] if '--units' in sys.argv else [build, build_units]
    for step in steps:
        step(sample=True)
        if '--full' in sys.argv:
            step(sample=False)
