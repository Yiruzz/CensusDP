"""Turn a DAS MDF release into the CSV and run record that benchmarks.metrics reads.

The point is that both systems get scored by the same code. metrics.py takes a
semicolon-delimited CSV of microdata plus a run JSON, so the DAS's output only has
to be expressed in our column space -- and it can be, exactly, because every
recode in the release's 1940 writer is injective:

    our column   MDF column   forward (ipums_1940_writer.py)   inverse here
    age          QAGE         age                              QAGE
    race         CENRACE      race + 1, zero-padded to 2       CENRACE - 1
    hispanic     CENHISP      0 -> '1', else '2'               CENHISP - 1
    sex          QSEX         0 -> '1', else '2'               QSEX - 1
    hhgq         GQTYPE       0->000 1->101 2->201 3->301       the same map,
                              4->401 5->501 6->601 else 701     read backwards

CITIZEN is absent. `IPUMSPersonWriter.var_list` (ipums_1940_writer.py:246) has it
commented out, so the DAS spends budget on that dimension -- it appears in the
`detailed` query and in `age * hispanic * cenrace * citizen`, 60% of the
within-level allocation between them -- and then does not release it. Nothing here
can recover it. The comparison therefore runs over five of the six columns, which
is what the DAS actually publishes, and both sides are scored the same way.
"""

import argparse
import json
import os
import sys

import duckdb

from benchmarks import common

DATASET = 'ipums_1940'
HIERARCHY = ['STATEFIP', 'COUNTY', 'SUPDIST', 'ENUMDIST']
# The five the MDF carries, in our names. citizen is not among them; see above.
COLUMNS = ['hhgq', 'sex', 'age', 'hispanic', 'race']

# GQTYPE back to hhgq. Forward map at ipums_1940_writer.py:162; injective over
# 0..7, so this is its inverse and not an approximation.
GQTYPE_TO_HHGQ = {'000': 0, '101': 1, '201': 2, '301': 3,
                  '401': 4, '501': 5, '601': 6, '701': 7}


def read_header(release_dir):
    """The MDF column order, from the 1_header file the writer emits beside the parts."""
    path = os.path.join(release_dir, '1_header')
    if not os.path.isfile(path):
        sys.exit(f'no 1_header in {release_dir}. Was write_metadata on?')
    with open(path, encoding='utf-8') as handle:
        names = handle.readline().strip().split('|')
    missing = {'TABBLKST', 'TABBLKCOU', 'SUPDIST', 'ENUMDIST',
               'QSEX', 'QAGE', 'CENHISP', 'CENRACE', 'GQTYPE'} - set(names)
    if missing:
        sys.exit(f'1_header is missing {sorted(missing)}; got {names}')
    if 'CITIZEN' in names:
        print('NOTE: this release carries CITIZEN. It was produced with a writer that '
              'does not drop it, so a six-column comparison is possible -- but this '
              'script still emits five. Edit COLUMNS if that is what you want.')
    return names


def convert(release_dir, name, epsilon):
    """Write <name>.csv in our column space, and <name>.json describing the run."""
    names = read_header(release_dir)
    parts = os.path.join(release_dir, 'part-*.csv')
    target = common.out_path(DATASET, f'{name}.csv')
    os.makedirs(os.path.dirname(target), exist_ok=True)

    gq_cases = ' '.join(f"WHEN '{code}' THEN {value}"
                        for code, value in GQTYPE_TO_HHGQ.items())
    con = duckdb.connect()
    # The part files carry no header of their own; 1_header is a separate file,
    # which is why the column names are supplied here.
    columns = ', '.join(f"'{n}': 'VARCHAR'" for n in names)
    select = f"""
        SELECT
            TABBLKST                                   AS STATEFIP,
            TABBLKCOU                                  AS COUNTY,
            SUPDIST                                    AS SUPDIST,
            ENUMDIST                                   AS ENUMDIST,
            CASE GQTYPE {gq_cases} END                 AS hhgq,
            CAST(QSEX    AS INTEGER) - 1               AS sex,
            CAST(QAGE    AS INTEGER)                   AS age,
            CAST(CENHISP AS INTEGER) - 1               AS hispanic,
            CAST(CENRACE AS INTEGER) - 1               AS race
        FROM read_csv('{parts}', delim='|', header=false, columns={{{columns}}})
    """
    con.execute(f"COPY ({select}) TO '{target}' (FORMAT CSV, DELIMITER ';', HEADER)")

    stats = con.execute(f"""
        SELECT count(*), count(hhgq), min(age), max(age),
               min(race), max(race), count(DISTINCT STATEFIP)
        FROM ({select})
    """).fetchone()
    records, hhgq_ok, age_lo, age_hi, race_lo, race_hi, states = stats

    print(f'{target}')
    print(f'  records          {records:,}')
    print(f'  states           {states}')
    print(f'  age range        {age_lo}..{age_hi}          (expected 0..115)')
    print(f'  race range       {race_lo}..{race_hi}            (expected 0..5)')
    if hhgq_ok != records:
        sys.exit(f'  hhgq: {records - hhgq_ok:,} rows have a GQTYPE outside '
                 f'{sorted(GQTYPE_TO_HHGQ)} and became NULL. The forward map changed.')
    if not (0 <= age_lo and age_hi <= 115 and 0 <= race_lo and race_hi <= 5):
        sys.exit('  a recoded value fell outside its declared domain; the inverse is wrong.')
    print('  hhgq             every row mapped')

    run = {
        'status': 'ok',
        'system': 'das_2020_dhc',
        'dataset': DATASET,
        'columns': COLUMNS,
        'hierarchy': HIERARCHY,
        'bags': None,           # full-joint, so metrics.py uses [columns]
        'sample': False,
        'epsilon': epsilon,
        'records': records,
        'release_dir': release_dir,
        'note': 'CITIZEN is not in the MDF; the comparison is over five columns.',
    }
    meta = os.path.join(release_dir, '0_metadata')
    if os.path.isfile(meta):
        with open(meta, encoding='utf-8') as handle:
            run['das_metadata'] = handle.read()
    with open(common.out_path(DATASET, f'{name}.json'), 'w', encoding='utf-8') as handle:
        json.dump(run, handle, indent=1)
    print(f'{common.out_path(DATASET, f"{name}.json")}')
    print(f'\nnext:  python -m benchmarks.metrics {DATASET} {name} --discard')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('release_dir', help='the directory holding part-*.csv and 1_header')
    parser.add_argument('name', help='run name, e.g. das_rep1')
    parser.add_argument('--epsilon', type=float, default=4.0)
    args = parser.parse_args()
    convert(args.release_dir, args.name, args.epsilon)


if __name__ == '__main__':
    main()
