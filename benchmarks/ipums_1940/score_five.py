import json
import os
import sys

from benchmarks import common

DATASET = 'ipums_1940'


def main(name):
    with open(common.out_path(DATASET, f'{name}.json'), encoding='utf-8') as handle:
        run = json.load(handle)
    if run.get('status') != 'ok':
        sys.exit(f'{name} did not finish.')
    run['columns'] = [column for column in run['columns'] if column != 'citizen']
    run['scored_over'] = 'the five columns the DAS MDF publishes; citizen is measured by neither'
    target = f'{name}_5col'
    with open(common.out_path(DATASET, f'{target}.json'), 'w', encoding='utf-8') as handle:
        json.dump(run, handle, indent=1)

    csv, link = (common.out_path(DATASET, f'{part}.csv') for part in (name, target))
    if os.path.exists(link):
        os.remove(link)
    os.link(csv, link)
    print(target)


if __name__ == '__main__':
    main(sys.argv[1])
