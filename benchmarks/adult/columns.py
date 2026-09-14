"""Inspect the Adult Parquet against the declared domains. Pass --sample for the subset."""

import sys

from benchmarks import common
from .domains import DOMAINS
from .driver import DATASET
from .hierarchy import HIERARCHY

if __name__ == '__main__':
    common.inspect_columns(common.parquet(DATASET, '--sample' in sys.argv), DOMAINS, HIERARCHY)
