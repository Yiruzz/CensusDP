"""Run a command and record the peak memory of its whole process tree.

    python -m benchmarks.sample_memory OUT.json -- command args...

The command runs in a session of its own and orchestrate.MemorySampler samples it: the PSS of the
session, the instrument of the structures table, and the machine-wide peak with its baseline. It
gives benchmarks/ipums_1940/campaign.sh the same instrument as benchmarks/orchestrate.py. The exit
code of the command is passed through. Linux only.
"""

import json
import subprocess
import sys

from benchmarks.orchestrate import MemorySampler


def main():
    if len(sys.argv) < 4 or sys.argv[2] != '--':
        sys.exit(__doc__)
    out, command = sys.argv[1], sys.argv[3:]
    process = subprocess.Popen(command, start_new_session=True)
    sampler = MemorySampler(process.pid)
    sampler.start()
    try:
        code = process.wait()
    finally:
        with open(out, 'w', encoding='utf-8') as handle:
            json.dump(sampler.result(), handle)
    sys.exit(128 - code if code < 0 else code)  # killed by a signal: the shell's convention


if __name__ == '__main__':
    main()
