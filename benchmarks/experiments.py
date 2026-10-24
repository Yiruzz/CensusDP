"""The runs each experiment of the paper needs, as (dataset, configuration) pairs.

A configuration spells out every runner flag that tells two runs apart, so a run that several
experiments share (the reference run of a dataset serves experiments 4 to 7) has a single name
and runs once. benchmarks/orchestrate.py runs them; benchmarks/README.md is the plan.

Part 1 compares against the DAS with the DAS budget: experiment 1 and the performance table run
from benchmarks/ipums_1940/campaign.sh, and experiment 2's scaling sweep is listed here, at the
Part 2 budget, since it is not compared with the DAS. Part 2 (experiments 3 to 7) starts with experiment 3, in two steps:
  1. exp3_composition: every composition at COMPOSITION_RHO. The winner goes in COMPOSITION.
  2. exp3_budget: every rho with COMPOSITION. The chosen one goes in RHO.
Until both are set, an experiment that needs them refuses to list its runs.
Experiment 4 reruns the reference configuration, because the synthetic CSVs are not kept and its
triples are sampled differently (see exp4).
"""

import importlib
import math

from benchmarks.common import COMPOSITIONS

DATASETS = ('adult', 'sinasc', 'spanish_census', 'chilean_census', 'ipums_1940')
RHOS = (0.1, 0.5, 1, 2, 5)
COMPOSITION_RHO = 1
COMPOSITION = 'sqrt'  # exp3_composition, 2026-09-22
RHO = 1               # exp3_budget, 2026-09-23
# Its structure is decided by its constraint scopes, not by its marginals -- see exp6.
EXP6_EXCLUDED = ('chilean_census',)
TRIPLES = 100
# Experiment 8: the flat synthesizers of benchmarks/baselines, MST first so that AIM, which may
# not finish on the large datasets, comes last in every round.
BASELINES = ('mst', 'aim')
# AIM on Spain does not finish: 3,160 candidates and 1,264 rounds at ~17 min a round (~15 days),
# and XLA fails compiling after 21 rounds (5.9 h, 36 GB PSS, 82 GB free; 2026-10-23). Its one
# round-1 attempt is the record; further rounds would repeat it.
EXP8_EXCLUDED = (('spanish_census', 'aim'),)
# Experiment 2: prefixes of the 78 ESCENSUS columns, dense where the full-joint arm still runs.
EXP2_COLUMNS = (2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 78)
# The full-joint measurement is one dense float64 vector of the joint per node. Its arm runs until
# one such vector alone exceeds the RAM of the server (123.5 GB, d = 8), that point included: the
# crash is the finding, and every larger d fails the same way, so the figure draws its width only.
FULL_JOINT_MAX_CELLS = 123.5e9 / 8


def decided(config):
    if None in config.values():
        raise SystemExit('Set COMPOSITION (after exp3_composition) and RHO (after exp3_budget) '
                         'in benchmarks/experiments.py first.')
    return config


def marginal(dataset, **changes):
    """The reference run of a dataset (its default structure, swept), with some flags changed."""
    structure = importlib.import_module(f'benchmarks.{dataset}.marginals').DEFAULT
    return decided({'structure': structure, 'rho': RHO, 'composition': COMPOSITION,
                    'rounding': 'sweep', **changes})


def das_budget():
    raise SystemExit('Experiments 1 and 2 run with the DAS budget (pure DP, epsilon = 4), which the '
                     'runner does not support yet: see benchmarks/README.md, Part 1.')


def exp1():
    """Utility of ours (full-joint) against the DAS on IPUMS 1940, both with the DAS budget."""
    das_budget()


def exp2():
    """Width, time and memory against the number of attributes on ESCENSUS, both pipelines.

    The first d columns of the questionnaire (`--columns d`): the marginal arm keeps every declared
    bag cut to them, and the edit rules whose columns are all kept, up to the 78 of the reference
    run; the full-joint arm measures their whole joint. Utility is not compared, since the TVD of
    different sets of attributes is not comparable, so every run samples its memory instead and
    skips the metrics, like exp6_memory. At d = 78 the marginal configuration is exp6_memory's
    declared row of this dataset, and is run once.

    ESCENSUS rather than IPUMS 1940, whose joint over its 6 attributes is 44,544 cells, so the
    full-joint arm never breaks there and the DAS is a single point already in Table 3. Here the
    joint passes 10^6 cells at d = 6 and 10^9 at d = 7, and the arm is expected to run out of
    memory (see FULL_JOINT_MAX_CELLS): that crash, against the marginal width, is the result. Both
    arms share every flag but the pipeline and the rounding, which is 'mip' because the sweep is
    undefined on the full joint.
    """
    dataset = 'spanish_census'
    domains = importlib.import_module(f'benchmarks.{dataset}.domains')
    runs = []
    for d in EXP2_COLUMNS:
        changes = {'columns': d} if d < len(domains.COLUMNS) else {}
        runs.append((dataset, marginal(dataset, **changes, memory=1)))
    for d in EXP2_COLUMNS:
        runs.append((dataset, decided({'full_joint': True, 'rho': RHO, 'composition': COMPOSITION,
                                       'rounding': 'mip', 'columns': d, 'memory': 1})))
        if math.prod(len(domains.DOMAINS[c]) for c in domains.COLUMNS[:d]) > FULL_JOINT_MAX_CELLS:
            break  # the joint only grows with d
    return runs


def exp3_composition():
    """Utility per tree level of every composition, at COMPOSITION_RHO."""
    return [(dataset, marginal(dataset, rho=COMPOSITION_RHO, composition=composition))
            for dataset in DATASETS for composition in COMPOSITIONS]


def exp3_budget():
    """Utility per tree level of every budget, with COMPOSITION."""
    return [(dataset, marginal(dataset, rho=rho)) for dataset in DATASETS for rho in RHOS]


def exp3_depth():
    """Utility per tree level of the reference run cut to 3 tree levels (national + 2), so every
    dataset spends the whole budget over the same depth. Spain already has 3 levels: its
    reference runs serve as they are."""
    return [(dataset, marginal(dataset, depth=2)) for dataset in DATASETS
            if dataset != 'spanish_census']


def reference():
    """Experiment 5 runs nothing of its own: it reads the reference run's metrics (every pair
    with its U in both files)."""
    return [(dataset, marginal(dataset)) for dataset in DATASETS]


def exp4():
    """3-way TVD per level of the reference configuration, with the triples stratified.

    The reference runs drew 100 triples uniformly, which reached 0 to 5 contained ones per dataset
    (0 on Chile and Spain), so this reruns the configuration and evaluates every contained triple
    plus TRIPLES of the rest per number of pairs a bag measures. The `triples` key goes to
    benchmarks.metrics, not to the driver, and keeps these runs apart from the reference ones.
    """
    return [(dataset, marginal(dataset, triples=TRIPLES)) for dataset in DATASETS]


def exp6():
    """Utility, time and memory along the family of structures derived from the declared one.

    The family is generated by `benchmarks/structures.py`: merging bags walks toward the joint
    distribution, splitting them walks away from it, and both rules are structural, so no budget is
    spent on selection. The declared structure is included as the anchor of the curve; it is the
    reference run of step 2.2 and the orchestrator skips it as already recorded.

    The x axis of the analysis is the COVERAGE -- the fraction of attribute pairs inside some bag --
    not the number of bags, which is not comparable across datasets. Coverage is dimensionless, it
    is what the mechanism depends on, and the family spans it: on IPUMS 1940 it runs from 33% at
    `das_s2` to 100% at `das_m1`, which is the full joint in one bag and is byte for byte the
    declared `joint` structure. So the family interpolates between the marginal and the full-joint
    pipelines rather than comparing two methods.

    Coverage, width and Delta are all recoverable from `bags` in the run record, which holds the
    bags the junction tree actually returned rather than the ones requested. That matters, because
    a refinement can be re-fused by the triangulation: measured, every merge rung is respected on
    adult, SINASC, Spain and IPUMS, while the splits are exact only on adult and IPUMS.

    chilean_census is excluded, and the exclusion is a result rather than an omission: its 21 bags
    come from 8 declared marginals plus 30 constraint scopes, 5 of them 3 or 4 attributes wide, and
    every scope is a mandatory clique (topdown.py:285). Merging or splitting the marginals moves
    the width by 2% and the bag count by 1. At 27 attributes with real edit rules the structure is
    set by the questionnaire, not by the analyst.
    """
    runs = []
    for dataset in DATASETS:
        if dataset in EXP6_EXCLUDED:
            continue
        marginals = importlib.import_module(f'benchmarks.{dataset}.marginals')
        for structure in [marginals.DEFAULT] + marginals.DERIVED:
            runs.append((dataset, marginal(dataset, structure=structure)))
    return runs


# The rows of tab:structures in the paper: each dataset's best split, its declaration and its last
# merge (docs/figures/fig_structures.py prints the whole family).
EXP6_TABLE = {'ipums_1940': ('das_s2', 'das_s1', 'das', 'das_m1'),
              'sinasc': ('blocks_s1', 'blocks', 'blocks_m4'),
              'spanish_census': ('blocks_s3', 'blocks', 'blocks_m4'),
              'adult': ('blocks_s1', 'blocks', 'blocks_m3')}


def exp6_memory():
    """The rows of tab:structures again, only to measure the peak memory of the whole run.

    exp6 recorded memory as runner.peak_memory_mb(): the main process plus the LARGEST single
    worker (RUSAGE_CHILDREN is a max, not a sum), a lower bound with 20 workers alive at once, and
    not the instrument of the DAS table. The `memory` flag makes the orchestrator sample the run's
    whole process tree from outside it and skip the metrics, since utility and time are already
    recorded by exp6. It also gives these runs their own names, and `select` in the figures refuses
    them unless asked, so they never enter a utility median.
    """
    return [(dataset, marginal(dataset, structure=structure, memory=1))
            for dataset, structures in EXP6_TABLE.items() for structure in structures]


def exp7():
    """Global rounding MIP vs the junction-tree sweep."""
    return [(dataset, marginal(dataset, rounding=rounding)) for dataset in DATASETS
            for rounding in ('sweep', 'mip')]


def exp8():
    """Flat baselines: MST and AIM with the leaf geography as one column, at RHO. TopDown's side
    is the reference run (exp5), which publishes the same release, so nothing of ours reruns.
    `memory` samples the session PSS, the instrument of TopDown's memory columns, and is what lets
    --min-available-gb stop a run that fills the shared host instead of letting it swap."""
    return [(dataset, decided({'baseline': mechanism, 'rho': RHO, 'memory': 1}))
            for mechanism in BASELINES for dataset in DATASETS
            if (dataset, mechanism) not in EXP8_EXCLUDED]


EXPERIMENTS = {'exp1': exp1, 'exp2': exp2, 'exp3_composition': exp3_composition,
               'exp3_budget': exp3_budget, 'exp3_depth': exp3_depth, 'exp4': exp4,
               'exp5': reference, 'exp6': exp6, 'exp6_memory': exp6_memory, 'exp7': exp7,
               'exp8': exp8}
