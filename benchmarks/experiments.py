"""The runs each experiment of the paper needs, as (dataset, configuration) pairs.

A configuration spells out every runner flag that tells two runs apart, so a run that several
experiments share (the reference run of a dataset serves experiments 4 to 7) has a single name
and runs once. benchmarks/orchestrate.py runs them; benchmarks/README.md is the plan.

Part 1 (experiments 1 and 2) compares against the DAS with the DAS budget, which the runner
cannot express yet. Part 2 (experiments 3 to 7) starts with experiment 3, in two steps:
  1. exp3_composition: every composition at COMPOSITION_RHO. The winner goes in COMPOSITION.
  2. exp3_budget: every rho with COMPOSITION. The chosen one goes in RHO.
Until both are set, an experiment that needs them refuses to list its runs.
Experiment 4 is pending: for now it reads the 3-way TVD per level of the reference run.
"""

import importlib

from benchmarks.common import COMPOSITIONS

DATASETS = ('adult', 'sinasc', 'spanish_census', 'chilean_census', 'ipums_1940')
RHOS = (0.1, 0.5, 1, 2, 5)
COMPOSITION_RHO = 1
COMPOSITION = None  # from exp3_composition
RHO = None          # from exp3_budget


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
    """Time, memory and width vs number of columns on IPUMS 1940, full-joint and marginal."""
    das_budget()


def exp3_composition():
    """Utility per tree level of every composition, at COMPOSITION_RHO."""
    return [(dataset, marginal(dataset, rho=COMPOSITION_RHO, composition=composition))
            for dataset in DATASETS for composition in COMPOSITIONS]


def exp3_budget():
    """Utility per tree level of every budget, with COMPOSITION."""
    return [(dataset, marginal(dataset, rho=rho)) for dataset in DATASETS for rho in RHOS]


def reference():
    """Experiments 4 and 5 run nothing of their own: they read the reference run's metrics
    (3-way TVD per level, and every pair with its U in both files)."""
    return [(dataset, marginal(dataset)) for dataset in DATASETS]


def exp6():
    """Utility, time and memory vs the declared marginal structure."""
    return [(dataset, marginal(dataset, structure=structure)) for dataset in DATASETS
            for structure in importlib.import_module(f'benchmarks.{dataset}.marginals').STRUCTURES]


def exp7():
    """Global rounding MIP vs the junction-tree sweep."""
    return [(dataset, marginal(dataset, rounding=rounding)) for dataset in DATASETS
            for rounding in ('sweep', 'mip')]


EXPERIMENTS = {'exp1': exp1, 'exp2': exp2, 'exp3_composition': exp3_composition,
               'exp3_budget': exp3_budget, 'exp4': reference, 'exp5': reference, 'exp6': exp6,
               'exp7': exp7}
