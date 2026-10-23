"""AIM and MST (McKenna et al.) under the replace-one neighbour relation TopDown uses.

Both come from Ryan McKenna's mbi repository (Apache 2.0), at common.mbi_root(). Upstream calibrates to add/remove neighbours. Replacing a record
moves a marginal by one in two cells, so every Gaussian measurement has squared L2 sensitivity 2
instead of 1, and every L1-error score fed to the exponential mechanism has sensitivity 2 instead
of 1. Each function below is the upstream one with exactly those two changes (NOISE and SCORE),
so the upstream accounting holds verbatim at the stated rho: AIM's 0.9/0.1 split per round, MST's
rho/3 per step.

The upstream code is not otherwise touched. In particular AIM does not enforce structural zeros
("Haven't incorporated structural zeros back yet after refactoring"), so the edit rules are not
seen by either mechanism.
"""

import itertools
import os
import sys

from benchmarks import common

# At census sizes float32 makes the estimation stall (mbi warns past 100K records). The compile
# cache only adds overhead to the many small programs mbi compiles. Both must be set before jax.
os.environ.setdefault('JAX_ENABLE_X64', '1')
os.environ.setdefault('JAX_ENABLE_COMPILATION_CACHE', 'false')

sys.path.append(str(common.mbi_root() / 'mechanisms'))

import jax  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from mbi import Dataset, LinearMeasurement, estimation  # noqa: E402
from scipy.cluster.hierarchy import DisjointSet  # noqa: E402

import aim as upstream_aim  # noqa: E402
import mst as upstream_mst  # noqa: E402

NOISE = np.sqrt(2)  # Gaussian standard deviation multiplier: squared L2 sensitivity 2
SCORE = 2.0         # exponential mechanism sensitivity multiplier: L1 sensitivity 2


class AIM(upstream_aim.AIM):
    """Upstream AIM at a given rho, replace-one neighbours. Records the cliques it measured."""

    def __init__(self, rho, **kwargs):
        super().__init__(epsilon=0, delta=0, **kwargs)
        self.rho = rho
        self.measured = []

    def worst_approximated(self, candidates, answers, model, eps, sigma):
        # sigma arrives already multiplied by NOISE, so the bias term is the true one.
        errors = {}
        for cl, wgt in candidates.items():
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(answers[cl] - xest, 1) - bias)
        sensitivity = SCORE * max(abs(wgt) for wgt in candidates.values())
        return self.exponential_mechanism(errors, eps, sensitivity)

    def run(self, data, workload, num_synth_rows=None):
        """Upstream AIM.run; `sigma` keeps driving the accounting, `NOISE * sigma` is drawn."""
        rounds = self.rounds or 16 * len(data.domain)
        candidates = upstream_aim.compile_workload(workload)
        answers = {cl: data.project(cl).datavector() for cl in candidates}
        oneway = [cl for cl in candidates if len(cl) == 1]

        sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
        epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)
        print(f'AIM: {len(candidates)} candidates, {rounds} rounds, initial sigma '
              f'{NOISE * sigma:.3f}', flush=True)

        measurements = []
        rho_used = len(oneway) * 0.5 / sigma**2
        for cl in oneway:
            y = answers[cl] + self.gaussian_noise(NOISE * sigma, answers[cl].size)
            measurements.append(LinearMeasurement(y, cl, stddev=NOISE * sigma))
        self.measured = list(oneway)
        model = estimation.MirrorDescent().estimate(data.domain, measurements,
                                                    iters=self.max_iters)

        terminate = False
        while not terminate:
            # Every projection is compiled for the model's cliques, which change each round, so the
            # executables of past rounds are dead. Kept, they piled up at ~210 per round and ~1 GB per
            # round on the SINASC sample, and on the server Chile and Spain died compiling with RAM
            # free ("LLVM compilation error: Cannot allocate memory"). Cleared, memory stays flat.
            jax.clear_caches()
            if self.rho - rho_used < 2 * (0.5 / sigma**2 + 1.0 / 8 * epsilon**2):
                # Use up whatever budget is left in one last round.
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True
            rho_used += 1.0 / 8 * epsilon**2 + 0.5 / sigma**2
            size_limit = self.max_model_size * rho_used / self.rho

            small = upstream_aim.filter_candidates(candidates, model, size_limit)
            cl = self.worst_approximated(small, answers, model, epsilon, NOISE * sigma)
            n = data.domain.size(cl)
            y = answers[cl] + self.gaussian_noise(NOISE * sigma, n)
            measurements.append(LinearMeasurement(y, cl, stddev=NOISE * sigma))
            self.measured.append(cl)
            z = model.project(cl).datavector()

            potentials = model.potentials.expand(list(set(m.clique for m in measurements)))
            model = estimation.MirrorDescent().estimate(
                data.domain, measurements, iters=self.max_iters, potentials=potentials)
            w = model.project(cl).datavector()
            print(f'  round {len(self.measured) - len(oneway)}: rho {rho_used:.4f}/{self.rho:g} '
                  f'measured {cl} ({n:,} cells), size limit {size_limit:.2f} MB', flush=True)
            if np.linalg.norm(w - z, 1) <= NOISE * sigma * np.sqrt(2 / np.pi) * n:
                sigma /= 2
                epsilon *= 2

        model = estimation.MirrorDescent().estimate(
            data.domain, measurements, iters=self.max_iters, potentials=potentials)
        return model, model.synthetic_data(rows=num_synth_rows)


def mst(data, rho, num_synth_rows=None):
    """Upstream MST at a given rho, replace-one neighbours. Returns (synthetic, 2-way cliques)."""
    sigma = NOISE * np.sqrt(3 / (2 * rho))
    log1 = upstream_mst.measure(data, [(col,) for col in data.domain], sigma)
    data, log1, undo_compress = upstream_mst.compress_domain(data, log1)
    cliques = _select(data, rho / 3.0, log1)
    print(f'MST: selected {cliques}', flush=True)
    log2 = upstream_mst.measure(data, cliques, sigma)
    model = estimation.MirrorDescent().estimate(data.domain, log1 + log2, iters=10000)
    synthetic = model.synthetic_data(rows=num_synth_rows)
    # mbi emits uint8 columns where they fit and upstream reverse_data writes the uncompressed
    # codes back in place, which pandas >= 3 refuses past 255.
    synthetic = Dataset({col: np.asarray(values, dtype=np.int64)
                         for col, values in synthetic.to_dict().items()}, synthetic.domain)
    return undo_compress(synthetic), cliques


def _select(data, rho, measurement_log):
    """Upstream mst.select with the exponential mechanism at sensitivity SCORE."""
    model = estimation.MirrorDescent().estimate(data.domain, measurement_log, iters=2500)
    candidates = list(itertools.combinations(data.domain.attrs, 2))
    weights = {(a, b): np.linalg.norm(data.project([a, b]).datavector()
                                      - model.project([a, b]).datavector(), 1)
               for a, b in candidates}

    tree = nx.Graph()
    tree.add_nodes_from(data.domain.attrs)
    components = DisjointSet(data.domain.attrs)
    epsilon = np.sqrt(8 * rho / (len(data.domain.attrs) - 1))
    for _ in range(len(data.domain.attrs) - 1):
        candidates = [e for e in candidates if not components.connected(*e)]
        scores = np.array([weights[e] for e in candidates])
        edge = candidates[upstream_mst.exponential_mechanism(scores, epsilon, sensitivity=SCORE)]
        tree.add_edge(*edge)
        components.merge(*edge)
    return list(tree.edges)
