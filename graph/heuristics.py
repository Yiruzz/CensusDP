"""Marginal-selection heuristics.

A heuristic decides which marginals (cliques of attributes) to measure. It
returns a list of cliques that are then embedded in the interaction graph and
triangulated into bags by :class:`~graph.junction_tree.JunctionTree`.

Three strategies live here:

  MaxSpanningTreeMI               2-way maximum spanning tree over mutual information,
  UnconstrainedMaxSpanningTreeMI  the same, blind to the constraints.
  CostAwareGreedySelection        greedy over "records recovered minus records of noise
                                  added", which is the one that can propose k-way cliques
                                  and the only one that knows what a clique costs.

The first two are kept as the measured baselines. Their shared blind spot is documented on
CostAwareGreedySelection and in analysis/experiments/seleccion/README.md: argmax over MI
buys P11COMUNA x P12COMUNA (MI 0.3173, 132,496 cells) over P11 x P12 (MI 0.1965, 110
cells), paying 1,205x the cells for 1.6x the MI, and it is right by its own criterion.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .association import PairStatistic, independence_residual_l1, mutual_information
from .cost_model import CostModel
from .junction_tree import JunctionTree


class _UnionFind:
    """Disjoint-set over integer ids (path-compressed)."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            # Path compression for more efficient future finds.
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Merge the sets of a and b; return True iff they were different."""
        ra, rb = self.find(a), self.find(b) # Roots
        if ra == rb: # Same root case, already connected components
            return False
        self._parent[ra] = rb # Merge the two components (root of a points to root of b)
        return True


class MarginalSelectionStrategy(ABC):
    """Selects the cliques (marginals) to embed in the interaction graph.

    Attributes:
        pair_statistic: Which per-pair statistic this strategy wants the association matrix
            to hold. The caller (TopDown._select_marginal_cliques) reads it and builds the
            matrix accordingly, so a strategy scored in records is not silently handed one
            in nats. Defaults to mutual information, which is what the two spanning-tree
            heuristics were written against.
    """

    pair_statistic: PairStatistic = staticmethod(mutual_information)

    @abstractmethod
    def select(self, columns: Sequence[str], association: np.ndarray,
               mandatory_cliques: Iterable[Iterable[str]],
               cost: Optional[CostModel] = None) -> List[FrozenSet[str]]:
        """Return the cliques to embed.

        Args:
            columns: All attribute columns.
            association: Symmetric pairwise association matrix aligned to
                columns, holding whatever `pair_statistic` asked for.
            mandatory_cliques: Constraint scopes that must each be contained in
                some bag.
            cost: What a candidate structure costs in cells and in noise. Optional because
                the two spanning-tree heuristics predate it and ignore it; a strategy that
                needs it should raise rather than quietly fall back to being cost-blind.

        Returns:
            List of cliques (column subsets). Always includes the mandatory ones.
        """
        raise NotImplementedError


class MaxSpanningTreeMI(MarginalSelectionStrategy):
    """Heuristic 1: 2-way marginals from a max spanning tree over MI that
    contains the mandatory constraint cliques.

    The constraint cliques are forced into the structure first (their columns are
    pre-merged into connected components). A maximum spanning tree is then grown
    with Kruskal over the mutual-information edges, adding a 2-way marginal only
    when it connects two columns not already joined — by the constraints or by a
    previously chosen edge. This means an MI edge is never added inside a
    component the constraints already connect, so no redundant marginals are
    selected. The result is a maximum-weight spanning tree constrained to contain
    the constraint cliques.
    """

    def select(self, columns: Sequence[str], association: np.ndarray,
               mandatory_cliques: Iterable[Iterable[str]],
               cost: Optional[CostModel] = None) -> List[FrozenSet[str]]:
        columns = list(columns)
        index = {c: i for i, c in enumerate(columns)}

        mandatory = [frozenset(mc) for mc in mandatory_cliques if mc]
        cliques: List[FrozenSet[str]] = list(mandatory)

        components = _UnionFind(len(columns))
        # Force the constraint cliques: all columns of a clique share a component,
        # so MI edges will never be added within an already-constrained group.
        for clique in mandatory:
            members = [index[c] for c in clique if c in index]
            for other in members[1:]:
                components.union(members[0], other)

        # Kruskal over MI edges, heaviest first; keep only inter-component edges.
        edges = [
            (float(association[i, j]), i, j)
            for i in range(len(columns))
            for j in range(i + 1, len(columns))
        ]
        edges.sort(reverse=True)
        for _weight, i, j in edges:
            if components.union(i, j):
                cliques.append(frozenset((columns[i], columns[j])))

        return cliques


class UnconstrainedMaxSpanningTreeMI(MarginalSelectionStrategy):
    """Baseline heuristic: MST over MI computed independently of the constraints.

    Builds the full maximum spanning tree over all columns by mutual information
    (n-1 edges), then unions the mandatory constraint cliques on top. Unlike
    MaxSpanningTreeMI, the spanning tree does not know about the constraints, 
    so it may pick 2-way edges that duplicate or cross constraint cliques.

    Triangulation later absorbs any edge that ends up contained in a clique, but
    an edge that adds a new chord across a constraint cycle can still enlarge a
    bag (raise the treewidth), so this baseline is expected to be no better — and
    sometimes worse — than the constraint-aware version. Kept for empirical
    comparison of the two selection strategies.
    """

    def select(self, columns: Sequence[str], association: np.ndarray,
               mandatory_cliques: Iterable[Iterable[str]],
               cost: Optional[CostModel] = None) -> List[FrozenSet[str]]:
        columns = list(columns)
        cliques: List[FrozenSet[str]] = [frozenset(mc) for mc in mandatory_cliques if mc]

        # Plain Kruskal maximum spanning tree; the constraints do not steer it.
        components = _UnionFind(len(columns))
        edges = [
            (float(association[i, j]), i, j)
            for i in range(len(columns))
            for j in range(i + 1, len(columns))
        ]
        edges.sort(reverse=True)
        for _weight, i, j in edges:
            if components.union(i, j):
                cliques.append(frozenset((columns[i], columns[j])))

        return cliques


def _pairs_in(bag: Iterable[str], index: Dict[str, int]) -> Set[Tuple[int, int]]:
    """Column-index pairs a bag puts together, as ordered (i, j) with i < j."""
    ids = sorted(index[c] for c in bag if c in index)
    return {(ids[a], ids[b]) for a in range(len(ids)) for b in range(a + 1, len(ids))}


def _covered_pairs(bags: Iterable[Iterable[str]],
                   index: Dict[str, int]) -> Set[Tuple[int, int]]:
    """Every pair that lands inside some bag, i.e. every pair the model measures exactly."""
    covered: Set[Tuple[int, int]] = set()
    for bag in bags:
        covered |= _pairs_in(bag, index)
    return covered


def _maximal(cliques: Sequence[FrozenSet[str]]) -> List[FrozenSet[str]]:
    """Drop the cliques nested inside another one; the interaction graph is unchanged.

    Embedding a clique adds every pairwise edge among its columns, so a subset of a clique
    already present contributes nothing. Same idea as mbi's clique_utils.maximal_subset.
    """
    ordered = sorted(cliques, key=len, reverse=True)
    kept: List[FrozenSet[str]] = []
    for clique in ordered:
        if not any(clique <= other for other in kept):
            kept.append(clique)
    return kept


class CostAwareGreedySelection(MarginalSelectionStrategy):
    """Greedy selection of k-way cliques scored as records recovered minus records of noise.

    The two spanning-tree heuristics maximise mutual information with no notion of what the
    marginal costs, and the measured consequence is that they buy geography x geography
    bridges: MI is right about those pairs carrying the most signal, and wrong about them
    being worth 1,205x the cells. This scores the same decision in a single currency.

    For a candidate clique C, with the junction tree rebuilt (and re-triangulated) as if C
    had been declared:

        score(C) =  sum over pairs (a, b) the NEW bags co-locate for the first time
                        of   max(0, residual[a, b] - selection_noise * cells(a, b))
                 -  lam * ( noise_burden(new bags) - noise_burden(current bags) )

    Both terms are in records. ``lam = 0`` recovers a cost-blind heuristic; raising lam buys
    cheap coverage before expensive coverage.

    **lam = 1 does NOT balance the two halves, and the reason is worth knowing before
    calibrating it.** The gain is an upper bound: it credits a pair that is not inside any
    bag as if the model said NOTHING about it, when in fact the junction tree still relates
    the two columns through the bags on the path between them. On the 27-column census the
    over-statement is roughly fiftyfold. Measured, over all 351 pairs (raw data, so this is
    the heuristic's ceiling, not a private number):

        arm             bags   W          residual covered   residual uncovered   TVD 2-way
        restricciones     17     372,795    304,866,216         674,874,455        0.129152
        mst_oraculo       22     646,782    346,520,249         633,220,422        (no soln)
        manual            21     381,585    381,908,705         597,831,966        0.086519

    The ORDER is right - more covered residual, lower TVD, and the hand-declared marginals
    come out on top of both heuristics - so the criterion ranks structures correctly. The
    LEVEL is not: uncovered residual differs by 1.13x between the two arms whose TVD differs
    by 1.49x, so a record of "uncovered residual" is worth far less than a record of noise.
    In consequence the score stays positive long past the point where the model stops being
    solvable, and ``max_width`` - not lam - is what ends the search, exactly as AIM's
    ``size_limit`` filter does the ending there. Sweep both:
    analysis/experiments/seleccion/coste.py.

    Where each piece comes from, and why:

    - **The gain is an L1 residual, not mutual information.** ``residual[a, b]`` is what the
      independent model already misplaces on that pair
      (association.independence_residual_l1), so it is exactly what measuring the pair buys
      back, in the units the utility metric is published in. This is McKenna's criterion:
      MST scores edges by ``||x - xhat||_1`` (mechanisms/mst.py), AIM by the same thing
      weighted by workload overlap.
    - **The de-biasing term is AIM's.** ``selection_noise * cells`` is the L1 a table of that
      many cells carries purely from the noise the selection measurement added; a candidate
      has to beat its own noise floor to score. AIM subtracts ``sqrt(2/pi)*sigma*n_cells``
      at mechanisms/aim.py:102, and jam.py repeats it verbatim. It is what stops a huge
      table from looking informative just for being huge.
    - **The cost is differenced, not local.** In AIM one more clique only costs its own
      noise. Here the sensitivity IS the number of bags, so a candidate that adds a bag
      raises sigma on every cell of the model. Differencing the whole burden charges both
      halves at once.
    - **The gain is read off the bags, not off the declared clique.** Triangulation adds
      fill, the fill is already paid for in the new width, so it also earns its utility.
      It is the only reading consistent with "declaring cliques does not guarantee you get
      those bags".

    Candidates each round are the uncovered pairs plus every current bag extended by one
    column. That is where arity above 2 comes from: a 3-way clique appears when widening an
    existing bag recovers more than widening it costs, which is how the seven hand-declared
    census marginals (five of them 3-way) were chosen by a human.

    Two approximations, both deliberate and both stated so they are not mistaken for
    exactness:

    - **The reference model is independence, not belief propagation.** A pair inside a bag
      is measured exactly, so its gain is 0; a pair outside every bag is scored as if the
      model said nothing about it, which OVERSTATES the gain when the two columns are
      linked through a path of bags - by about fiftyfold on the census, see above. Fitting
      the real marginal model would mean building McKenna's MRF, which this pipeline does
      not have. MST operates in the same regime - it scores every edge once against the
      1-way-only fit. The cheap way to do better, if this is ever revisited: for a
      one-column separator s, the tree's own prediction is p(i,j) = sum_s p(i,s)p(j,s)/p(s),
      and every one of those 2-way tables is already measured.
    - **The bag-widening candidates are a greedy neighbourhood, not an enumeration.** All
      C(d, 3) triples would be 2,925 candidates a round against roughly d^2/2 + bags*d;
      a 3-way clique is reachable only by widening a bag that already exists.

    Selection is greedy argmax, not an exponential mechanism as in MST and AIM. Those two
    score against the TRUE data every round and so must pay for each comparison; here the
    residuals come from one already-paid, already-noised measurement of the 2-way marginals,
    and picking the maximum of a released quantity is post-processing. The cells and the
    noise scales are declarations. So the whole search is free.

    Attributes:
        lam: Weight on the cost term. Not a balance point (see above) - it decides which
            coverage gets bought inside the width cap, and how much it matters depends on
            how loose that cap is. Measured on the census: at a 400,000 cap nothing changes
            between lam 0 and lam 16 (only cheap cliques fit anyway, so the cap already
            imposes the cost-awareness); at a 3,000,000 cap lam 16 covers 47% more residual
            than lam 0 at the same W, by buying the cheap pairs first and fitting 19 cliques
            instead of 10. Coverage is NOT monotone in lam - lam changes the first purchase
            and with it the whole neighbourhood of the later ones.
        max_cliques: Stop after this many selected cliques (the mandatory ones do not
            count). None means run until no candidate scores positive - in practice, until
            the CostModel's max_width bites.
        expand_bags: Whether to offer k-way candidates. False restricts the search to
            2-way, which is the ablation that isolates how much of the gap to the manual
            marginals is arity rather than cost.
        trace: One record per round, filled by select(). Diagnostics only.
    """

    pair_statistic: PairStatistic = staticmethod(independence_residual_l1)

    def __init__(self, lam: float = 1.0, max_cliques: Optional[int] = None,
                 expand_bags: bool = True) -> None:
        if lam < 0:
            raise ValueError(f"lam must be non-negative, got {lam}.")
        self.lam = float(lam)
        self.max_cliques = max_cliques
        self.expand_bags = expand_bags
        self.trace: List[Dict] = []

    def select(self, columns: Sequence[str], association: np.ndarray,
               mandatory_cliques: Iterable[Iterable[str]],
               cost: Optional[CostModel] = None) -> List[FrozenSet[str]]:
        if cost is None:
            raise ValueError(
                "CostAwareGreedySelection needs a CostModel: its whole point is trading "
                "utility against cells and noise, and without one it would silently "
                "degrade into a cost-blind heuristic. Build one with "
                "selection_cost.build_cost_model(...) and pass it as cost=."
            )

        columns = list(columns)
        index = {c: i for i, c in enumerate(columns)}
        missing = [c for c in columns if c not in cost.cardinalities]
        if missing:
            raise ValueError(f"CostModel has no cardinality for columns: {missing}")

        residual = self._debiased_residual(columns, association, cost)

        mandatory = [frozenset(mc) for mc in mandatory_cliques if mc]
        cliques: List[FrozenSet[str]] = list(mandatory)
        tree = JunctionTree.build(columns, cliques, cost.cardinalities)
        covered = _covered_pairs(tree.bags, index)
        burden = cost.noise_burden(tree.bags)

        self.trace = []
        chosen: List[FrozenSet[str]] = []
        while self.max_cliques is None or len(chosen) < self.max_cliques:
            best = self._best_candidate(columns, index, residual, cliques, tree,
                                        covered, burden, cost)
            if best is None:
                break

            cliques.append(best['clique'])
            chosen.append(best['clique'])
            tree = best['tree']
            covered = best['covered']
            burden = best['burden']
            self.trace.append({
                'clique': sorted(best['clique']),
                'gain': best['gain'],
                'cost': best['cost'],
                'score': best['score'],
                'n_bags': tree.n_bags,
                'width': best['width'],
            })

        # Growing a bag one column at a time leaves the intermediate steps behind: picking
        # {a,b} and later {a,b,c} both earned their round, but only the second says anything
        # the graph does not already have. Dropping the nested ones changes no edge and no
        # bag - it just makes the selection readable, and keeps a caller from thinking the
        # heuristic asked for a marginal it did not.
        return mandatory + _maximal(chosen)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _debiased_residual(self, columns: Sequence[str], association: np.ndarray,
                           cost: CostModel) -> np.ndarray:
        """The association matrix with the selection measurement's own noise removed.

        A pair table of n cells carries about ``selection_noise * n`` records of L1 that are
        noise rather than signal, and n is public. Subtracting it is AIM's bias term
        (mechanisms/aim.py:102); the clamp at zero is what turns "indistinguishable from
        noise" into "worth nothing" rather than into a negative weight. With
        selection_noise = 0 (an oracle arm fed raw data) this is the identity.

        Measured on simulated tables at the census's scale (N = 17.6M) and the two selection
        sigmas the campaign used, the term behaves as intended and is not merely cosmetic:

            table                         true residual   inflation by noise   subtracted
            110 cells, near-independent          22,331           -70 .. -121          822
            132,496 cells, near-independent     951,275     449,154 .. 1,178,898  990,583 .. 1,981,166
            132,496 cells, concentrated       28,040,990          -257 .. -708    990,583 .. 1,981,166

        Two things to read there. On a STRONG pair the noise cancels almost exactly - the
        residual is a difference in which both sides carry it - so the subtraction is a flat
        7% haircut and changes no ranking. On a WEAK pair the inflation is real and grows
        with the table, which is precisely the failure mode a cost-blind heuristic has: a
        huge table looking informative for being huge. The term over-subtracts by about
        1.7x there, because clipping at zero already absorbs the negative half of the noise.
        That erring is conservative in the direction this heuristic exists to correct.
        """
        residual = np.array(association, dtype=np.float64, copy=True)
        if cost.selection_noise:
            sizes = np.array([cost.cardinalities[c] for c in columns], dtype=np.float64)
            residual -= cost.selection_noise * np.outer(sizes, sizes)
        np.clip(residual, 0.0, None, out=residual)
        np.fill_diagonal(residual, 0.0)

        if not residual.any():
            # Selecting nothing is a legitimate answer, but reaching it this way is almost
            # never the intended one: the budget for the pairwise measurement was already
            # spent and the run is about to proceed as if no marginals had been asked for.
            warnings.warn(
                "Marginal selection found no pair worth measuring: every association fell "
                "below the noise the selection measurement itself carries "
                f"({cost.selection_noise:.3g} records per cell). The run will use only the "
                "constraint scopes, having paid for the selection. Spend a larger "
                "budget_fraction, or drop the selection and declare the marginals.",
                stacklevel=3,
            )
        return residual

    def _candidates(self, columns: Sequence[str], residual: np.ndarray,
                    tree: JunctionTree,
                    covered: Set[Tuple[int, int]]) -> List[FrozenSet[str]]:
        """Uncovered pairs worth anything, plus every bag widened by one column."""
        candidates: List[FrozenSet[str]] = []
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                # A covered pair is already measured exactly, so it can only add cost. A
                # zero residual means the independent model already gets it right (or the
                # noise swallowed the difference), so there is nothing to buy back.
                if (i, j) not in covered and residual[i, j] > 0.0:
                    candidates.append(frozenset((columns[i], columns[j])))

        if self.expand_bags:
            seen = set(candidates)
            for bag in tree.bags:
                bag_set = frozenset(bag)
                for column in columns:
                    if column in bag_set:
                        continue
                    widened = bag_set | {column}
                    if widened not in seen:
                        seen.add(widened)
                        candidates.append(widened)
        return candidates

    def _best_candidate(self, columns: Sequence[str], index: Dict[str, int],
                        residual: np.ndarray, cliques: List[FrozenSet[str]],
                        tree: JunctionTree, covered: Set[Tuple[int, int]],
                        burden: float, cost: CostModel) -> Optional[Dict]:
        """Evaluate every candidate exactly and return the best positive-scoring one."""
        best: Optional[Dict] = None

        for candidate in self._candidates(columns, residual, tree, covered):
            # Cheap rejection before rebuilding anything: a clique whose own columns already
            # exceed the width cap is not going to fit once triangulation puts them in a bag
            # with whatever else they fuse into. It is a pre-filter, not the decision - the
            # exact width check two lines down is - and it earns its keep by skipping the
            # triangulation and the full width sum for the candidates that are hopeless,
            # which is most of them once high-cardinality columns are in play. Uses the
            # restricted count, so a clique the edit rules cut down is judged on its real
            # size and not on a product of cardinalities.
            if cost.max_width is not None:
                own_cells = cost.bag_cells(self._canonical(candidate, index))
                if own_cells > cost.max_width:
                    continue

            new_tree = JunctionTree.build(columns, cliques + [candidate],
                                          cost.cardinalities)
            width = cost.width(new_tree.bags)
            if cost.max_width is not None and width > cost.max_width:
                continue

            new_covered = _covered_pairs(new_tree.bags, index)
            gained = new_covered - covered
            if not gained:
                # Triangulation absorbed it: no pair is measured that was not measured
                # before, so there is nothing to gain and only cells to pay for.
                continue

            gain = float(sum(residual[i, j] for i, j in gained))
            new_burden = cost.tree_noise(new_tree.n_bags) * width
            candidate_cost = new_burden - burden
            score = gain - self.lam * candidate_cost

            if score > 0.0 and (best is None or score > best['score']):
                best = {'clique': candidate, 'tree': new_tree, 'covered': new_covered,
                        'burden': new_burden, 'width': width, 'gain': gain,
                        'cost': candidate_cost, 'score': score}

        return best

    @staticmethod
    def _canonical(clique: Iterable[str], index: Dict[str, int]) -> Tuple[str, ...]:
        """Order a column set the way JunctionTree canonicalises bags, for cache hits."""
        return tuple(sorted(clique, key=lambda c: index[c]))
