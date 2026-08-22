"""Rounding by sweeping the junction tree, as an alternative to the global binary program.

Walk the bags in the junction tree's BFS order. When a bag is reached,
its parent bag is already rounded, so the mass it must show on their shared separator is
fixed. Rounding that bag is then a collection of independent 2-way transportation problems,
one per separator value s:

    rows    = the K geographic children, with fixed total T[k][s] = what the parent bag already
              spent on s for child k                                   (separator consistency)
    columns = the bag cells c that project to s, with fixed total parent[c]
                                                                       (geographic consistency)

Always feasible, because the margins close:

    sum_k T[k][s] = the parent estimate summed over group s in the parent bag
                  = the parent estimate summed over group s in this bag

- the first equality by the geographic rows already imposed, the second because the parent
node's own estimate satisfies its separator rows. A transportation matrix is TUM, so the
continuous LP returns an integer vertex with no branch and bound.

We lose Global optimality (this is greedy over the tree) and adjacency past the first
bag: a cell can end further than 1 from x_tilde.
On the 27-column instance: 13.0s against 271.5s for the MIP (20.9x) and +1.22% of L1 against
the truth.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

from constraints.sparse_constraint import SparseConstraint

# The LP is TUM, so its optimum is an integer vertex. Anything past this is a structural bug,
# not a tolerance issue.
INTEGRALITY_TOLERANCE = 1e-6


def _largest_remainder(target: np.ndarray, col_totals: np.ndarray) -> np.ndarray:
    """Split each column's integer total across the K children, 1-D and exactly.

    Optimal in L1 and adjacent, with no solver, whenever the columns are independent - which
    holds exactly when sum_k target[k, c] == col_totals[c], as the QP imposes the geographic
    rows as hard equalities. The sweep uses it on the degenerate m == 1 case: rounding the K
    fractional child populations against the parent's integer total.

    Args:
        target (np.ndarray): (K, m) fractional target.
        col_totals (np.ndarray): (m,) integer total per column.

    Returns:
        np.ndarray: (K, m) integers summing to col_totals along axis 0.
    """
    floor = np.floor(target).astype(np.int64)
    remainder = target - floor
    missing = np.asarray(col_totals, dtype=np.int64) - floor.sum(axis=0)

    # We want to know which remainders are the largest, so we round them first when needed.
    order = np.argsort(-remainder, axis=0, kind="stable")
    # wins is a boolean array of the same shape as floor, where wins[k, c] is True if child k's
    # remainder for column c is among the largest missing[c] remainders.
    wins = np.zeros_like(floor, dtype=bool)
    np.put_along_axis(wins, order, np.arange(len(floor))[:, None] < missing, axis=0)
    return floor + wins


def _group_columns(groups: np.ndarray, n_groups: int) -> Tuple[np.ndarray, np.ndarray]:
    """Lay the columns out contiguously by separator group.

    Every separator value is an independent transportation problem, so the sweep needs each
    group's columns together.

    Args:
        groups (np.ndarray): (m,) group id per column, renumbered to [0, n_groups).
        n_groups (int): Number of distinct groups.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (order, bounds), so group g owns the columns
        `order[bounds[g]:bounds[g + 1]]`.
    """
    order = np.argsort(groups, kind="stable")
    bounds = np.searchsorted(groups[order], np.arange(n_groups + 1))
    return order, bounds


def _verify_rows(constraints: Sequence[SparseConstraint],
                 x_joint: np.ndarray) -> Tuple[float, int]:
    """Evaluate every row against the solution, exactly.

    Checking the rows the sweep already enforces is not redundant. "Enforced by construction"
    runs through a long chain of index mappings, position_of, bag_cells, columns, order/bounds,
    the separator renumbering and a bug anywhere in it yields an assignment that looks feasible
    and is wrong.

    Returns:
        Tuple[float, int]: (worst absolute violation, number of rows violated).
    """
    worst, violated = 0.0, 0
    for row in constraints:
        lhs = float(np.dot(row.coefs, x_joint[row.indices]))
        residual = abs(lhs - row.rhs)
        if row.sense == "=":
            if residual > 0.0:
                violated += 1
                worst = max(worst, residual)
        elif row.sense == "<=" and lhs > row.rhs:
            violated += 1
            worst = max(worst, residual)
        elif row.sense == ">=" and lhs < row.rhs:
            violated += 1
            worst = max(worst, residual)
    return worst, violated


class SweepRoundingModel:
    """Rounds by sweeping the junction tree, delegates the QP to another backend.

    Args:
        backend: An optimizer backend instance (OptimizationModelLP or OptimizationModel). Its
            QP is used unchanged and its rounding MIP is kept as the fallback for the
            hierarchy's root node, which has no parent and therefore no column margins.
        data_handler (DataHandler): Supplies the junction tree and the bag layout. Must already
            have `build_marginal_domains` applied.
        solver_options (dict): Merged solver options. They configure this model's own Gurobi
            environment. See __init__ for why the backend's cannot be reused.

    Raises:
        ValueError: If the data handler carries no junction tree (full-joint pipeline).
    """

    def __init__(self, backend: Any, data_handler: Any, solver_options: dict = {}) -> None:
        if data_handler is None or getattr(data_handler, "junction_tree", None) is None:
            raise ValueError("Sweep rounding needs a junction tree.")
        self._backend = backend
        self._handler = data_handler
        self.solver_options = solver_options or {}

        # Own gurobipy environment rather than the backend's. Both backends expose `.env`, but
        # they hold different classes under that name: write_lp a gurobipy.Env, pyoptinterface a
        # pyoptinterface._src.gurobi.Env and _solve_transport builds its models with gurobipy.
        # Borrowing raised `AttributeError: 'Env' object has no attribute '_cenv'` on the first
        # transport whenever the backend was pyoptinterface.

        # TODO: See if writing the LP is better here.
        self.env = gp.Env(empty=True)
        for key, value in self.solver_options.items():
            self.env.setParam(key, value)
        self.env.start()

        self._solution_type = backend._solution_type

        self._junction_tree = data_handler.junction_tree
        self._separator_cells: Dict[Tuple[str, ...], int] = {}

    # ------------------------------------------------------------------ QP: straight delegation

    def non_negative_real_estimation(self, *args, **kwargs) -> np.ndarray:
        """Delegate to the wrapped backend. The sweep only replaces the rounding step."""
        return self._backend.non_negative_real_estimation(*args, **kwargs)

    # ------------------------------------------------------------------ rounding

    def rounding_estimation(self, x_tilde: np.ndarray, node_id: int,
                            constraints: List[SparseConstraint],
                            active: Optional[List[int]] = None,
                            n: Optional[int] = None) -> sp.csc_matrix:
        """Round by sweeping the junction tree, then verify every row.

        Args:
            x_tilde (np.ndarray): Non-negative real cell counts from the QP step, aligned to
                active.
            node_id (int): The node whose children are being solved, for error messages.
            constraints (List[SparseConstraint]): Sparse constraint objects, already offset into
                joint space and restricted to active indices by the caller.
            active (Optional[List[int]]): Global joint-space indices of non-pruned cells. None
                means the hierarchy's root, which is delegated to the backend's MIP.
            n (Optional[int]): Joint vector length, required when active is provided.

        Returns:
            scipy.sparse.csc_matrix: Column vector of shape (n, 1) with integer cell counts.

        Raises:
            ValueError: On the same active/n mismatches the MIP rounders reject.
            RuntimeError: If the parent's values cannot be recovered from x_tilde, a
                transportation problem is infeasible or comes back fractional, or the final
                solution violates any row.
        """
        x_tilde = np.asarray(x_tilde, dtype=float)

        # The hierarchy's root node has no parent, so there are no column margins and the sweep
        # is not defined. Cheap to leave to the MIP.
        if active is None:
            return self._backend.rounding_estimation(x_tilde, node_id, constraints, active, n)

        # The QP bounds x at 0, but a returned value can still be -1e-13, and floor() would send
        # it to -1: segment A would get ub = -1 against lb = 0 and the transport would come back
        # infeasible for a reason unreadable from the solver.
        x_tilde = np.maximum(x_tilde, 0.0)

        active = np.asarray(list(active), dtype=np.int64)
        if n is None:
            raise ValueError("rounding_estimation requires `n` when `active` is provided.")
        if len(x_tilde) != len(active):
            raise ValueError(
                f"x_tilde length {len(x_tilde)} does not match active length {len(active)}."
            )
        if len(active) == 0: # Case of an empty node.
            return sp.csc_matrix((n, 1), dtype=np.int64)

        # Size of the concatenation of all the bags' marginal domains.
        width = int(self._handler.marginal_width)
        n_children = int(n // width) # n = n_children * width
        n_support = len(active) // n_children
        if n_children * n_support != len(active):
            raise ValueError(
                f"active has {len(active)} entries, not a multiple of the {n_children} children "
                f"implied by n={n} over a marginal width of {width}."
            )

        # The parent's estimate is the column margin of every transport, and it is recovered from
        # x_tilde rather than passed in: the QP carries the geographic rows as hard equalities, so
        # sum_k x_tilde[k][p] is the parent's integer count at p. `active` is children-major over
        # an ascending support, so its first block gives the positions.
        support = active[:n_support] % width
        sums = x_tilde.reshape(n_children, n_support).sum(axis=0)
        # The QP is continuous, so the totals are integers only up to floating point error.
        parent_values = np.rint(sums).astype(np.int64)
        drift = float(np.abs(sums - parent_values).max()) if n_support else 0.0
        if drift > 1e-3:
            raise RuntimeError(
                f"sweep rounding: the QP solution misses the geographic rows by {drift:g} at node "
                f"{node_id}, so the parent's integer counts cannot be recovered from it. Every "
                f"transport takes those counts as its column margins and should be integers."
            )

        # position_of[p] = position of cell p in the support, or -1 if p is not in the support.
        position_of = np.full(width, -1, dtype=np.int64)
        position_of[support] = np.arange(n_support)

        # Sweep over the junction tree, rounding each bag as a collection of independent
        # 2-way transports.
        solution = self._sweep(parent_values, x_tilde, n_children, position_of, node_id)

        x_joint = np.zeros(n, dtype=np.int64)
        x_joint[active] = solution.reshape(-1)

        worst, violated = _verify_rows(constraints, x_joint)
        if violated:
            raise RuntimeError(
                f"sweep rounding: {violated} of {len(constraints)} rows violated "
                f"(max residual {worst:g}) at node {node_id}. The sweep enforces geographic "
                f"consistency, separator consistency and the per-child populations; any other "
                f"within-bag user constraint is only checked. Use rounding='mip' for those."
            )

        rows = active[x_joint[active] > 0]
        values = x_joint[rows]
        return sp.csc_matrix(
            (values, (rows, np.zeros(len(rows), dtype=np.int64))),
            shape=(n, 1), dtype=self._solution_type)

    # ------------------------------------------------------------------ internals

    def _separator_size(self, separator: Tuple[str, ...]) -> int:
        """Number of cells of a separator's value space, memoised."""
        if separator not in self._separator_cells:
            self._separator_cells[separator] = \
                self._handler.contingency_domain.subdomain(separator).n_cells
        return self._separator_cells[separator]

    def _sweep(self, parent_values: np.ndarray, x_tilde: np.ndarray, n_children: int,
               position_of: np.ndarray, node_id: int) -> np.ndarray:
        """Walk the bags in BFS order, rounding each as 2-way transports.

        Returns:
            np.ndarray: (n_children, len(support)) integers.
        """
        tree = self._junction_tree
        n_support = len(parent_values)
        target_all = x_tilde.reshape(n_children, n_support)
        solution = np.zeros((n_children, n_support), dtype=np.int64)

        # The sweep is greedy over the tree, so the parent bag is already rounded
        # when a bag is reached.
        for bag in tree.order:
            offset = self._handler.bag_offsets[bag]
            n_cells = self._handler.bag_domains[bag].n_cells
            local = position_of[offset:offset + n_cells]
            # bag_cells = active cells in this bag
            bag_cells = np.flatnonzero(local >= 0)
            if len(bag_cells) == 0: # Case of an empty bag
                continue

            # Here we get the hierarchical consistency constraint in the active space
            columns = local[bag_cells] # positions inside the support
            col_totals = parent_values[columns]
            target = target_all[:, columns]

            parent_bag = tree.parent[bag]
            if parent_bag is None: # Case root bag, which has no separator margin constraint.
                self._solve_root_bag(target, col_totals, columns, solution, node_id)
                continue

            # groups = group id per column of `bag` renumbered to [0, G)
            # row_totals = (K, G) integer totals per child and separator value,
            # n_groups = number of distinct separator values in this bag's support.
            groups, row_totals, n_groups = self._separator_margins(
                bag, parent_bag, bag_cells, col_totals, solution, position_of, node_id)

            # One model per separator value. They share no variable and no row, so solving them
            # apart is exactly solving them together, and it keeps every model as small as the
            # problem really is.
            # TODO: Analyse how many groups can be merged and have a faster solve time.
            #       In a preliminary test, we processed every bag as a single transport
            #       and the solver time went up. There should be a sweet spot in between.
            order, bounds = _group_columns(groups, n_groups)
            for group in range(n_groups):
                group_columns = order[bounds[group]:bounds[group + 1]]
                if len(group_columns) == 0:
                    continue
                solution[:, columns[group_columns]] = self._solve_transport(
                    target[:, group_columns], col_totals[group_columns],
                    row_totals[:, group], node_id)

        return solution

    def _solve_root_bag(self, target: np.ndarray, col_totals: np.ndarray,
                        columns: np.ndarray, solution: np.ndarray, node_id: int) -> None:
        """Round the junction tree's root bag, the only one with no separator margin.

        Its row margins are the per-child populations, and they are read off the QP. Every
        record falls in exactly one cell of every bag, so sum_c x_tilde[k][c] over this bag is
        child k's population as the QP settled it. Rounding those K values against the parent's
        integer total makes the bag a 2-way transport again and imposing them here propagates
        for free. Each T[k][s] computed downstream then sums to exactly child k's population.

        Feasible by construction, with nothing to check. The QP carries the geographic rows as
        hard equalities, so the unrounded margins already satisfy sum_k population[k] =
        sum_c col_totals[c], and rounding them jointly preserves it.

        What it guarantees is adjacency, not the nearest integer: the K margins must also sum to
        the parent's total, so a child can land up to 1 away from its own fractional population.
        """
        totals = _largest_remainder(target.sum(axis=1)[:, None],
                                    np.array([col_totals.sum()]))[:, 0]
        solution[:, columns] = self._solve_transport(target, col_totals, totals, node_id)

    def _separator_margins(self, bag: int, parent_bag: int, bag_cells: np.ndarray,
                           col_totals: np.ndarray, solution: np.ndarray,
                           position_of: np.ndarray,
                           node_id: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Row margins for a non-root bag: what the parent bag already spent per separator value.

        The parent bag is rounded already - it comes first in the BFS order - so summing its
        solution by separator value gives a fixed integer margin this bag has to reproduce.

        Args:
            bag (int): The bag being rounded.
            parent_bag (int): Its parent in the junction tree, already rounded.
            bag_cells (np.ndarray): Active cells of `bag`, in its own local cell space.
            col_totals (np.ndarray): The parent node's count per active cell of `bag`, i.e. the
                column margins. Only used to check that both margins agree.
            solution (np.ndarray): (K, len(support)) rounded so far; the parent bag's block of
                it is what gets aggregated here.
            position_of (np.ndarray): Length-width map from a per-node position to its offset in
                the support, or -1 when pruned.
            node_id (int): For the error messages only.

        Returns:
            Tuple: (group id per column of `bag` renumbered to [0, G), (K, G) row totals, G).
        """
        handler = self._handler
        tree = self._junction_tree
        n_children = solution.shape[0]

        separator = tree.parent_separator[bag]

        # The separator's value space is deliberately not restricted (RestrictedDomain does not
        # override subdomain), so both bags of the edge number a value alike and the two
        # projections below can be crossed. n_separator is therefore an upper bound: the groups
        # that exist come out of `present` further down, and they depend on the parent's
        # support, not only on the declared restriction.
        n_separator = self._separator_size(separator)
        groups_full = handler.separator_projection(bag, separator)[bag_cells]

        parent_offset = handler.bag_offsets[parent_bag]
        parent_local = position_of[parent_offset:
                                   parent_offset + handler.bag_domains[parent_bag].n_cells]
        parent_cells = np.flatnonzero(parent_local >= 0)
        parent_groups = handler.separator_projection(parent_bag, separator)[parent_cells]
        parent_columns = parent_local[parent_cells]

        # The parent bag is rounded already, so aggregating its block of the solution by separator
        # value gives the integer total this bag has to reproduce.
        totals_full = np.zeros((n_children, n_separator), dtype=np.int64)
        for k in range(n_children):
            totals_full[k] = np.bincount(parent_groups, weights=solution[k, parent_columns],
                                         minlength=n_separator).astype(np.int64)

        # present is the number of values in the separator domain that is in the support.
        # We have that many independent transportation problems.
        present = np.unique(groups_full)
        renumber = np.full(n_separator, -1, dtype=np.int64)
        renumber[present] = np.arange(len(present))
        # The groups are renumbered to [0, G) so they can be used as indices into the row totals.
        groups = renumber[groups_full]
        # The row totals are the parent bag's integer counts for each separator value, which
        # is what this bag has to reproduce. They are the row margins of the transport.
        row_totals = totals_full[:, present]

        # The margins must close.
        # If they do not, the parent's estimate did not satisfy its own separator rows and every
        # transport below would be infeasible for a reason impossible to read off the solver.
        column_sums = np.bincount(groups, weights=col_totals, minlength=len(present))
        gap = float(np.abs(row_totals.sum(axis=0) - column_sums).max())
        if gap > 0:
            raise RuntimeError(
                f"sweep rounding: margins do not close at node {node_id}, bag {bag} "
                f"(max |sum_k T - sum of parent values| = {gap:g}). The parent estimate does "
                f"not satisfy its separator rows."
            )
        stranded = int(totals_full.sum() - row_totals.sum())
        if stranded:
            raise RuntimeError(
                f"sweep rounding: at node {node_id}, bag {bag} carries {stranded} records in "
                f"separator groups with no active cell."
            )
        return groups, row_totals, len(present)

    def _solve_transport(self, target: np.ndarray, col_totals: np.ndarray,
                         row_totals: np.ndarray, node_id: int) -> np.ndarray:
        """Solve ONE 2-way transportation problem: K children by m cells, both margins fixed.

        The objective is L1 against the fractional target, written as a convex separable cost in
        three segments per variable: A in [0, floor(t)] at cost -1, B of width 1 at cost
        1 - 2*frac(t), C free above at cost +1. Since -1 <= 1 - 2r <= 1 the LP always fills A,
        then B, then C, and the cost of an integer v then works out to |v - t| - t: the
        objective is L1, up to a constant per variable.

        That encoding is the standard incremental (delta) method for a piecewise-linear cost.
        In general it needs binaries to force the segments to fill in order, a convex cost makes
        them unnecessary, since the slopes increase and a minimising LP fills cheapest first.
        Rigorously: the LP relaxation of the classical formulations approximates the cost by its
        lower convex envelope, which for a convex cost is the cost itself, so the relaxation is
        exact: Croxton, Gendron & Magnanti, "A Comparison of Mixed-Integer Programming Models
        for Nonconvex Piecewise Linear Cost Minimization Problems", Management Science 49(9),
        2003, 1268-1273.

        In network terms, which is what this problem really is, the three segments are three
        parallel arcs between child k and cell c, capacity = segment width, cost = segment
        slope. See Ahuja, Magnanti & Orlin, "Network Flows", Prentice Hall 1993, ch. 14 "Convex
        Cost Flows", which also carries the integrality property relied on below.

        The segments are duplicated columns of the same transportation column, and duplicating
        columns preserves total unimodularity. Integrality also needs the bounds to be integral
        (Hoffman-Kruskal) and they are: floor(t), 1, and infinity. Anything added here later
        (an adjacency box, say) has to keep them integral.

        Method 1 (dual simplex) returns a basic solution, hence a vertex, hence integral.
        Barrier without crossover would return an interior point.

        Args:
            target (np.ndarray): (K, m) fractional target.
            col_totals (np.ndarray): (m,) total per column - the parent node's estimate in that
                cell, which is geographic consistency.
            row_totals (np.ndarray): (K,) total per row - what the neighbouring bag already
                spent on this separator value for each child, which is separator consistency.
            node_id (int): For error messages.

        Returns:
            np.ndarray: (K, m) integers.
        """
        n_children, m = target.shape
        n_vars = 3 * n_children * m

        floor = np.floor(target)
        remainder = target - floor
        upper = np.concatenate([floor.ravel(),
                                np.ones(n_children * m),
                                np.full(n_children * m, GRB.INFINITY)])
        cost = np.concatenate([-np.ones(n_children * m),
                               (1.0 - 2.0 * remainder).ravel(),
                               np.ones(n_children * m)])

        variable = np.arange(n_vars)
        column = variable % m
        child = (variable // m) % n_children

        # m column rows first (the three segments of every child's cell sum to the parent's
        # value), then K row rows (everything child k puts in this group sums to its total).
        rows = np.concatenate([column, m + child])
        cols = np.concatenate([variable, variable])
        rhs = np.concatenate([np.asarray(col_totals, dtype=float),
                              np.asarray(row_totals, dtype=float)])
        matrix = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                               shape=(m + n_children, n_vars))

        model = gp.Model("transport", env=self.env)
        try:
            x = model.addMVar(n_vars, lb=0.0, ub=upper, vtype=GRB.CONTINUOUS)
            model.setObjective(cost @ x, GRB.MINIMIZE)
            model.addMConstr(matrix, x, "=", rhs)
            model.setParam("Method", 1)
            model.optimize()

            if model.SolCount == 0:
                raise RuntimeError(
                    f"sweep rounding: transportation problem infeasible or unsolved at node "
                    f"{node_id} (Gurobi status {model.Status})."
                )
            segments = np.asarray(x.X, dtype=float).reshape(3, n_children, m)
        finally:
            model.dispose()

        totals = segments.sum(axis=0)
        integral = np.rint(totals).astype(np.int64)
        deviation = float(np.abs(totals - integral).max()) if totals.size else 0.0
        if deviation > INTEGRALITY_TOLERANCE:
            raise RuntimeError(
                f"sweep rounding: transportation problem came back fractional at node "
                f"{node_id} "
                f"(max deviation to integer {deviation:g}). The LP should have an integer "
                f"vertex - check that Method=1 took effect."
            )
        return integral
