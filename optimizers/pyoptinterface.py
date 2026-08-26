import pyoptinterface as poi
from pyoptinterface import gurobi, quicksum

import os
import gurobipy as gp
import numpy as np
import scipy.sparse as sp
from typing import List, Optional

from constraints.sparse_constraint import SparseConstraint

_SENSE_MAP = {"=": poi.Eq, "<=": poi.Leq, ">=": poi.Geq}

class OptimizationModel:
    '''
    Use PyOptInterface instead of gurobipy directly.
    PyOptInterface is a thin C++ wrapper that communicates with the native solver API,
    resulting in less Python-level overhead when adding variables and linear constraints.

    As a result, the solve times are practically identical to those obtained with the gurobipy API,
    while model construction times are significantly lower.Compared to writing LP files directly,
    this approach avoids CPU-bound file generation and eliminates kernel calls for disk I/O.
    It also reduces synchronization overhead with other processes that may be waiting for file
    operations to complete. 

    PyOptInterface also provides a matrix-based API.
    However, for the current formulation, it does not offer any performance improvement and
    is actually slightly slower than the approach used here.
    '''

    def __init__(self, dtype: type, lp_problems_dir: str, solver_options: dict = {}) -> None:
        '''Constructor for the OptimizationModel.

        Args:
            dtype (type): NumPy integer type (e.g. np.int32, np.int64) used to cast optimization results.
            lp_problems_dir (str): Directory created by the Data Handler to store LP problem files.
            solver_options (dict): Dictionary of Gurobi parameters to configure the environment and solver.
                Connection/license params (set on Env): ComputeServer, ServerPassword, LogFile, OutputFlag, etc.
                Algorithmic params (inherited by models): Threads, TimeLimit, MIPGap, etc.
        '''

        # NumPy integer type used to cast the last optimization result
        self._solution_type = dtype

        # Directory created by the Data Handler to store LP problem files
        # with infasibles solutions
        self._tmp_dir = lp_problems_dir

        # Each optimizer instance owns its Gurobi environment.
        # Parameters are set before starting the env so they are inherited by every model created from it.
        self.solver_options = solver_options or {}
        self.env = gurobi.Env(empty=True)
        for key, val in self.solver_options.items():
            self.env.set_raw_parameter(key, val)
        self.env.start()

    def non_negative_real_estimation(self, noisy_measurements: List[np.ndarray], node_id: int, constraints: List[SparseConstraint], query_matrix: Optional[np.ndarray] = None, active: Optional[np.ndarray] = None) -> np.ndarray:
        '''Non-negative estimation of the contingency vector using PyOptInterface (Gurobi backend).

        Minimizes sum_k ||Q @ x_k - y_k||^2, where noisy_measurements is a list of per-child
        measurement blocks y_k (each of length n_queries = Q.shape[0]) and the decision
        variable x is the concatenation of per-child cell-count blocks x_k
        (each of length n_cells = Q.shape[1]). For the identity workload, Q = np.eye(n_cells)
        is passed by TopDown.initialize(), so this path handles both cases uniformly.

        Args:
            noisy_measurements (List[np.ndarray]): List of per-child noisy query answers y_k = Q @ x_k + noise.
            node_id (int): The ID of the node for which the estimation is being performed.
            constraints (List[SparseConstraint]): Constraints for all children of the node.
                They are already expressed in terms of active indices and mapped to the global index space.
            query_matrix (Optional[np.ndarray]): Query matrix Q of shape (n_queries, n_cells),
                                            or None for the identity workload.
            active (Optional[np.ndarray]): Global joint-space indices (in 0..n_children*n_cells-1)
                of the non-pruned cells — i.e. {k*n_cells + j} for each child k and each cell j
                in the parent's support. By non-negativity + consistency, children can only be
                non-zero there, so only those variables are created. When None (root / individual
                node), every position is active.

        Returns:
            np.ndarray: Estimated non-negative real cell counts for the active cells only,
                aligned to `active` (position p holds the value for cell active[p]), shape
                (len(active),). Pruned cells are 0 and are not stored; the caller recovers
                each global index from the shared active list by position.
        '''
        model = gurobi.Model(self.env)

        # query_matrix=None is the identity workload case
        is_identity = query_matrix is None
        n_children = len(noisy_measurements)
        if is_identity:
            n_queries = n_cells = len(noisy_measurements[0])
        else:
            n_queries, n_cells = query_matrix.shape   # len(noisy_measurements[i]) == n_queries
        n = n_children * n_cells

        # Active (non-pruned) global indices, in joint cell space.
        active = (np.arange(n, dtype=np.int64) if active is None
                  else np.asarray(active, dtype=np.int64))

        # Everything pruned: the only feasible solution is all zeros (empty active-aligned vector).
        if active.size == 0:
            return np.zeros(0)

        # Only the lifted objective needs the active set, so we can use a set for O(1)
        # membership tests.
        active_set = set() if is_identity else set(active.tolist())
        x = {i: model.add_variable(lb=0.0, domain=poi.VariableDomain.Continuous, name=f"x[{i}]") for i in active}

        # Indices where the matrix Q has nonzeros (in this case just a 1).
        # Needed to detect what are we actually querying for in each row.
        # For sparse CSR: extract nonzero indices directly from internal structure (no densification)
        nz_per_row = [] if is_identity else [
            query_matrix.indices[query_matrix.indptr[r]: query_matrix.indptr[r + 1]]
            for r in range(n_queries)
        ]

        # Auxiliary variables q_x[k, r] = Q[r, :] @ x_k, lifted via linear equalities so the
        # objective stays as a sum of one-term squares. Without this,
        # expanding (sum_j x_j - y)^2 directly would materialise s^2
        # cross terms in the QuadExpr, one per pair of nonzeros in the row.

        # This way we are ensuring that the number of quadratic terms correspond to exactly the number of query answers,
        # which is the intended design of the objective.

        # Only rows with multiple nonzeros need lifting; single-nonzero rows (e.g. the whole
        # identity workload) are squared directly. q_x is created only for the (child, row)
        # pairs that actually need it, so the identity workload allocates no auxiliary vars.
        lift_rows = [] if is_identity else [r for r in range(n_queries) if len(nz_per_row[r]) > 1]
        q_x = {}
        if lift_rows:
            for k in range(n_children):
                for r in lift_rows:
                    q_x[(k, r)] = model.add_variable(
                        domain=poi.VariableDomain.Continuous, name=f"q_x[{k},{r}]"
                    )

            # AuxLink: q_x[k, r] == sum(active x_j for j in Q row r of child k), written
            # as q_x - sum(x) == 0.
            for r in lift_rows:
                nz = nz_per_row[r]
                for k in range(n_children):
                    base = k * n_cells
                    active_terms = [
                        float(query_matrix[r, j]) * x[base + int(j)]
                        for j in nz
                        if (base + int(j)) in active_set
                    ]
                    expr = quicksum(active_terms) if active_terms else 0.0
                    model.add_linear_constraint(
                        q_x[(k, r)] - expr, poi.Eq, 0.0, name=f"AuxLink[{k},{r}]"
                    )

        # Objective: sum_{k, r} (q_x[k, r] - y[k, r])^2 — one quadratic term per (k, r).
        # For single-nonzero Q rows the lift is skipped, so use the underlying x directly.
        # Pruned identity terms reduce to the constant y^2 (x fixed at 0) and are dropped.
        obj_terms = []
        if is_identity:
            # Q = I: query r is cell r, so one square per active cell.
            for g in active:
                k, j = divmod(int(g), n_cells)
                diff = x[g] - float(noisy_measurements[k][j])
                obj_terms.append(diff * diff)
        else:
            for r in range(n_queries):
                nz = nz_per_row[r]
                if len(nz) == 1:
                    j = int(nz[0])
                    for k in range(n_children):
                        base = k * n_cells
                        if (base + j) not in active_set:
                            continue
                        y_kr = float(noisy_measurements[k][r])
                        diff = x[base + j] - y_kr
                        obj_terms.append(diff * diff)
                else:
                    for k in range(n_children):
                        y_kr = float(noisy_measurements[k][r])
                        diff = q_x[(k, r)] - y_kr
                        obj_terms.append(diff * diff)

        model.set_objective(quicksum(obj_terms), poi.ObjectiveSense.Minimize)

        # coef x[i] + coef x[j]... = / <= 7 >= value
        for i, sc in enumerate(constraints):
            lhs_expr = quicksum(float(c) * x[int(idx)] for idx, c in zip(sc.indices, sc.coefs))
            model.add_linear_constraint(lhs_expr, _SENSE_MAP[sc.sense], float(sc.rhs), name=f"Constraint_{i}")

        model.optimize()

        # Check solution
        status = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)

        if status == poi.TerminationStatusCode.OPTIMAL or status == poi.TerminationStatusCode.LOCALLY_SOLVED:
            result = np.array([model.get_value(x[i]) for i in active], dtype=float)
            model.close()
            return result
        elif status == poi.TerminationStatusCode.INFEASIBLE:
            debug_path = os.path.join(self._tmp_dir, f"infeasible_model_node_{node_id}.lp")
            model.write(debug_path)
            model.close()
            raise ValueError(f"Model is infeasible for node {node_id}. See {debug_path} for debugging.")
        else:
            model.close()
            if status == poi.TerminationStatusCode.TIME_LIMIT:
                raise RuntimeError(
                    f"The QP for node {node_id} hit the TimeLimit in solver_options "
                    f"({self.solver_options.get('TimeLimit')} s) before converging. Unlike the "
                    f"rounding step, a truncated QP is not usable: barrier returns an interior "
                    f"point that need not satisfy the geographic or separator rows, and "
                    f"everything downstream assumes it does - the sweep rounder reads the "
                    f"parent's integer counts straight off it. Raise TimeLimit or drop it, "
                    f"rather than accepting what came back."
                )
            raise RuntimeError(f"Solver failed for node {node_id}. Status: {status}")

    def rounding_estimation(self, x_tilde: np.ndarray, node_id: int, constraints: List[SparseConstraint], active: Optional[np.ndarray] = None, n: Optional[int] = None) -> sp.csc_matrix:
        '''Rounding estimation of the contingency vector using PyOptInterface (Gurobi backend).

        Uses a linearized objective: since y_i ∈ {0,1}, y_i² = y_i. The floor part is moved
        to the right-hand side once per constraint.

        Args:
            x_tilde (np.ndarray): Non-negative real cell counts from the QP step, aligned to active.
            node_id (int): The ID of the node for which the estimation is being performed.
            constraints (List[SparseConstraint]): Sparse constraint objects, already offset
                into joint space and restricted to active indices by the caller.
            active (Optional[np.ndarray]): Global joint-space indices of non-pruned cells.
            n (Optional[int]): Joint vector length, required when active is provided.

        Returns:
            scipy.sparse.csc_matrix: Column vector of shape (n, 1) with integer cell counts.
        '''
        model = gurobi.Model(self.env)

        x_tilde = np.asarray(x_tilde, dtype=float)
        x_floor = np.floor(x_tilde)                     # aligned to active (positional)
        residual_round = x_tilde - x_floor              # aligned to active (positional)

        # Resolve active / n. x_tilde is aligned to active, so len(x_tilde) == len(active).
        if active is None:
            n = len(x_tilde)
            active = np.arange(n, dtype=np.int64)
        else:
            active = np.asarray(active, dtype=np.int64)
            if n is None:
                raise ValueError("rounding_estimation requires `n` when `active` is provided.")
            if len(x_tilde) != len(active):
                raise ValueError(
                    f"x_tilde length {len(x_tilde)} does not match active length {len(active)}."
                )

        # If everything is pruned, no binary decisions needed: all zeros.
        if active.size == 0:
            return sp.csc_matrix((n, 1), dtype=np.int64)

        # Maps each global index to its position in the `active` list (and therefore in
        # `x_floor`, which is aligned to `active`). x_floor runs from 0 to len(active)-1,
        # so x_floor[0] holds the floor value for active[0], x_floor[1] for active[1], etc.
        # SparseConstraints store global indices (e.g. active[0] = 7), so to recover the
        # corresponding floor value we need the reverse mapping: pos_of[7] -> 0,
        # which gives x_floor[0] directly, without an O(n) search through `active`.
        pos_of = {g: p for p, g in enumerate(active)}

        y = {i: model.add_variable(domain=poi.VariableDomain.Binary, name=f"y[{i}]") for i in active }

        # Objective: linear. For binary vars, (r-y)² = r² + y(1-2r); drop the constant r².
        model.set_objective(
            quicksum((1.0 - 2.0 * residual_round[p]) * y[i] for p, i in enumerate(active)),
            poi.ObjectiveSense.Minimize,
        )

        for i, sc in enumerate(constraints):
            floor_contrib = 0.0
            terms = []

            # The rounding decision variable is only the binary correction y[i]. 
            # The actual cell value is floor[i] + y[i]. 
            # <=> sum(coef_i * (floor_i + y_i)) sense rhs
            # <=> sum(coef_i * y_i) sense (rhs - sum(coef_i * floor_i))
            for idx, coef in zip(sc.indices, sc.coefs):
                idx = int(idx)
                floor_contrib += float(coef) * x_floor[pos_of[idx]]  # coef_i * floor_i
                terms.append(float(coef) * y[idx])                   # coef_i * y_i
            
            model.add_linear_constraint(quicksum(terms), _SENSE_MAP[sc.sense], float(sc.rhs) - floor_contrib, name=f"Constraint_{i}")

        model.optimize()

        # Check solution.
        # TIME_LIMIT with an incumbent is a usable answer, not a failure. Every hard
        # constraint holds exactly in any incumbent, so a truncated solution is feasible.
        status = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
        accepted = (status == poi.TerminationStatusCode.OPTIMAL
                    or status == poi.TerminationStatusCode.LOCALLY_SOLVED
                    or (status == poi.TerminationStatusCode.TIME_LIMIT
                        and model.get_model_attribute(poi.ModelAttribute.PrimalStatus)
                        == poi.ResultStatusCode.FEASIBLE_POINT))
        if not accepted:
            if status == poi.TerminationStatusCode.INFEASIBLE:
                debug_path = os.path.join(self._tmp_dir, f"infeasible_model_node_{node_id}.lp")
                model.write(debug_path)
                model.close()
                raise ValueError(f"Model is infeasible for node {node_id}. See {debug_path} for debugging.")
            else:
                model.close()
                if status == poi.TerminationStatusCode.TIME_LIMIT:
                    raise RuntimeError(
                        f"The rounding MIP for node {node_id} hit the TimeLimit in "
                        f"solver_options ({self.solver_options.get('TimeLimit')} s) with no "
                        f"incumbent at all, so there is nothing to return - an incumbent would "
                        f"have been accepted. On these models the root relaxation is usually "
                        f"what did not finish, not the search. Raise TimeLimit, set a reachable "
                        f"MIPGap (1e-4 is Gurobi's default and is not reachable here), or use "
                        f"the junction-tree sweep."
                    )
                raise RuntimeError(f"Solver failed for node {node_id}. Status: {status}")

        # Reconstruct as a sparse column vector, keeping only positive cells.
        # x_floor is positional (aligned to active); i is the global index for the CSC row.
        rows = []
        data = []
        for p, i in enumerate(active):
            val = int(x_floor[p] + round(model.get_value(y[i])))
            if val > 0:
                rows.append(i)
                data.append(val)

        model.close()

        return sp.csc_matrix(
            (data, (rows, np.zeros(len(rows), dtype=self._solution_type))),
            shape=(n, 1),
            dtype=self._solution_type,
        )