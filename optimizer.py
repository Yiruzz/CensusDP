import pyomo.environ as pyo
import gurobipy as gp

import numpy as np
import scipy.sparse as sp
from typing import List, Callable, Any, Optional, Union

from constraints.sparse_constraint import SparseConstraint


class OptimizationModel:
    '''
    Represents the Pyomo model that is used to do the estimation of the contingency vectors.

    Pyomo builds the model and is an interface over the actual optimization engine that solves the problem (e.g. Gurobi).
    Uses ConcreteModels for direct model construction.
    '''

    def __init__(self, solver_name='gurobi', solver_options={}, optimizer_path=None) -> None:
        '''Constructor for the OptimizationModel class.

        Args:
            solver_name (str): The name of the solver to use. Defaults to 'gurobi'.
            solver_options (dict): Dictionary of options to pass to the solver.
            optimizer_path (str): Path to the optimizer executable. If None, defaults to None.
        '''
        self.solver = pyo.SolverFactory(solver_name, manage_env= True)
        self.solver_options = solver_options
        if optimizer_path is not None:
            self.solver.set_executable(optimizer_path)

    def _solve_pyomo_model(self, instance: pyo.ConcreteModel, node_id: int) -> Any:
        '''Auxiliar function to solve the instance of the model and handle infeasibility.

        Args:
            instance (ConcreteModel): The concrete model instance to solve.
            node_id (int): The id of the node being solved.

        Returns:
            SolverResults: The results from the solver.
        '''
        # Use the solver options provided during initialization
        results = self.solver.solve(instance, tee=False, options=self.solver_options)

        # Check termination conditions
        if results.solver.termination_condition == pyo.TerminationCondition.optimal or \
           results.solver.termination_condition == pyo.TerminationCondition.locallyOptimal:
            return results
        elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
            # Write model for debugging
            filename = f"infeasible_model_node_{node_id}.nl"
            instance.write(filename)
            raise ValueError(f'Model is infeasible for node {node_id}. See {filename} file for debugging.')
        else:
            raise RuntimeError(f"Solver termination failed for node {node_id}. Status: {results.solver.status}, Condition: {results.solver.termination_condition}")

    def non_negative_real_estimation(self, noisy_measurements: np.ndarray, node_id: int, constraints: Union[List[Callable], List[SparseConstraint]], query_matrix: np.ndarray, active: Optional[List[int]] = None) -> np.ndarray:
        '''Non-negative estimation of the contingency vector using Pyomo ConcreteModel.

        Minimizes sum_k ||Q @ x_k - y_k||^2, where noisy_measurements is the concatenation
        of per-child measurement blocks y_k (each of length n_queries = Q.shape[0]) and the
        decision variable x is the concatenation of per-child cell-count blocks x_k
        (each of length n_cells = Q.shape[1]). For the identity workload, Q = np.eye(n_cells)
        is passed by TopDown.initialize(), so this path handles both cases uniformly.

        Constraints are always expressed in cell space (indices 0..n_cells-1 per child).

        Args:
            noisy_measurements (np.ndarray): Concatenated noisy query answers y = Q @ x + noise.
            node_id (int): The ID of the node for which the estimation is being performed.
            constraints (List[Callable]): Constraint callables, each given the decision accessor
                (here the raw Var, valid on the active global indices) and returning a Pyomo
                expression. Referencing a non-active index is the caller's responsibility to avoid.
            query_matrix (np.ndarray): Query matrix Q of shape (n_queries, n_cells).
            active (Optional[List[int]]): Global joint-space indices (in 0..n_children*n_cells-1)
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
        n_queries, n_cells = query_matrix.shape
        n_children = len(noisy_measurements) // n_queries
        n = n_children * n_cells

        # Active (non-pruned) global indices, in joint cell space.
        active = list(range(n)) if active is None else list(active)

        # Everything pruned: the only feasible solution is all zeros (empty active-aligned vector).
        if not active:
            return np.zeros(0)

        active_set = set(active)

        # Create a ConcreteModel directly
        instance = pyo.ConcreteModel(name=f'RealEstimation_NodeID_{node_id}')

        # Decision variables only for active cells.
        instance.I = pyo.Set(initialize=active, ordered=True)
        instance.x = pyo.Var(instance.I, domain=pyo.NonNegativeReals)

        # Indices where the matrix Q has nonzeros (in this case just a 1).
        # Needed to detect what are we actually querying for in each row.
        # For sparse CSR: extract nonzero indices directly from internal structure (no densification)
        nz_per_row = [query_matrix.indices[query_matrix.indptr[r]:query_matrix.indptr[r+1]]
                      for r in range(n_queries)]

        # Auxiliary variables q_x[k, r] = Q[r, :] @ x_k, lifted via linear equalities so the
        # objective stays as a sum of one-term squares. Without this, squaring a Pyomo
        # LinearExpression with s nonzeros materialises s^2 cross terms. Problematic for dense Q.
        # Rows with a single nonzero (e.g. identity workload) skip the lift to avoid one pointless equality per row.

        # This way we are ensuring that the number of quadratic terms correspond to exactly the number of query answers,
        # which is the intended design of the objective.

        # Only rows with multiple nonzeros need lifting; single-nonzero rows (e.g. the whole
        # identity workload) are squared directly. q_x is created only for the (child, row)
        # pairs that actually need it, so the identity workload allocates no auxiliary vars.
        lift_rows = [r for r in range(n_queries) if len(nz_per_row[r]) > 1]
        if lift_rows:
            instance.QX = pyo.Set(initialize=[(k, r) for k in range(n_children) for r in lift_rows], dimen=2)
            instance.q_x = pyo.Var(instance.QX, domain=pyo.Reals)

            # q_x[k, r] = Q[r, :] @ x_k restricted to active cells.
            instance.AuxLink = pyo.ConstraintList()
            for r in lift_rows:
                nz = nz_per_row[r]
                for k in range(n_children): # Iterate over the joint space
                    base = k * n_cells
                    active_terms = [instance.x[base + int(j)] for j in nz
                                    if (base + int(j)) in active_set]
                    # Q is binary, so q_x[k, r] is just the sum of the queried cells for child k.
                    instance.AuxLink.add(instance.q_x[k, r] == (sum(active_terms) if active_terms else 0))

        # Objective: sum_{k, r} (q_x[k, r] - y[k, r])^2 — one quadratic term per (k, r).
        # For single-nonzero Q rows the lift is skipped, so use the underlying x directly.
        # Pruned identity terms reduce to the constant y^2 (x fixed at 0) and are dropped.
        def objective_rule(model):
            total = 0
            for r in range(n_queries):
                nz = nz_per_row[r]
                if len(nz) == 1: # Identity workload case
                    j = int(nz[0])
                    for k in range(n_children):
                        base = k * n_cells
                        if (base + j) not in active_set:
                            continue  # x pruned to 0 -> constant term, does not affect argmin
                        y_kr = float(noisy_measurements[k * n_queries + r])
                        total += (model.x[base + j] - y_kr) ** 2
                else: # General case with lifted variables
                    for k in range(n_children):
                        y_kr = float(noisy_measurements[k * n_queries + r]) # The noisy value for the k-th child and r-th query
                        total += (model.q_x[k, r] - y_kr) ** 2 # just a simple squared error with the lifted variable q_x[k, r]
            return total

        instance.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Add constraints: support both SparseConstraint and legacy callables
        instance.ConstraintList = pyo.ConstraintList()
        for i, constraint in enumerate(constraints):
            try:
                # SparseConstraint: build expression from indices and coefs
                lhs = sum(float(c) * instance.x[int(idx)] for idx, c in zip(constraint.indices, constraint.coefs))

                if constraint.sense == "=":
                    pyomo_expr = lhs == constraint.rhs
                elif constraint.sense == "<=":
                    pyomo_expr = lhs <= constraint.rhs
                elif constraint.sense == ">=":
                    pyomo_expr = lhs >= constraint.rhs
                else:
                    raise ValueError(f"Unknown sense: {constraint.sense}")

                instance.ConstraintList.add(pyomo_expr)
                
            except Exception as e:
                print(f"Error adding constraint {i}: {e}")
                raise e

        # Solve the model
        self._solve_pyomo_model(instance, node_id)

        # Extract results aligned to active: position p holds the value for cell active[p].
        # Pruned cells are not stored (their value is 0); the caller recovers each global
        # index from the shared active list by position, so no index is re-stored here.
        return np.array([pyo.value(instance.x[i]) for i in active], dtype=float)

    def rounding_estimation(self, x_tilde: np.ndarray, node_id: int, constraints: Union[List[Callable], List[SparseConstraint]], active: Optional[List[int]] = None, n: Optional[int] = None) -> sp.csc_matrix:
        '''Rounding estimation of the contingency vector using Pyomo ConcreteModel.

        Uses a linearized objective: since y_i ∈ {0,1}, y_i² = y_i.
        The linear form avoids MIQP and lets MIP solvers use faster LP-based branching.

        When active is provided, cell positions outside it (where the parent value is 0) are
        excluded from the binary decision space entirely.

        Args:
            x_tilde (np.ndarray): Non-negative real cell counts from the previous (QP) step,
                aligned to active (position p is the value for cell active[p]), length
                len(active). When active is None, this is the full vector of length n.
            node_id (int): The ID of the node for which the estimation is being performed.
            constraints (List[Callable]): Constraint callables, each given the decision accessor
                (here a {global_index: floor+binary} dict valid on the active indices) and returning
                a Pyomo expression. Referencing a non-active index is the caller's responsibility to
                avoid.
            active (Optional[List[int]]): Global joint-space indices (in 0..n-1) of the
                non-pruned cells. Only those positions get a binary variable. When None (root /
                individual node), every position is active and n is inferred from x_tilde.
            n (Optional[int]): Joint vector length n_children * n_cells, used for the CSC shape.
                Required when active is provided (not derivable from the support); ignored when
                active is None (then n = len(x_tilde)).

        Returns:
            scipy.sparse.csc_matrix: Column vector of shape (n, 1) with non-negative integer
                cell counts. Stores only positive cells (canonical: no explicit zeros).
        '''
        x_tilde = np.asarray(x_tilde, dtype=float)
        x_floor = np.floor(x_tilde)               # aligned to active (positional)
        residual_round = x_tilde - x_floor        # aligned to active (positional)

        # Resolve active / n. x_tilde is aligned to active, so len(x_tilde) == len(active).
        if active is None:
            n = len(x_tilde)
            active = list(range(n))
        else:
            active = list(active)
            if n is None:
                raise ValueError("rounding_estimation requires `n` (joint vector length) when `active` is provided.")
            if len(x_tilde) != len(active):
                raise ValueError(f"x_tilde length {len(x_tilde)} does not match active length {len(active)}.")

        # If everything is pruned, no binary decisions needed: all zeros.
        if not active:
            return sp.csc_matrix((n, 1), dtype=np.int64)

        # Create a ConcreteModel directly
        instance = pyo.ConcreteModel(name=f'RoundingEstimation_NodeID_{node_id}')

        # Index set: only active (non-zero-parent) positions.
        instance.I = pyo.Set(initialize=active, ordered=True)

        # Parameters: residual and floor values, keyed by global index, value taken by position.
        instance.r = pyo.Param(instance.I, initialize={i: float(residual_round[p]) for p, i in enumerate(active)})
        instance.f = pyo.Param(instance.I, initialize={i: float(x_floor[p]) for p, i in enumerate(active)})

        # Decision variable: binary (only for active indices)
        instance.y = pyo.Var(instance.I, domain=pyo.Binary)

        # Objective: linear. For binary vars, (r-y)² = r² + y(1-2r); drop the constant r².
        def objective_rule(model):
            return sum(model.y[i] * (1 - 2 * model.r[i]) for i in model.I)

        instance.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # The rounding decision variable is only the binary correction y[i]. The actual cell value
        # is floor[i] + y[i]. Constraints need the full value, so expose it as a plain dict over the
        # active indices for legacy callables.
        cell_value = {i: instance.f[i] + instance.y[i] for i in active}

        # Add constraints: support both SparseConstraint and legacy callables
        instance.ConstraintList = pyo.ConstraintList()
        for i, constraint in enumerate(constraints):
            try:
                # SparseConstraint: build expression from indices and coefs
                lhs = sum(float(c) * cell_value[int(idx)] for idx, c in zip(constraint.indices, constraint.coefs)) 

                if constraint.sense == "=":
                    pyomo_expr = lhs == constraint.rhs
                elif constraint.sense == "<=":
                    pyomo_expr = lhs <= constraint.rhs
                elif constraint.sense == ">=":
                    pyomo_expr = lhs >= constraint.rhs
                else:
                    raise ValueError(f"Unknown sense: {constraint.sense}")

                instance.ConstraintList.add(pyomo_expr)
               
            except Exception as e:
                print(f"Error adding constraint {i}: {e}")
                raise e

        # Solve the model
        self._solve_pyomo_model(instance, node_id)

        # Reconstruct as a sparse column vector, keeping only positive cells.
        # x_floor is positional (aligned to active); i is the global index for the CSC row.
        rows = []
        data = []
        for p, i in enumerate(active):
            val = int(x_floor[p] + pyo.value(instance.y[i]))
            if val > 0:
                rows.append(i)
                data.append(val)

        return sp.csc_matrix(
            (data, (rows, np.zeros(len(rows), dtype=np.int64))),
            shape=(n, 1),
            dtype=np.int64,
        )
