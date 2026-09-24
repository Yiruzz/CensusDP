import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
from typing import List, Optional, IO
import tempfile
import os

from constraints.sparse_constraint import SparseConstraint

# LP format's hard line-length limit is 999 characters (see Gurobi's LP format reference).
# Expressions can freely span multiple physical lines (continuation at token boundaries), so
# we insert a line break every this-many terms in any loop that could otherwise produce an
# arbitrarily long single line (objective terms, AuxLink sums, SparseConstraint rows). At
# ~25-30 chars per term in the worst case, 15 terms/line stays comfortably under the limit.
TERMS_PER_LINE = 15

def _fmt(c: float) -> str:
    '''Format a coefficient with a leading sign and full double precision (round-trips exactly through text), e.g. 1.0 -> "+1", -2.5 -> "-2.5".'''
    return f"{c:+.17g}"

def _write_term(f: IO, term: str, count: int, terms_per_line: int = TERMS_PER_LINE) -> int:
    '''Write one already-formatted " coef var"-style term, breaking the physical line every
    `terms_per_line` terms so a long expression never approaches the LP format's 999-character
    line limit. Returns the updated term count (pass it back in on the next call).

    Args:
        f: Open file handle being streamed to.
        term (str): The term text, including its own leading space (e.g. " +2 x[5]").
        count (int): Number of terms written so far in this expression, before this one.
        terms_per_line (int): How many terms to allow per physical line.

    Returns:
        int: count + 1.
    '''
    f.write(term)
    count += 1
    if count % terms_per_line == 0:
        f.write("\n")
    return count


def _write_identity_objective(f: IO, noisy_measurements: List[np.ndarray],
                               active: np.ndarray, n_cells: int) -> None:
    """Objective for the IDENTITY workload: sum_k ||x_k - y_k||^2, in one pass over `active`.

    With Q = I every row has its single nonzero on the diagonal, so query r is cell r and the
    objective decomposes into one independent square per active cell.

    Args:
        f (IO): Open LP file, positioned right after the "obj:" tag.
        noisy_measurements (List[np.ndarray]): Per-child measurement y_k, length n_cells.
        active (np.ndarray): Global joint-space indices k * n_cells + j, ascending.
        n_cells (int): Cells per child, used to split a global index into (child, cell).
    """
    # Pass 1: linear terms  -2*y * x[g]
    term_count = 0
    for g in active:
        k, j = divmod(int(g), n_cells)
        y = float(noisy_measurements[k][j])
        if y == 0.0:
            continue  # -2*0*x = 0
        term_count = _write_term(f, f" {_fmt(-2.0 * y)} x[{g}]", term_count)

    # Pass 2: quadratic terms  2 * x[g]^2, wrapped in the "[ ... ] / 2" LP convention so each
    # evaluates to 1 * x^2.
    if active.size:
        f.write(" + [")
        for g in active:
            term_count = _write_term(f, f" {_fmt(2.0)} x[{g}] ^ 2", term_count)
        f.write(" ] / 2")
    elif term_count == 0:
        f.write(" 0")


def _write_workload_objective(f: IO, noisy_measurements: List[np.ndarray], active_set: set,
                              nz_per_row: List[np.ndarray], n_queries: int,
                              n_children: int, n_cells: int) -> None:
    """Objective for a general workload: sum_k ||Q x_k - y_k||^2.

    Rows with several nonzeros are lifted through the auxiliary q_x[k, r] so the objective
    stays a sum of one-term squares; single-nonzero rows are squared directly on x.

    Args:
        f (IO): Open LP file, positioned right after the "obj:" tag.
        noisy_measurements (List[np.ndarray]): Per-child measurement y_k, length n_queries.
        active_set (set): Global joint-space indices that survived pruning.
        nz_per_row (List[np.ndarray]): Column indices of Q's nonzeros, per row.
        n_queries (int): Rows of Q.
        n_children (int): Children solved jointly.
        n_cells (int): Cells per child.
    """
    # Pass 1: linear terms, written directly.
    term_count = 0
    for r in range(n_queries):
        nz = nz_per_row[r]
        if len(nz) == 1:  # Identity-like row
            j = int(nz[0])
            for k in range(n_children):
                base = k * n_cells
                if (base + j) not in active_set:
                    continue
                y_kr = float(noisy_measurements[k][r])
                if y_kr == 0.0:
                    continue  # -2*0*x = 0
                term_count = _write_term(f, f" {_fmt(-2.0 * y_kr)} x[{base + j}]", term_count)
        else:  # General case with lifted variables
            for k in range(n_children):
                y_kr = float(noisy_measurements[k][r])  # noisy value for child k and query r
                if y_kr == 0.0:  # -2*0*q_x = 0
                    continue
                term_count = _write_term(f, f" {_fmt(-2.0 * y_kr)} q_x[{k},{r}]", term_count)

    # Pass 2: quadratic terms
    bracket_open = False
    for r in range(n_queries):
        nz = nz_per_row[r]
        if len(nz) == 1:
            j = int(nz[0])
            for k in range(n_children):
                base = k * n_cells
                if (base + j) not in active_set:
                    continue  # x pruned to 0 -> constant term, does not affect argmin
                if not bracket_open:
                    f.write(" + [")
                    bracket_open = True
                term_count = _write_term(f, f" {_fmt(2.0)} x[{base + j}] ^ 2", term_count)
        else:
            for k in range(n_children):
                if not bracket_open:
                    f.write(" + [")
                    bracket_open = True
                term_count = _write_term(f, f" {_fmt(2.0)} q_x[{k},{r}] ^ 2", term_count)
    if bracket_open:
        f.write(" ] / 2")
    if term_count == 0:
        f.write(" 0")


class OptimizationModelLP:
    '''
    Builds the model as raw Gurobi LP-format text (no gurobipy Var/Constr objects at all
    during construction) and loads it directly with gp.read(). This bypasses the Python
    object API entirely for model construction; only reading the file and extracting the
    solution still goes through gurobipy.

    STREAMING CONSTRUCTION: lines are written directly to the temp file as they're generated,
    never accumulated into a Python list/string first. The one structural subtlety this
    requires: LP format wants the objective's linear terms before its quadratic bracket
    `[ ... ] / 2`, but each query row produces one linear AND one quadratic contribution at
    once. Rather than buffer one of the two, we do two cheap passes over `range(n_queries)`
    (not over the constraints, which is where actual memory/time would matter) — pass 1
    writes linear terms, pass 2 writes quadratic terms.

    LINE LENGTH: LP format caps physical lines at 999 characters, but explicitly allows an
    expression to continue across multiple lines at any token boundary. Every loop that could
    produce many terms on what would otherwise be one giant line (objective terms, AuxLink
    sums, SparseConstraint rows) goes through `_write_term`, which breaks the line every
    `TERMS_PER_LINE` terms.

    IMPORTANT — the quadratic objective convention: Gurobi's LP format represents the
    quadratic part of the objective as `[ ... ] / 2`, mirroring the solver's internal
    f(x) = (1/2) x^T Q x + c^T x convention. Whatever coefficient you write inside the
    brackets gets halved. Since every quadratic term in this model is a single squared
    variable with the desired final coefficient 1.0 (no cross terms — that's exactly why
    the q_x lift exists), every quadratic term is written as "2 var ^ 2" (spaces on both
    sides of `^`, matching the form in Gurobi's own LP format reference examples) so that,
    after the implicit /2, it evaluates to "1 var ^ 2" as intended.
    '''

    def __init__(self, dtype: type, lp_problems_dir: str, solver_options: dict = {}) -> None:
        '''Constructor for the OptimizationModel.

        Args:
            dtype (type): NumPy integer type (e.g. np.int32, np.int64) used to cast optimization results.
            lp_problems_dir (str): Directory created by the Data Handler to temporarily store LP problem files.
            solver_options (dict): Dictionary of Gurobi parameters to configure the environment and solver.
                Connection/license params (set on Env): ComputeServer, ServerPassword, LogFile, OutputFlag, etc.
                Algorithmic params (inherited by models): Threads, TimeLimit, MIPGap, etc.
        '''
        # NumPy integer type used to cast the last optimization result
        self._solution_type = dtype

        # Directory created by the Data Handler to temporarily store LP problem files
        self._tmp_dir = lp_problems_dir

        # Each optimizer instance owns its Gurobi environment.
        # Parameters are set before starting the env so they are inherited by every model created from it.
        self.solver_options = solver_options or {}
        self.env = gp.Env(empty=True)
        for key, val in self.solver_options.items():
            self.env.setParam(key, val)
        self.env.start()
        
    def non_negative_real_estimation(self, noisy_measurements: List[np.ndarray], node_id: int, constraints: List[SparseConstraint], query_matrix: Optional[np.ndarray] = None, active: Optional[np.ndarray] = None) -> np.ndarray:
        '''Non-negative estimation of the contingency vector, written directly to an .lp file.
        There is no container that encapsulates all elements, like Pyomo's ConcreteModel.

        Minimizes sum_k ||Q @ x_k - y_k||^2, where noisy_measurements is a list of per-child
        measurement blocks y_k (each of length n_queries = Q.shape[0]) and the decision
        variable x is the concatenation of per-child cell-count blocks x_k
        (each of length n_cells = Q.shape[1]).

        Constraints are always expressed in cell space (indices 0..n_cells-1 per child).

        Args:
            noisy_measurements (List[np.ndarray]): List of per-child noisy query answers y_k = Q @ x_k + noise.
            node_id (int): The ID of the node for which the estimation is being performed.
            constraints (List[SparseConstraint]): Constraints for all children of the node.
                They are already expressed in terms of active indices and mapped to the global index space.
            query_matrix (Optional[np.ndarray]): Query matrix Q of shape (n_queries, n_cells),
                or None for the identity workload, which is never materialised.
            active (Optional[np.ndarray]): Global joint-space indices (in 0..n_children*n_cells-1)
                of the non-pruned cells — i.e. {k*n_cells + j} for each child k and each cell j
                in the parent's support.

        Returns:
            np.ndarray: Estimated non-negative real cell counts for the active cells only,
                aligned to `active` (position p holds the value for cell active[p]), shape
                (len(active),). Pruned cells are 0 and are not stored; the caller recovers
                each global index from the shared active list by position.
        '''
        is_identity = query_matrix is None
        n_children = len(noisy_measurements)
        if is_identity:
            # Under the identity n_queries == n_cells, and the measurement length gives both
            n_queries = n_cells = len(noisy_measurements[0])
        else:
            n_queries, n_cells = query_matrix.shape   # len(noisy_measurements[i]) == n_queries
        n = n_children * n_cells

        # Active (non-pruned) global indices, in joint cell space.
        active = (np.arange(n, dtype=np.int64) if active is None
                  else np.asarray(active, dtype=np.int64))

        # Everything pruned: the only feasible solution is all zeros (empty active-aligned vector)
        if active.size == 0:
            return np.zeros(0)

        # Only the lifted objective needs membership tests; the identity one walks `active` directly.
        active_set = set() if is_identity else set(active.tolist())

        # Indices where the matrix Q has nonzeros (in this case just a 1).
        # Needed to detect what are we actually querying for in each row.
        # For sparse CSR: extract nonzero indices directly from internal structure (no densification)
        nz_per_row = [] if is_identity else [
            query_matrix.indices[query_matrix.indptr[r] : query_matrix.indptr[r + 1]]
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
        q_x = [(k, r) for r in lift_rows for k in range(n_children)]

        # Start writing
        # \\ Comments to identify the file
        #
        # Minimize
        # obj: <objective expression>   # First linear_terms then [ 2 * quadratics ^ 2 ] / 2
        #
        # Subject To
        # name1: <constraint 1>
        # name2: <constraint 2>
        #
        # Bounds
        # ...
        #
        # End

        fd, tmp_path = tempfile.mkstemp(suffix=".lp", dir=self._tmp_dir)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(f"\\ Model RealEstimation_NodeID_{node_id}\n")
                f.write("\\ LP format - for model browsing. Use MPS format to capture full model detail.\n")
                
                # ---------------------------------------------------------------------------
                # Objective
                # ---------------------------------------------------------------------------

                # Objective: sum_{k, r} (q_x[k, r] - y[k, r])^2 — one quadratic term per (k, r).
                # For single-nonzero Q rows the lift is skipped, so use the underlying x directly.
                # Pruned identity terms reduce to the constant y^2 (x fixed at 0) and are dropped.
                f.write("Minimize\n")
                f.write(" obj:")

                # Constant terms do not affect the optimal solution and are not written.
                if is_identity:
                    _write_identity_objective(f, noisy_measurements, active, n_cells)
                else:
                    _write_workload_objective(
                        f, noisy_measurements, active_set, nz_per_row,
                        n_queries, n_children, n_cells)
                f.write("\n")

                # ---------------------------------------------------------------------------
                # Constraints
                # ---------------------------------------------------------------------------
                
                f.write("Subject To\n")

                # AuxLink: q_x[k, r] == sum(active x_j for j in Q row r of child k), written
                # as -q_x + sum(x) == 0.
                cidx = 0
                for r in lift_rows:
                    nz = nz_per_row[r]
                    for k in range(n_children):
                        base = k * n_cells

                        f.write(f" AuxLink_{cidx}:") # Only a name for the constraint

                        # -q_x + sum(x)
                        tcount = _write_term(f, f" {_fmt(-1.0)} q_x[{k},{r}]", 0)
                        for j in nz:
                            g = base + int(j)
                            if g in active_set:
                                tcount = _write_term(f, f" {_fmt(1.0)} x[{g}]", tcount)
                        
                        # == 0
                        f.write(" = 0\n")
                        
                        cidx += 1

                # SparseConstraint rows.
                for i, sc in enumerate(constraints):
                    f.write(f" Constraint_{i}:") # Only a name for the constraint

                    # + coef x[i] + coef x[j]...
                    tcount = 0
                    for idx, c in zip(sc.indices, sc.coefs):
                        tcount = _write_term(f, f" {_fmt(float(c))} x[{int(idx)}]", tcount)
                    
                    # = / <= / >= rhs # (unsigned, positive by default: cell counts are non-negative)
                    f.write(f" {sc.sense} {sc.rhs:.17g}\n")

                # ---------------------------------------------------------------------------
                # Bounds
                # ---------------------------------------------------------------------------

                # x defaults to [0, +inf) in LP format, matching Gurobi's own writer,
                # which omits bound lines for variables already at their default. q_x MUST be
                # declared free — LP format's implicit default lower bound is 0, which would
                # silently (and wrongly) forbid the negative q_x values the lift allows.
                # The "Bounds" header itself is only written lazily,
                # right before the first bound line, to avoid an empty section.
                if len(q_x) > 0:  f.write("Bounds\n")
                for (k, r) in q_x:
                    f.write(f" q_x[{k},{r}] free\n")

                # ---------------------------------------------------------------------------
                # End
                # ---------------------------------------------------------------------------
                f.write("End\n")
            
            model = gp.read(tmp_path, env=self.env)
            model.optimize()

            if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
                result = np.array([model.getVarByName(f"x[{i}]").X for i in active], dtype=float)
                model.dispose()
                return result
            
            elif model.status == GRB.INFEASIBLE:
                debug_path = os.path.join(self._tmp_dir, f"infeasible_model_node_{node_id}.lp")
                model.write(debug_path)
                model.dispose()
                raise ValueError(f"Model is infeasible for node {node_id}. See {debug_path}for debugging. ")
            else:
                status = model.status
                model.dispose()
                if status == GRB.TIME_LIMIT:
                    raise RuntimeError(
                        f"The QP for node {node_id} hit the TimeLimit in solver_options "
                        f"({self.solver_options.get('TimeLimit')} s) before converging. Unlike "
                        f"the rounding step, a truncated QP is not usable: barrier returns an "
                        f"interior point that need not satisfy the geographic or separator "
                        f"rows, and everything downstream assumes it does - the sweep rounder "
                        f"reads the parent's integer counts straight off it. Raise TimeLimit "
                        f"or drop it, rather than accepting what came back."
                    )
                raise RuntimeError(f"Solver failed for node {node_id}. Status: {status}")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def rounding_estimation(self, x_tilde: np.ndarray, node_id: int, constraints: List[SparseConstraint], active: Optional[np.ndarray] = None, n: Optional[int] = None) -> sp.csc_matrix:
        '''Rounding estimation of the contingency vector, written directly to an .lp file.

        Purely linear (binary objective linearized as y_i, see other versions' docstrings for
        the derivation), so this one needs no multi-pass trick — every section is a simple
        single pass, written line by line as it's generated.

        Args:
            x_tilde (np.ndarray): Non-negative real cell counts from QP step, aligned to active.
            node_id (int): The ID of the node for which the estimation is being performed.
            constraints (List[SparseConstraint]): Sparse constraint objects, already offset
                into joint space and restricted to active indices by the caller.
            active (Optional[np.ndarray]): Global joint-space indices of non-pruned cells.
            n (Optional[int]): Joint vector length, required when active is provided.

        Returns:
            scipy.sparse.csc_matrix: Column vector of shape (n, 1) with integer cell counts.
        '''
        x_tilde = np.asarray(x_tilde, dtype=float)
        x_floor = np.floor(x_tilde)             # aligned to active (positional)
        residual_round = x_tilde - x_floor      # aligned to active (positional)

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

        # Start writing
        # \\ Comments to identify the file
        #
        # Minimize
        # obj: <objective expression>   # Only linear_terms
        #
        # Subject To
        # name1: <constraint 1>
        # name2: <constraint 2>
        #
        # Binaries
        # ...
        #
        # End
        fd, tmp_path = tempfile.mkstemp(suffix=".lp", dir=self._tmp_dir)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(f"\\ Model RoundingEstimation_NodeID_{node_id}\n")
                f.write("\\ LP format - for model browsing. Use MPS format to capture full model detail.\n")
                
                # ---------------------------------------------------------------------------
                # Objective
                # ---------------------------------------------------------------------------
                
                f.write("Minimize\n")
                f.write(" obj:")

                # Objective: linear. For binary vars, (r-y)² = r² + y(1-2r); drop the constant r².
                term_count = 0
                for p, i in enumerate(active):
                    coeff = 1.0 - 2.0 * residual_round[p]
                    term_count = _write_term(f, f" {_fmt(coeff)} y[{i}]", term_count)
                f.write("\n")

                # ---------------------------------------------------------------------------
                # Constraints
                # ---------------------------------------------------------------------------
                
                f.write("Subject To\n")

                # Maps each global index to its position in the `active` list (and therefore in
                # `x_floor`, which is aligned to `active`). x_floor runs from 0 to len(active)-1,
                # so x_floor[0] holds the floor value for active[0], x_floor[1] for active[1], etc.
                # SparseConstraints store global indices (e.g. active[0] = 7), so to recover the
                # corresponding floor value we need the reverse mapping: pos_of[7] -> 0,
                # which gives x_floor[0] directly, without an O(n) search through `active`.
                pos_of = {g: p for p, g in enumerate(active)}

                for i, sc in enumerate(constraints):

                    # The rounding decision variable is only the binary correction y[i]. 
                    # The actual cell value is floor[i] + y[i]. 
                    # <=> sum(coef_i * (floor_i + y_i)) sense rhs
                    # <=> sum(coef_i * y_i) sense (rhs - sum(coef_i * floor_i))
                    
                    f.write(f" Constraint_{i}:")

                    tcount = 0
                    floor_contrib = 0.0
                    for idx, coef in zip(sc.indices, sc.coefs):
                        idx = int(idx)
                        floor_contrib += float(coef) * x_floor[pos_of[idx]] # coef_i * floor_i
                        tcount = _write_term(f, f" {_fmt(float(coef))} y[{idx}]", tcount) # coef_i * y_i
                    
                    # sense (rhs - sum(coef_i * floor_i))
                    adjusted_rhs = float(sc.rhs) - floor_contrib
                    f.write(f" {sc.sense} {adjusted_rhs:.17g}\n")

                # ---------------------------------------------------------------------------
                # Binaries
                # ---------------------------------------------------------------------------

                # No Bounds section needed: LP format's default bounds for variables listed
                # under `Binaries` are ignored/overridden to [0, 1] integer regardless.
                f.write("Binaries\n")
                for i in active:
                    f.write(f" y[{i}]\n")

                # ---------------------------------------------------------------------------
                # End
                # ---------------------------------------------------------------------------

                f.write("End\n")

            model = gp.read(tmp_path, env=self.env)
            model.optimize()
            y_values = {}

            # TIME_LIMIT with an incumbent is a usable answer, not a failure. Every hard
            # constraint holds exactly in any incumbent, so a truncated solution is feasible.
            if (model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL
                    or (model.status == GRB.TIME_LIMIT and model.SolCount > 0)):
                y_values = {i: model.getVarByName(f"y[{i}]").X for i in active}
                model.dispose()

            elif model.status == GRB.INFEASIBLE:
                debug_path = os.path.join(self._tmp_dir, f"infeasible_model_node_{node_id}.lp")
                model.write(debug_path)
                model.dispose()
                raise ValueError(
                    f"Model is infeasible for node {node_id}. See {debug_path}for debugging. "
                )
            else:
                status = model.status
                model.dispose()
                if status == GRB.TIME_LIMIT:
                    raise RuntimeError(
                        f"The rounding MIP for node {node_id} hit the TimeLimit in "
                        f"solver_options ({self.solver_options.get('TimeLimit')} s) with no "
                        f"incumbent at all, so there is nothing to return - an incumbent would "
                        f"have been accepted. On these models the root relaxation is usually "
                        f"what did not finish, not the search. Raise TimeLimit, set a reachable "
                        f"MIPGap, or use the junction-tree sweep."
                    )
                raise RuntimeError(f"Solver failed for node {node_id}. Status: {status}")

            # Reconstruct as a sparse column vector, keeping only positive cells.
            # x_floor is positional (aligned to active); i is the global index for the CSC row.
            rows = []
            data = []
            for p, i in enumerate(active):
                val = int(x_floor[p] + round(y_values[i]))
                if val > 0:
                    rows.append(i)
                    data.append(val)

            return sp.csc_matrix(
                (data, (rows, np.zeros(len(rows), dtype=self._solution_type))),
                shape=(n, 1),
                dtype=self._solution_type,
            )
        
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)