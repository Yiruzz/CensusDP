"""Per-node optimization for the TopDown estimation phase.

Three back ends are available, selected by:
- the `_backend` key inside `solver_options`, or
- the `TOPDOWN_OPTIMIZER_BACKEND` environment variable, or
- the default ('matrix') if neither is set.

Back ends:
- 'matrix' — build the Gurobi model in-process from a scipy.sparse CSR matrix
             via addMVar / addMConstr / setMObjective. No file written. Fastest
             at COMUNA scale (see bench_results.json).
- 'lp'     — write a Gurobi-flavour LP file to $TMPDIR and gp.read it. Useful
             when debugging because the file is human-readable.
- 'mps'    — write an MPS file and gp.read it. Slower than 'lp' here because
             the column-oriented writer is heavier in Python; kept for
             completeness.
"""
from __future__ import annotations

import io
import os
import tempfile
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import gurobipy as gp

from constraints.constraint import SparseRow


_VALID_BACKENDS = ('lp', 'mps', 'matrix')
_SENSE_LONG_TO_SHORT = {'=': '=', '<=': '<', '>=': '>'}


def _resolve_backend(opts: dict | None) -> str:
    if opts and '_backend' in opts:
        b = opts['_backend']
    else:
        b = os.environ.get('TOPDOWN_OPTIMIZER_BACKEND', 'matrix')
    if b not in _VALID_BACKENDS:
        raise ValueError(f"unknown backend {b!r}; expected one of {_VALID_BACKENDS}")
    return b


class OptimizationModel:
    """Per-worker optimizer. One instance per spawn process; reuses one Gurobi env."""

    def __init__(self, solver_name: str = 'gurobi', solver_options: dict | None = None,
                 optimizer_path: str | None = None) -> None:
        if solver_name != 'gurobi':
            raise ValueError(f"This OptimizationModel only supports 'gurobi'; got {solver_name!r}.")

        opts = dict(solver_options or {})
        self.backend = _resolve_backend(opts)
        opts.pop('_backend', None)  # do not forward our private key to Gurobi
        self.options = opts

        self.env = gp.Env(empty=True)
        for k, v in self.options.items():
            try:
                self.env.setParam(k, v)
            except gp.GurobiError:
                pass  # parameter is model-only; re-applied per model below
        self.env.start()

    # Public entry points ----------------------------------------------------

    def non_negative_real_estimation(self, contingency_vector: np.ndarray, node_id: int,
                                     constraints: List[SparseRow]) -> np.ndarray:
        if self.backend == 'lp':
            return self._real_via_file(contingency_vector, node_id, constraints, fmt='lp')
        if self.backend == 'mps':
            return self._real_via_file(contingency_vector, node_id, constraints, fmt='mps')
        return self._real_matrix(contingency_vector, node_id, constraints)

    def rounding_estimation(self, x_tilde: np.ndarray, node_id: int,
                            constraints: List[SparseRow]) -> np.ndarray:
        if self.backend == 'lp':
            return self._round_via_file(x_tilde, node_id, constraints, fmt='lp')
        if self.backend == 'mps':
            return self._round_via_file(x_tilde, node_id, constraints, fmt='mps')
        return self._round_matrix(x_tilde, node_id, constraints)

    # File-based path: shared scaffolding for lp / mps ------------------------

    def _real_via_file(self, c: np.ndarray, node_id: int,
                       constraints: List[SparseRow], fmt: str) -> np.ndarray:
        n = len(c)
        path = _mkstemp(suffix=f'.{fmt}')
        try:
            if fmt == 'lp':
                _write_real_lp(path, c, constraints, n)
            else:
                _write_real_mps(path, c, constraints, n)
            return self._read_solve(path, n, node_id, var_prefix='x')
        finally:
            _safe_unlink(path)

    def _round_via_file(self, x_tilde: np.ndarray, node_id: int,
                        constraints: List[SparseRow], fmt: str) -> np.ndarray:
        n = len(x_tilde)
        x_floor = np.floor(x_tilde)
        residual = x_tilde - x_floor
        path = _mkstemp(suffix=f'.{fmt}')
        try:
            if fmt == 'lp':
                _write_round_lp(path, residual, x_floor, constraints, n)
            else:
                _write_round_mps(path, residual, x_floor, constraints, n)
            y = self._read_solve(path, n, node_id, var_prefix='y')
            return (x_floor + np.round(y)).astype(np.int64)
        finally:
            _safe_unlink(path)

    def _read_solve(self, path: str, n: int, node_id: int, var_prefix: str) -> np.ndarray:
        model = gp.read(path, env=self.env)
        for k, v in self.options.items():
            try:
                model.setParam(k, v)
            except gp.GurobiError:
                pass
        model.optimize()
        _check_status(model, node_id, fail_path=f'infeasible_model_node_{node_id}.lp')
        return _read_solution_by_name(model, var_prefix, n)

    # Matrix path -------------------------------------------------------------

    def _real_matrix(self, c: np.ndarray, node_id: int,
                     constraints: List[SparseRow]) -> np.ndarray:
        n = len(c)
        m = gp.Model(env=self.env)
        _apply_model_options(m, self.options)
        x = m.addMVar(shape=n, lb=0.0, name='x')

        # Objective: min sum (x_i - c_i)^2 = x'Ix - 2 c'x + const  (const dropped).
        # setMObjective: xQ_L' * Q * xQ_R + c' * xc + constant.
        Q = sp.eye(n, format='csr')
        c_lin = -2.0 * np.asarray(c, dtype=np.float64)
        m.setMObjective(Q, c_lin, 0.0, x, x, x, gp.GRB.MINIMIZE)

        if constraints:
            A, sense_arr, rhs = _rows_to_csr(constraints, n)
            m.addMConstr(A, x, sense_arr, rhs)

        m.optimize()
        _check_status(m, node_id, fail_path=f'infeasible_model_node_{node_id}.lp', model_dump=m)
        return x.X.copy()

    def _round_matrix(self, x_tilde: np.ndarray, node_id: int,
                      constraints: List[SparseRow]) -> np.ndarray:
        n = len(x_tilde)
        x_floor = np.floor(x_tilde)
        residual = x_tilde - x_floor

        m = gp.Model(env=self.env)
        _apply_model_options(m, self.options)
        y = m.addMVar(shape=n, vtype=gp.GRB.BINARY, name='y')

        # y_i in {0,1} => y_i^2 = y_i. So sum (r_i - y_i)^2 collapses to
        # sum (1 - 2 r_i) y_i + const.
        c_lin = 1.0 - 2.0 * residual
        m.setMObjective(None, c_lin, 0.0, None, None, y, gp.GRB.MINIMIZE)

        if constraints:
            A, sense_arr, rhs_orig = _rows_to_csr(constraints, n)
            shifts = np.fromiter(
                (float(np.dot(row.coefs, x_floor[row.indices])) for row in constraints),
                dtype=np.float64,
                count=len(constraints),
            )
            m.addMConstr(A, y, sense_arr, rhs_orig - shifts)

        m.optimize()
        _check_status(m, node_id, fail_path=f'infeasible_model_node_{node_id}.lp', model_dump=m)
        return (x_floor + np.round(y.X)).astype(np.int64)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _mkstemp(suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix='topdown_')
    os.close(fd)
    return path


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _read_solution_by_name(model: gp.Model, var_prefix: str, n: int) -> np.ndarray:
    by_name = {v.VarName: v.X for v in model.getVars()}
    return np.fromiter((by_name[f'{var_prefix}_{i}'] for i in range(n)),
                       dtype=np.float64, count=n)


def _check_status(model: gp.Model, node_id: int, fail_path: str,
                  model_dump: gp.Model | None = None) -> None:
    status = model.Status
    if status == gp.GRB.INFEASIBLE:
        (model_dump or model).write(fail_path)
        raise ValueError(f"Model is infeasible for node {node_id}. See {fail_path}.")
    if status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
        raise RuntimeError(f"Solver finished with status {status} for node {node_id}.")


def _apply_model_options(model: gp.Model, opts: dict) -> None:
    for k, v in opts.items():
        try:
            model.setParam(k, v)
        except gp.GurobiError:
            pass


# ---------------------------------------------------------------------------
# LP writer
# ---------------------------------------------------------------------------

def _coef_term(coef: float, var: str) -> str:
    if coef == 0.0:
        return ''
    if coef == 1.0:
        return f'+ {var}'
    if coef == -1.0:
        return f'- {var}'
    if coef > 0.0:
        return f'+ {coef:.17g} {var}'
    return f'- {-coef:.17g} {var}'


def _write_row_lp(buf: io.StringIO, row: SparseRow, var_prefix: str, name: str) -> None:
    parts = [f' {name}:']
    first = True
    for idx, coef in zip(row.indices.tolist(), row.coefs.tolist()):
        term = _coef_term(float(coef), f'{var_prefix}_{int(idx)}')
        if not term:
            continue
        if first:
            parts.append(term[2:] if term.startswith('+ ') else term)
            first = False
        else:
            parts.append(term)
    if first:
        parts.append('0')
    parts.append(row.sense)
    parts.append(f'{float(row.rhs):.17g}')
    buf.write(' '.join(parts) + '\n')


def _write_real_lp(path: str, c: np.ndarray, rows: List[SparseRow], n: int) -> None:
    buf = io.StringIO()
    buf.write('Minimize\n obj:')

    wrote_linear = False
    for i in range(n):
        ci = float(c[i])
        if ci == 0.0:
            continue
        term = _coef_term(-2.0 * ci, f'x_{i}')
        if term:
            buf.write(' ' + term)
            wrote_linear = True

    quad_body = ' + '.join(f'2 x_{i} ^ 2' for i in range(n))
    buf.write(' + ' if wrote_linear else ' ')
    buf.write(f'[ {quad_body} ] / 2\n')

    buf.write('Subject To\n')
    for j, row in enumerate(rows):
        _write_row_lp(buf, row, var_prefix='x', name=f'r_{j}')

    buf.write('Bounds\n')
    for i in range(n):
        buf.write(f' x_{i} >= 0\n')
    buf.write('End\n')

    with open(path, 'w') as f:
        f.write(buf.getvalue())


def _write_round_lp(path: str, r: np.ndarray, x_floor: np.ndarray,
                    rows: List[SparseRow], n: int) -> None:
    buf = io.StringIO()
    buf.write('Minimize\n obj:')

    first = True
    for i in range(n):
        coef = 1.0 - 2.0 * float(r[i])
        if coef == 0.0:
            continue
        term = _coef_term(coef, f'y_{i}')
        if first:
            buf.write(' ' + (term[2:] if term.startswith('+ ') else term))
            first = False
        else:
            buf.write(' ' + term)
    if first:
        buf.write(' 0 y_0')
    buf.write('\n')

    buf.write('Subject To\n')
    for j, row in enumerate(rows):
        shift = float(np.dot(row.coefs, x_floor[row.indices]))
        shifted = SparseRow(indices=row.indices, coefs=row.coefs, sense=row.sense,
                            rhs=row.rhs - shift)
        _write_row_lp(buf, shifted, var_prefix='y', name=f'r_{j}')

    buf.write('Binary\n')
    for i in range(n):
        buf.write(f' y_{i}\n')
    buf.write('End\n')

    with open(path, 'w') as f:
        f.write(buf.getvalue())


# ---------------------------------------------------------------------------
# MPS writer
# ---------------------------------------------------------------------------

def _rows_to_columns(rows: List[SparseRow], n_vars: int) -> List[List[Tuple[int, float]]]:
    cols: List[List[Tuple[int, float]]] = [[] for _ in range(n_vars)]
    for ri, row in enumerate(rows):
        for idx, coef in zip(row.indices.tolist(), row.coefs.tolist()):
            c = float(coef)
            if c == 0.0:
                continue
            cols[int(idx)].append((ri, c))
    return cols


def _write_real_mps(path: str, c: np.ndarray, rows: List[SparseRow], n: int) -> None:
    """MPS file for: min sum (x_i - c_i)^2 s.t. sparse rows, x_i >= 0.

    Gurobi MPS quadratic uses QUADOBJ where entries form Q in `0.5 x'Q x + c'x`.
    For sum x_i^2 = x'Ix, we need diagonal Q_ii = 2."""
    buf = io.StringIO()
    buf.write('NAME          topdown_real\n')
    buf.write('ROWS\n')
    buf.write(' N  obj\n')
    sense_short = {'=': 'E', '<=': 'L', '>=': 'G'}
    for j, row in enumerate(rows):
        buf.write(f' {sense_short[row.sense]}  r_{j}\n')

    cols = _rows_to_columns(rows, n)
    buf.write('COLUMNS\n')
    for i in range(n):
        ci = float(c[i])
        entries = []
        if ci != 0.0:
            entries.append(('obj', -2.0 * ci))
        entries.extend((f'r_{ri}', coef) for ri, coef in cols[i])
        if not entries:
            # Variable only appears in QUADOBJ; emit a zero entry to register it explicitly.
            entries.append(('obj', 0.0))
        for row_name, coef in entries:
            buf.write(f'    x_{i}  {row_name}  {coef:.17g}\n')

    buf.write('RHS\n')
    for j, row in enumerate(rows):
        buf.write(f'    rhs  r_{j}  {float(row.rhs):.17g}\n')

    buf.write('BOUNDS\n')  # defaults to lb=0, ub=+inf; section can be empty
    buf.write('QUADOBJ\n')
    for i in range(n):
        buf.write(f'    x_{i}  x_{i}  2.0\n')
    buf.write('ENDATA\n')

    with open(path, 'w') as f:
        f.write(buf.getvalue())


def _write_round_mps(path: str, r: np.ndarray, x_floor: np.ndarray,
                     rows: List[SparseRow], n: int) -> None:
    buf = io.StringIO()
    buf.write('NAME          topdown_round\n')
    buf.write('ROWS\n')
    buf.write(' N  obj\n')
    sense_short = {'=': 'E', '<=': 'L', '>=': 'G'}
    # Pre-compute shifted rhs per row (linear-in-y form).
    shifts = np.fromiter(
        (float(np.dot(row.coefs, x_floor[row.indices])) for row in rows),
        dtype=np.float64,
        count=len(rows),
    )
    for j, row in enumerate(rows):
        buf.write(f' {sense_short[row.sense]}  r_{j}\n')

    cols = _rows_to_columns(rows, n)
    buf.write('COLUMNS\n')
    buf.write("    MARKER  'MARKER'  'INTORG'\n")  # everything is integer/binary
    for i in range(n):
        coef_obj = 1.0 - 2.0 * float(r[i])
        entries = []
        if coef_obj != 0.0:
            entries.append(('obj', coef_obj))
        entries.extend((f'r_{ri}', coef) for ri, coef in cols[i])
        if not entries:
            entries.append(('obj', 0.0))
        for row_name, coef in entries:
            buf.write(f'    y_{i}  {row_name}  {coef:.17g}\n')
    buf.write("    MARKER  'MARKER'  'INTEND'\n")

    buf.write('RHS\n')
    for j, row in enumerate(rows):
        buf.write(f'    rhs  r_{j}  {float(row.rhs - shifts[j]):.17g}\n')

    buf.write('BOUNDS\n')
    for i in range(n):
        buf.write(f' BV bnd  y_{i}\n')
    buf.write('ENDATA\n')

    with open(path, 'w') as f:
        f.write(buf.getvalue())


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def _rows_to_csr(rows: List[SparseRow], n_vars: int):
    """Build a CSR matrix + per-row sense array + rhs vector in one pass."""
    m = len(rows)
    if m == 0:
        return sp.csr_matrix((0, n_vars), dtype=np.float64), np.empty(0, dtype='U1'), np.empty(0, dtype=np.float64)

    nnz_per_row = np.fromiter((row.indices.shape[0] for row in rows), dtype=np.int64, count=m)
    total_nnz = int(nnz_per_row.sum())

    data = np.empty(total_nnz, dtype=np.float64)
    indices = np.empty(total_nnz, dtype=np.int32)
    indptr = np.empty(m + 1, dtype=np.int32)
    indptr[0] = 0

    offset = 0
    sense = np.empty(m, dtype='U1')
    rhs = np.empty(m, dtype=np.float64)
    for i, row in enumerate(rows):
        k = int(nnz_per_row[i])
        data[offset:offset + k] = row.coefs
        indices[offset:offset + k] = row.indices
        offset += k
        indptr[i + 1] = offset
        sense[i] = _SENSE_LONG_TO_SHORT[row.sense]
        rhs[i] = float(row.rhs)

    A = sp.csr_matrix((data, indices, indptr), shape=(m, n_vars))
    return A, sense, rhs
