"""Per-node optimization for the TopDown estimation phase.

Writes an LP file to a per-process temp path and hands it to gurobipy.read, so
all model materialization happens in Gurobi's C parser rather than building a
Pyomo / gurobipy expression graph in Python.
"""
from __future__ import annotations

import io
import os
import tempfile
from typing import List

import numpy as np
import gurobipy as gp

from constraints.constraint import SparseRow


class OptimizationModel:
    """LP-file backed optimizer. One instance per worker process; reuses one Gurobi env."""

    def __init__(self, solver_name: str = 'gurobi', solver_options: dict | None = None,
                 optimizer_path: str | None = None) -> None:
        if solver_name != 'gurobi':
            raise ValueError(f"This OptimizationModel only supports 'gurobi'; got {solver_name!r}.")
        self.options = dict(solver_options or {})
        self.env = gp.Env(empty=True)
        for k, v in self.options.items():
            try:
                self.env.setParam(k, v)
            except gp.GurobiError:
                # Model-only parameter; will be re-applied after read.
                pass
        self.env.start()

    def non_negative_real_estimation(self, contingency_vector: np.ndarray, node_id: int,
                                     constraints: List[SparseRow]) -> np.ndarray:
        """L2 projection onto non-negative reals subject to sparse linear rows."""
        n = len(contingency_vector)
        path = _mkstemp_lp()
        try:
            _write_real_lp(path, contingency_vector, constraints, n)
            return self._solve(path, n, node_id, var_prefix='x')
        finally:
            _safe_unlink(path)

    def rounding_estimation(self, x_tilde: np.ndarray, node_id: int,
                            constraints: List[SparseRow]) -> np.ndarray:
        """Discrete rounding step: pick y in {0,1}^n that minimizes ||residual - y||^2 under
        the same sparse rows, then return floor(x_tilde) + y as integers."""
        n = len(x_tilde)
        x_floor = np.floor(x_tilde)
        residual = x_tilde - x_floor
        path = _mkstemp_lp()
        try:
            _write_round_lp(path, residual, x_floor, constraints, n)
            y = self._solve(path, n, node_id, var_prefix='y')
            return (x_floor + np.round(y)).astype(np.int64)
        finally:
            _safe_unlink(path)

    def _solve(self, lp_path: str, n: int, node_id: int, var_prefix: str) -> np.ndarray:
        model = gp.read(lp_path, env=self.env)
        for k, v in self.options.items():
            try:
                model.setParam(k, v)
            except gp.GurobiError:
                pass
        model.optimize()
        status = model.Status
        if status == gp.GRB.INFEASIBLE:
            keep = f'infeasible_model_node_{node_id}.lp'
            model.write(keep)
            raise ValueError(f"Model is infeasible for node {node_id}. See {keep} for debugging.")
        if status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
            raise RuntimeError(f"Solver finished with status {status} for node {node_id}.")
        return _read_solution(model, var_prefix, n)


# ----- LP-file generation -----------------------------------------------------

def _mkstemp_lp() -> str:
    fd, path = tempfile.mkstemp(suffix='.lp', prefix='topdown_')
    os.close(fd)
    return path


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _read_solution(model: gp.Model, var_prefix: str, n: int) -> np.ndarray:
    """Read variable values by name; one C call per .X via attribute access."""
    by_name = {v.VarName: v.X for v in model.getVars()}
    return np.fromiter((by_name[f'{var_prefix}_{i}'] for i in range(n)),
                       dtype=np.float64, count=n)


def _coef_term(coef: float, var: str) -> str:
    """Format ' + 3 x' / ' - x' / '' (zero suppressed). Sign-leading so terms compose freely."""
    if coef == 0.0:
        return ''
    if coef == 1.0:
        return f'+ {var}'
    if coef == -1.0:
        return f'- {var}'
    if coef > 0.0:
        return f'+ {coef:.17g} {var}'
    return f'- {-coef:.17g} {var}'


def _write_row(buf: io.StringIO, row: SparseRow, var_prefix: str, name: str) -> None:
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
    """min sum (x_i - c_i)^2 = sum x_i^2 - 2 sum c_i x_i (constant dropped).

    Gurobi LP quadratic syntax uses [ ... ] / 2, so for a coefficient 1 on x_i^2 the
    bracketed term is `2 x_i ^ 2`.
    """
    buf = io.StringIO()
    buf.write('Minimize\n obj:')

    # Linear part: -2 c_i x_i, omit zero-c_i terms.
    wrote_linear = False
    for i in range(n):
        ci = float(c[i])
        if ci == 0.0:
            continue
        term = _coef_term(-2.0 * ci, f'x_{i}')
        if term:
            buf.write(' ' + term)
            wrote_linear = True

    # Quadratic part: always emit one term per variable so every var is registered.
    quad_terms = [f'2 x_{i} ^ 2' for i in range(n)]
    quad_body = ' + '.join(quad_terms)
    if wrote_linear:
        buf.write(' + ')
    else:
        buf.write(' ')
    buf.write(f'[ {quad_body} ] / 2\n')

    buf.write('Subject To\n')
    for j, row in enumerate(rows):
        _write_row(buf, row, var_prefix='x', name=f'r_{j}')

    # Bounds: LP defaults are x >= 0 (continuous), which is exactly what we need.
    # Explicit bounds keep variables that were skipped from the linear part registered.
    buf.write('Bounds\n')
    for i in range(n):
        buf.write(f' x_{i} >= 0\n')

    buf.write('End\n')

    with open(path, 'w') as f:
        f.write(buf.getvalue())


def _write_round_lp(path: str, r: np.ndarray, x_floor: np.ndarray,
                    rows: List[SparseRow], n: int) -> None:
    """min sum (r_i - y_i)^2 with y_i in {0,1}.

    y_i^2 = y_i since binary, so the objective collapses to linear:
        sum_i (1 - 2 r_i) y_i + const
    and each row sum_k coef_k * x_{idx_k} <sense> rhs becomes
        sum_k coef_k * y_{idx_k} <sense> rhs - sum_k coef_k * x_floor[idx_k].
    """
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
        # Make sure there is at least one term so the file is well-formed.
        buf.write(' 0 y_0')
    buf.write('\n')

    buf.write('Subject To\n')
    for j, row in enumerate(rows):
        shift = float(np.dot(row.coefs, x_floor[row.indices]))
        shifted = SparseRow(indices=row.indices, coefs=row.coefs, sense=row.sense, rhs=row.rhs - shift)
        _write_row(buf, shifted, var_prefix='y', name=f'r_{j}')

    # Declare every variable as binary so all are registered, even those absent from obj/rows.
    buf.write('Binary\n')
    for i in range(n):
        buf.write(f' y_{i}\n')

    buf.write('End\n')

    with open(path, 'w') as f:
        f.write(buf.getvalue())
