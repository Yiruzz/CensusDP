import os
import tempfile
from typing import List, Optional, IO

import numpy as np
import scipy.sparse as sp
import gurobipy as gp

from constraints.constraint import SparseRow

class OptimizationModel:
    '''LP-based optimization model builder for census DP problems.

    Solves non-negative real estimation and rounding estimation problems
    by writing them to Gurobi LP format and solving with Gurobi.
    '''

    def __init__(self, solver_name: str = 'gurobi', solver_options: Optional[dict] = None) -> None:
        '''Initialize the optimization model.

        Args:
            solver_name (str): Solver to use (must be 'gurobi').
            solver_options (Optional[dict]): Parameters to pass to Gurobi (e.g., TimeLimit, MIPGap).
        '''
        if solver_name != 'gurobi':
            raise ValueError(f"This OptimizationModel only supports 'gurobi'; got {solver_name!r}.")

        # Apply solver options to the Gurobi environment
        self.options = dict(solver_options or {})
        self.env = gp.Env(empty=True)
        for k, v in self.options.items():
            try:
                self.env.setParam(k, v)
            except gp.GurobiError:
                pass  # parameter is model-only; re-applied per model below
        self.env.start()

    def non_negative_real_estimation(self, noisy_measurements: np.ndarray, node_id: int, constraints: List[SparseRow], query_matrix: sp.csr_matrix) -> np.ndarray:
        '''Estimate non-negative values minimizing squared error to measurements.

        Solves: minimize ||Q*x - y||^2 subject to user-defined constraints and x >= 0.

        Args:
            noisy_measurements (np.ndarray): Noisy measurement vector y (length n_queries * n_children).
            node_id (int): Node identifier for logging/debugging.
            constraints (List[SparseRow]): User-defined constraints on x variables.
            query_matrix (sp.csr_matrix): Query matrix Q (n_queries x n_cells).

        Returns:
            np.ndarray: Estimated values x (non-negative, minimizing squared error).
        '''
        num_queries, num_cells = query_matrix.shape
        num_children = len(noisy_measurements) // num_queries
        num_x_variables = num_children * num_cells

        # Create temporary LP file and write the model
        path = _create_temp_lp_file(suffix='.lp')
        _write_non_negative_real_estimation_lp(path, query_matrix, noisy_measurements, num_queries, num_children, num_cells, constraints)

        try:
            return self._read_solve(path, num_x_variables, node_id, var_prefix='x')
        finally:
            _safe_delete_file(path)

    def rounding_estimation(self, x_tilde: np.ndarray, node_id: int, constraints: List[SparseRow]) -> np.ndarray:
        '''Round a continuous solution to binary (0/1) values via quadratic optimization.

        Finds binary values y that minimize Euclidean distance to the rounded solution
        while respecting user-defined constraints.

        Mathematical Problem:
            minimize: sum (residual - y)² where y_i ∈ {0, 1}
            subject to: constraints adjusted for binary variables

        Objective Function Transformation (Quadratic → Linear):
            LP format only supports linear objectives, so the quadratic problem is
            mathematically transformed to an equivalent linear one:

            For binary y_i: y_i² = y_i (since y_i ∈ {0, 1})
            min: sum (1 - 2·residual_i)y_i + sum residual_i²

        Args:
            x_tilde (np.ndarray): Continuous solution to round (length n).
            node_id (int): Node identifier for logging and error messages.
            constraints (List[SparseRow]): Constraints on binary variables.

        Returns:
            np.ndarray: Binary solution x_final = floor(x_tilde) + y_optimal.
        '''
        num_variables = len(x_tilde)
        floor_values = np.floor(x_tilde)
        residual_values = x_tilde - floor_values
        linear_coefficients = 1.0 - 2.0 * residual_values

        path = _create_temp_lp_file(suffix='.lp')
        _write_binary_rounding_lp(path, linear_coefficients, floor_values, constraints, num_variables)

        try:
            return self._read_solve_binary(path, num_variables, node_id, floor_values, var_prefix='y')
        finally:
            _safe_delete_file(path)

    def _read_solve(self, path: str, num_x_variables: int, node_id: int, var_prefix: str) -> np.ndarray:
        '''Read LP file, apply options, solve, and extract solution.

        Args:
            path (str): Path to the LP file.
            num_x_variables (int): Number of x variables to extract from solution.
            node_id (int): Node identifier for logging/debugging.
            var_prefix (str): Prefix of variables to extract (e.g., 'x').

        Returns:
            np.ndarray: Solution values for the num_x_variables variables.

        Raises:
            ValueError: If model is infeasible.
            RuntimeError: If solver fails to find a solution.
        '''
        # Read the LP model from file
        model = gp.read(path, env=self.env)

        # Apply user-provided solver options (time limits, gaps, etc.)
        for param_name, param_value in self.options.items():
            try:
                model.setParam(param_name, param_value)
            except gp.GurobiError:
                pass

        # Solve the optimization problem
        model.optimize()

        # Validate solver status and raise on failure
        self._check_status(model, node_id, fail_path=f'infeasible_model_node_{node_id}.lp')
        # Extract solution: only read the main x variables, not auxiliary variables
        solution = _read_solution_by_name(model, var_prefix, num_x_variables)
        model.dispose()
        return solution

    def _read_solve_binary(self, path: str, num_variables: int, node_id: int, x_floor: np.ndarray,
                          var_prefix: str) -> np.ndarray:
        '''Read LP file with binary variables, solve, and return rounded result.

        Args:
            path (str): Path to the LP file.
            num_variables (int): Number of binary variables to extract.
            node_id (int): Node identifier for logging/debugging.
            x_floor (np.ndarray): Floor values to add to binary solution.
            var_prefix (str): Prefix of variables to extract (e.g., 'y').

        Returns:
            np.ndarray: Final solution as x_floor + y_binary.
        '''
        # Read the LP model from file
        model = gp.read(path, env=self.env)
        # Apply user-provided solver options
        for param_name, param_value in self.options.items():
            try:
                model.setParam(param_name, param_value)
            except gp.GurobiError:
                pass
        # Solve the binary optimization problem
        model.optimize()
        # Validate solver status and raise on failure
        self._check_status(model, node_id, fail_path=f'infeasible_model_node_{node_id}.lp')
        # Extract binary variables and round to integers
        binary_values = np.round(_read_solution_by_name(model, var_prefix, num_variables)).astype(np.int64)
        # Reconstruct full solution: x_final = floor(x_tilde) + y_binary
        final_solution = (x_floor + binary_values).astype(np.int64)
        model.dispose()
        return final_solution

    def _check_status(self, model: gp.Model, node_id: int, fail_path: str, model_dump: Optional[gp.Model] = None) -> None:
        '''Check solver status and raise exceptions on failure.

        Validates that the optimization completed successfully. If the model is
        infeasible, writes it to a file for inspection and raises ValueError.
        For other non-optimal statuses, raises RuntimeError.

        Args:
            model (gp.Model): The solved Gurobi model.
            node_id (int): Node identifier (for error messages).
            fail_path (str): Path where to write the infeasible model file.
            model_dump (Optional[gp.Model]): Alternative model to dump if infeasible
                                             (useful if original is already disposed).
        '''
        status = model.Status
        # Check for infeasibility (no feasible solution exists)
        if status == gp.GRB.INFEASIBLE:
            # Write the model interpreted for Gurobi to a file for debugging
            (model_dump or model).write(fail_path)
            raise ValueError(f"Model is infeasible for node {node_id}. See {fail_path}.")
        # Check that solver found at least a suboptimal solution
        if status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
            raise RuntimeError(f"Solver finished with status {status} for node {node_id}.")


def _read_solution_by_name(model: gp.Model, var_prefix: str, number_of_variables: int) -> np.ndarray:
    '''Extract solution values for variables with a specific name prefix.

    Reads the solution from a solved Gurobi model and returns values for all
    variables matching the naming pattern "{var_prefix}_{i}".

    Args:
        model (gp.Model): The solved Gurobi model.
        var_prefix (str): Prefix for variable names to extract (e.g., 'x', 'y', 'q').
        number_of_variables (int): Number of variables to extract (0 to number_of_variables-1).

    Returns:
        np.ndarray: Array of shape (number_of_variables,) with solution values as float64.
                   Element i contains the value of variable "{var_prefix}_{i}".
    '''
    # Build a dictionary mapping variable names to their solution values
    solution_by_name = {variable.VarName: variable.X for variable in model.getVars()}
    # Extract values in order for variables "{var_prefix}_0", "{var_prefix}_1", ..., "{var_prefix}_{number_of_variables-1}"
    return np.fromiter((solution_by_name[f'{var_prefix}_{var_idx}'] for var_idx in range(number_of_variables)), dtype=np.float64, count=number_of_variables)

def _create_temp_lp_file(suffix: str) -> str:
    '''Create a temporary file to write a mathematical model.

    Args:
        suffix (str): File extension (e.g., '.lp', '.mps').

    Returns:
        str: Path to the created temporary file.
    '''
    fd, path = tempfile.mkstemp(suffix=suffix, prefix='topdown_')
    os.close(fd)
    return path

def _safe_delete_file(path: str) -> None:
    '''Safely delete a file, ignoring errors.

    Deletes the file at the given path. If the file cannot be deleted
    (e.g., already deleted, permission error), the error is silently ignored.

    Args:
        path (str): Path to the file to delete.
    '''
    try:
        os.unlink(path)
    except OSError:
        pass

def _format_coefficient_term(coef: float, var: str) -> str:
    '''Format a coefficient-variable term for LP format.

    Returns a string representation with sign and coefficient. For coefficients
    other than 0, 1, or -1, uses 17 significant digits.

    Args:
        coef (float): The coefficient to format.
        var (str): The variable name (e.g., 'x_i' or 'q_i').

    Returns:
        str: Formatted term as "sign coef var" (e.g., "+ 2.5 x_0") or empty string if coef is 0.
    '''
    if coef == 0.0:
        return ''
    if coef == 1.0:
        return f'+ {var}'
    if coef == -1.0:
        return f'- {var}'
    if coef > 0.0:
        return f'+ {coef:.17g} {var}'
    return f'- {-coef:.17g} {var}'

def _write_constraints(file: IO, constraint: SparseRow, var_prefix: str, name: str) -> None:
    '''Write a constraint to LP format.

    Formats and writes a single constraint from a SparseRow to the LP file.

    Args:
        file (IO): Open file object to write to.
        constraint (SparseRow): Constraint with indices, coefs, sense, and rhs attributes.
        var_prefix (str): Prefix for variable names (e.g., 'x', 'y').
        name (str): Name/label of the constraint.
    '''
    constraint_parts = [f' {name}:']
    is_first_term = True
    # Build left side: sum of coefficient*variable terms
    for variable_idx, coefficient in zip(constraint.indices.tolist(), constraint.coefs.tolist()):
        formatted_term = _format_coefficient_term(float(coefficient), f'{var_prefix}_{int(variable_idx)}')
        if not formatted_term:
            continue
        # Omit leading '+' for the first term
        if is_first_term:
            constraint_parts.append(formatted_term[2:] if formatted_term.startswith('+ ') else formatted_term)
            is_first_term = False
        else:
            constraint_parts.append(formatted_term)
    # Edge case: constraint with no terms
    if is_first_term:
        constraint_parts.append('0')

    # Add constraint sense and right-hand side
    constraint_parts.append(constraint.sense)
    constraint_parts.append(f'{float(constraint.rhs):.17g}')

    # Write the complete constraint
    file.write(' '.join(constraint_parts) + '\n')

def _write_non_negative_real_estimation_lp(path: str, query_matrix: sp.csr_matrix, noisy_measurements: np.ndarray,
                                            num_queries: int, num_children: int, num_cells: int,
                                            constraints: List[SparseRow]) -> None:
    '''Write a non-negative real estimation problem to an LP format file.

    Constructs and writes a Gurobi LP model that minimizes ||Q*x - y||^2
    subject to user-defined constraints, where x variables are non-negative.

    Args:
        path (str): Path to the output LP file.
        query_matrix (sp.csr_matrix): Query matrix Q (num_queries x num_cells).
        noisy_measurements (np.ndarray): Noisy measurements y (num_queries * num_children).
        num_queries (int): Number of queries.
        num_children (int): Number of children (data subsets).
        num_cells (int): Number of cells per query.
        constraints (List[SparseRow]): User-defined constraints over x variables.
    '''
    # Total number of x variables: x_{0}, x_{1}, ..., x_{num_x_variables-1}
    num_x_variables = num_children * num_cells

    # ------------------------------------------------------------------ #
    # Objective                                                          #
    # ------------------------------------------------------------------ #

    # Precompute nonzero column indices for each query row directly from CSR structure
    # (identifies which cells have nonzero query coefficients per query)
    nonzero_indices_per_row = [query_matrix.indices[query_matrix.indptr[r]:query_matrix.indptr[r+1]]
                               for r in range(num_queries)]

    # Identify query rows with multiple nonzero coefficients
    # These rows need auxiliary variables to linearize the quadratic terms
    multi_rows = [r for r in range(num_queries) if len(nonzero_indices_per_row[r]) > 1]
    multi_row_set = set(multi_rows)

    # Precalculate coefficients and coefficient pairs for each row (one-time cost)
    row_coeffs = {}          # For single-element lookup: row_coeffs[r][cell_idx]
    row_coeffs_list = {}     # For iteration: row_coeffs_list[r] = [(coef, idx), ...]
    for r in range(num_queries):
        row_start, row_end = query_matrix.indptr[r], query_matrix.indptr[r+1]
        nz_indices = nonzero_indices_per_row[r]
        coeffs = query_matrix.data[row_start:row_end]
        row_coeffs[r] = dict(zip(nz_indices, coeffs))
        row_coeffs_list[r] = list(zip(coeffs, nz_indices))

    # Open file in write mode to construct the LP model
    with open(path, 'w') as f:
        # Write LP format header: "Minimize" with objective function label "obj"
        line = 'Minimize\n obj:'
        f.write(line)

        # Accumulate objective function terms (linear and quadratic)
        # These will be written in LP format later
        linear_terms: list[tuple[float, str]] = []
        quad_terms:   list[str]               = []

        # Iterate over all (child, query) pairs to build error terms
        for child_idx in range(num_children):
            for query_idx in range(num_queries):
                # Extract the measurement error for this (child, query) pair
                noisy_measurement = float(noisy_measurements[child_idx * num_queries + query_idx])
                # Get nonzero column indices for this query row
                nonzero_indices = nonzero_indices_per_row[query_idx]

                # Case 1: Single nonzero coefficient in this row
                # No auxiliary variable needed; expand (coef*x_j - measurement)^2 directly
                if query_idx not in multi_row_set:
                    # Get the single nonzero column index and its coefficient
                    cell_idx = int(nonzero_indices[0])
                    coefficient = float(row_coeffs[query_idx][cell_idx])
                    # Form variable name using child and cell indices
                    x_variable = f'x_{child_idx * num_cells + cell_idx}'

                    # Expand (coef*x_j - measurement)^2 = coef^2*x_j^2 - 2*coef*measurement*x_j + measurement^2
                    # Write quadratic term as 2*coef^2*x_j^2 (LP format divides by 2)
                    quadratic_coefficient = 2.0 * coefficient * coefficient
                    quad_terms.append(f'{quadratic_coefficient:.17g} {x_variable} ^ 2')

                    # Add linear term coefficient: -2*coef*measurement
                    linear_coefficient = -2.0 * coefficient * noisy_measurement
                    if linear_coefficient != 0.0:
                        linear_terms.append((linear_coefficient, x_variable))
                    # The constant term measurement^2 is omitted (doesn't affect optimization)

                # Case 2: Multiple nonzero coefficients in this row
                # Use auxiliary variable to handle the quadratic form
                else:
                    # Create auxiliary variable: q_{child_idx}_{query_idx}
                    # This variable will be defined by a constraint (added later)
                    # Constraint: q_{child_idx}_{query_idx} = coefâ*xâ + coefâ*xâ + ...
                    auxiliary_variable = f'q_{child_idx}_{query_idx}'

                    # Expand (q - measurement)Â² = qÂ² - 2*measurement*q + measurementÂ²
                    # Write quadratic term: 2*q^2 (LP format divides by 2)
                    quad_terms.append(f'2 {auxiliary_variable} ^ 2')

                    # Add linear term coefficient: -2*measurement*q
                    linear_coefficient = -2.0 * noisy_measurement
                    if linear_coefficient != 0.0:
                        linear_terms.append((linear_coefficient, auxiliary_variable))
                    # The constant term measurement^2 is omitted (doesn't affect optimization)

        # Write linear terms to the objective function
        # For the first term, omit the leading '+' sign per LP format convention
        is_first_term = True
        for coefficient_value, variable_name in linear_terms:
            formatted_term = _format_coefficient_term(coefficient_value, variable_name)
            if not formatted_term:
                continue
            # Strip leading '+' from first term, keep signs for subsequent terms
            if is_first_term:
                line_text = ' ' + (formatted_term[2:] if formatted_term.startswith('+ ') else formatted_term)
                is_first_term = False
            else:
                line_text = ' ' + formatted_term
            f.write(line_text)

        # Write quadratic terms enclosed in [...]/2 per Gurobi LP format
        # Gurobi requires quadratic terms inside brackets divided by 2
        if quad_terms:
            # Add opening bracket with '+' if we have linear terms, else just '['
            opening_bracket = ' + [ ' if not is_first_term else ' [ '
            f.write(opening_bracket)
            # Join all quadratic terms with '+' operator
            quadratic_terms_str = ' + '.join(quad_terms)
            f.write(quadratic_terms_str)
            # Close bracket with division by 2
            closing_bracket = ' ] / 2'
            f.write(closing_bracket)
        elif is_first_term:
            # Edge case: no linear or quadratic terms, write zero objective
            f.write(' 0')

        # Complete the objective function section with newline
        f.write('\n')

        # ------------------------------------------------------------------ #
        # Constraints                                                        #
        # ------------------------------------------------------------------ #
        # Write "Subject To" to start the constraints section
        f.write('Subject To\n')

        # Auxiliary constraints linking q variables to their definitions
        # These constraints ensure q_{child_idx}_{query_idx} equals the weighted sum it represents
        # Constraint form: sum_j Q[query_idx,j] * x_{child_idx*num_cells+j} - q_{child_idx}_{query_idx} = 0
        auxiliary_constraint_count = 0
        for query_idx in multi_rows:
            # Get coefficient pairs from precalculated list (already computed, O(1) lookup)
            coefficient_pairs = row_coeffs_list[query_idx]
            # Create one constraint for each (query, child) pair
            for child_idx in range(num_children):
                constraint_name = f'AuxLink_{child_idx}_{query_idx}'
                constraint_parts = [f' {constraint_name}:']
                is_first_constraint_term = True

                # Build the left side of the constraint: sum of query coefficients
                # Format each term using _coef_term for proper sign handling
                for cell_idx, coefficient in coefficient_pairs:
                    # Variable index: position of variable in child child_idx, cell cell_idx
                    x_variable = f'x_{child_idx * num_cells + int(cell_idx)}'
                    formatted_term = _format_coefficient_term(float(coefficient), x_variable)
                    if not formatted_term:
                        continue
                    # Strip leading '+' from first term
                    if is_first_constraint_term:
                        constraint_parts.append(formatted_term[2:] if formatted_term.startswith('+ ') else formatted_term)
                        is_first_constraint_term = False
                    else:
                        constraint_parts.append(formatted_term)

                # Complete the constraint: sum(...) - q_{child_idx}_{query_idx} = 0
                # This enforces the definition of the auxiliary variable
                constraint_parts.append(f'- q_{child_idx}_{query_idx}')
                constraint_parts.append('=')
                constraint_parts.append('0')
                # Write the complete constraint line
                f.write(' '.join(constraint_parts) + '\n')
                auxiliary_constraint_count += 1

        # User-defined constraints over x variables (provided by the problem)
        # These are problem-specific constraints (e.g., sum constraints, bounds)
        for j, constraint in enumerate(constraints):
            _write_constraints(f, constraint, var_prefix='x', name=f'r_{j}')

        # ------------------------------------------------------------------ #
        # Bounds                                                             #
        # ------------------------------------------------------------------ #
        # Write "Bounds" to start the variable bounds section
        f.write('Bounds\n')

        # Define bounds for x variables: must be non-negative (real-valued)
        # x_i >= 0 for all i (enforces non-negative estimates)
        for var_idx in range(num_x_variables):
            f.write(f' x_{var_idx} >= 0\n')

        # Define bounds for auxiliary q variables: unbounded (free)
        # q variables can be negative or positive to fit the linear forms
        for query_idx in multi_rows:
            for child_idx in range(num_children):
                f.write(f' q_{child_idx}_{query_idx} free\n')

        # Write "End" to mark the end of the LP file
        f.write('End\n')
    
def _write_binary_rounding_lp(path: str, linear_coefficients: np.ndarray, floor_values: np.ndarray, constraints: List[SparseRow], num_variables: int) -> None:
    '''Write a rounding/binary estimation problem to an LP format file.

    Creates a model for rounding a continuous solution x_tilde to binary.

    Args:
        path (str): Path to the output LP file.
        linear_coefficients (np.ndarray): Linear coefficients for objective (1.0 - 2.0 * residual).
        floor_values (np.ndarray): Floor values of continuous solution.
        constraints (List[SparseRow]): Original constraints adjusted for binary variables.
        num_variables (int): Number of binary variables.
    '''
    with open(path, 'w') as f:
        # ------------------------------------------------------------------ #
        # Objective                                                          #
        # ------------------------------------------------------------------ #
        line = 'Minimize\n obj:'
        f.write(line)

        # Write linear objective coefficients
        is_first_term = True
        for var_idx in range(num_variables):
            coefficient = float(linear_coefficients[var_idx])
            if coefficient == 0.0:
                continue
            formatted_term = _format_coefficient_term(coefficient, f'y_{var_idx}')
            if is_first_term:
                line_text = ' ' + (formatted_term[2:] if formatted_term.startswith('+ ') else formatted_term)
                f.write(line_text)
                is_first_term = False
            else:
                line_text = ' ' + formatted_term
                f.write(line_text)

        # Edge case: all coefficients are zero
        if is_first_term:
            line_text = ' 0 y_0'
            f.write(line_text)

        # Complete the objective section with newline
        f.write('\n')

        # ------------------------------------------------------------------ #
        # Constraints                                                        #
        # ------------------------------------------------------------------ #
        f.write('Subject To\n')

        # Adjust each constraint for the binary variables
        # Original constraint: sum coef[i] * x[i] sense rhs
        # becomes: sum coef[i] * (floor_values[i] + y[i]) sense rhs
        # which simplifies to: sum coef[i] * y[i] sense (rhs - sum coef[i] * floor_values[i])
        for constraint_idx, constraint_row in enumerate(constraints):
            # Calculate the shift: sum of coef[i] * floor_values[i]
            rhs_shift = float(np.dot(constraint_row.coefs, floor_values[constraint_row.indices]))
            # Create shifted constraint with adjusted RHS
            shifted_constraint = SparseRow(indices=constraint_row.indices, coefs=constraint_row.coefs,
                                          sense=constraint_row.sense, rhs=constraint_row.rhs - rhs_shift)
            _write_constraints(f, shifted_constraint, var_prefix='y', name=f'r_{constraint_idx}')

        # ------------------------------------------------------------------ #
        # BINARY                                                     #
        # ------------------------------------------------------------------ #
        # Declare all y variables as binary (0 or 1)
        # "Binary" keyword implicitly sets bounds: 0 <= y_i <= 1
        f.write('Binary\n')
        for var_idx in range(num_variables):
            f.write(f' y_{var_idx}\n')

        # Mark end of LP file
        f.write('End\n')