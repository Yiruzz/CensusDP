import pyomo.environ as pyo
import numpy as np
from typing import List, Callable, Any

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
        self.solver = pyo.SolverFactory(solver_name)
        self.solver_options = solver_options
        if optimizer_path is not None:
            self.solver.set_executable(optimizer_path)

    def _solve_pyomo_model(self, instance: pyo.ConcreteModel, id_node: int) -> Any:
        '''Auxiliar function to solve the instance of the model and handle infeasibility.

        Args:
            instance (ConcreteModel): The concrete model instance to solve.
            id_node (int): The id of the node being solved.

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
            filename = f"infeasible_model_node_{id_node}.nl"
            instance.write(filename)
            raise ValueError(f'Model is infeasible for node {id_node}. See {filename} file for debugging.')
        else:
            raise RuntimeError(f"Solver termination failed for node {id_node}. Status: {results.solver.status}, Condition: {results.solver.termination_condition}")

    def non_negative_real_estimation(self, noisy_measurements: np.ndarray, id_node: int, constraints: List[Callable], query_matrix: np.ndarray) -> np.ndarray:
        '''Non-negative estimation of the contingency vector using Pyomo ConcreteModel.

        Minimizes sum_k ||Q @ x_k - y_k||^2, where noisy_measurements is the concatenation
        of per-child measurement blocks y_k (each of length n_queries = Q.shape[0]) and the
        decision variable x is the concatenation of per-child cell-count blocks x_k
        (each of length n_cells = Q.shape[1]). For the identity workload, Q = np.eye(n_cells)
        is passed by TopDown.initialize(), so this path handles both cases uniformly.

        Constraints are always expressed in cell space (indices 0..n_cells-1 per child).

        Args:
            noisy_measurements (np.ndarray): Concatenated noisy query answers y = Q @ x + noise.
            id_node (int): The ID of the node for which the estimation is being performed.
            constraints (List[Callable]): List of additional constraints to apply to the model.
            query_matrix (np.ndarray): Query matrix Q of shape (n_queries, n_cells).

        Returns:
            np.ndarray: Estimated cell-count vector with non-negative real values,
                shape (n_children * n_cells,).
        '''
        n_queries, n_cells = query_matrix.shape
        n_children = len(noisy_measurements) // n_queries
        n = n_children * n_cells

        instance = pyo.ConcreteModel(name=f'RealEstimation_NodeID_{id_node}')
        instance.I = pyo.RangeSet(0, n - 1)
        instance.x = pyo.Var(instance.I, domain=pyo.NonNegativeReals)

        # Objective: sum_k || Q @ x_k - y_k ||^2
        # Q entries are Python floats (constants) so Pyomo builds a pure quadratic expression.
        def objective_rule(model):
            total = 0
            for k in range(n_children):
                for r in range(n_queries):
                    q_x_r = sum(float(query_matrix[r, j]) * model.x[k * n_cells + j] for j in range(n_cells) if query_matrix[r, j] != 0)
                    total += (q_x_r - float(noisy_measurements[k * n_queries + r])) ** 2
            return total

        instance.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Add constraints (always expressed in cell space — indices 0..n_cells-1 per child)
        instance.ConstraintList = pyo.ConstraintList()
        for i, constraint_func in enumerate(constraints):
            try:
                pyomo_expression = constraint_func(instance.x)
                # Skip trivially true constraints (e.g. from empty index sets)
                if isinstance(pyomo_expression, bool):
                    if not pyomo_expression:
                        raise ValueError(f"Constraint {i} is statically infeasible (evaluates to False).")
                    continue
                instance.ConstraintList.add(pyomo_expression)
            except Exception as e:
                print(f"Error adding constraint {i}: {e}. Ensure the constraint function accepts Pyomo's Var and returns a Pyomo expression.")
                raise e

        # Solve the model
        self._solve_pyomo_model(instance, id_node)

        # Extract results
        return np.array([pyo.value(instance.x[i]) for i in range(n)])

    def rounding_estimation(self, x_tilde: np.ndarray, id_node: int, constraints: List[Callable]) -> np.ndarray:
        '''Rounding estimation of the contingency vector using Pyomo ConcreteModel.

        Args:
            x_tilde (np.ndarray): The contingency vector from the previous optimization step.
            id_node (int): The ID of the node for which the estimation is being performed.
            constraints (List[Callable]): List of additional constraints to apply to the model.

        Returns:
            np.ndarray: Estimated contingency vector with non-negative integer values.
        '''
        n = len(x_tilde)
        x_floor = np.floor(x_tilde)
        residual_round = x_tilde - x_floor

        # Create a ConcreteModel directly
        instance = pyo.ConcreteModel(name=f'RoundingEstimation_NodeID_{id_node}')

        # Set of indices
        instance.I = pyo.RangeSet(0, n - 1)

        # Parameters: residual and floor values
        instance.r = pyo.Param(instance.I, initialize={i: residual_round[i] for i in range(n)})
        instance.f = pyo.Param(instance.I, initialize={i: x_floor[i] for i in range(n)})

        # Decision variable: binary
        instance.y = pyo.Var(instance.I, domain=pyo.Binary)

        # Objective: minimize L2 norm of residuals
        def objective_rule(model):
            return sum((model.r[i] - model.y[i])**2 for i in model.I)

        instance.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Define rounded vector expression for constraints
        x_rounded = {i: instance.f[i] + instance.y[i] for i in instance.I}

        # Add constraints
        instance.ConstraintList = pyo.ConstraintList()
        for i, constraint_func in enumerate(constraints):
            try:
                pyomo_expression = constraint_func(x_rounded)
                # Skip trivially true constraints (e.g. from empty index sets)
                if isinstance(pyomo_expression, bool):
                    if not pyomo_expression:
                        raise ValueError(f"Constraint {i} is statically infeasible (evaluates to False).")
                    continue
                instance.ConstraintList.add(pyomo_expression)
            except Exception as e:
                print(f"Error adding constraint {i}: {e}. Ensure the constraint function accepts Pyomo's expression dict and returns a Pyomo expression.")
                raise e

        # Solve the model
        self._solve_pyomo_model(instance, id_node)

        # Extract results
        y_estimated_array = np.array([pyo.value(instance.y[i]) for i in range(n)])

        # Final result: floor + binary decisions
        return x_floor + y_estimated_array

