import gurobipy as gp
import numpy as np

from typing import List, Callable

class OptimizationModel:
    '''Represents an optimization model for a specific optimization problem of the geographic Tree using Gurobi.'''

    def __init__(self):
        '''Constructor for the OptimizationModel class.
        
        Attributes:
            model (gp.Model): Gurobi model instance.
        '''
        self.model = None

    def non_negative_real_estimation(self, contingency_vector: np.ndarray, id_node: int, constraints: List[Callable]) -> np.ndarray:
        '''Non-negative estimation of the contingency vector.
        
        This method creates a Gurobi model to estimate the contingency vector using non-negative constraints.

        Args:
            contingency_vector (np.ndarray): The contingency vector with noisy counts.
            id_node (int): The ID of the node for which the estimation is being performed.
            constraints (List[Callable], optional): List of additional constraints to apply to the model. Defaults to None.
        
        Returns:
            np.ndarray: Estimated contingency vector with non-negative real values.
        '''
        # Create a new Gurobi model
        self.model = gp.Model(f'NonNegativeRealEstimation. NodeID: {id_node}')
        self.model.setParam('OutputFlag', 0)  # Suppress Gurobi output

        # self.model.setParam('OptimalityTol', 1e-6)  # Approximate optimality tolerance (default: 1e-6, min: 1e-9, max: 1e-2)
        # self.model.setParam('BarConvTol', 1e-6) # Tolerance for barrier convergence (default 1e-8, min: 0.0, max: 1.0)
        # self.model.setParam('TimeLimit', 60)  # Stop after 60 seconds
        # self.model.setParam('Heuristics', .5)  # Allocate 50% of time to heuristics

        # Length of the contingency vector that we want to estimate
        n = len(contingency_vector)
        
        # Decision variable (vector of non-negative real values)
        x = self.model.addMVar(shape=n, lb=0.0, name="x")

        # Objective function: minimize the sum of squared differences (L2 norm)
        self.model.setObjective(gp.quicksum((x[i] - contingency_vector[i]) * (x[i] - contingency_vector[i]) for i in range(n)), gp.GRB.MINIMIZE)

        # Additional constraints provided by the user
        for i, constraint in enumerate(constraints):
            self.model.addConstr(constraint(x), name=f"GivenConstraint_{i}")

        # Run the model
        self.model.optimize()

        # Check for infeasibility
        if self.model.status == gp.GRB.INFEASIBLE:
            # Write the model to a file for debugging
            self.model.write("infeasible_model.lp")
            raise ValueError(f'Model is infeasible for node {id_node}. See infeasible_model.lp file for debugging.')
        
        return np.asarray(x.X)

    
    def rounding_estimation(self, x_tilde: np.ndarray, id_node: int, constraints: List[Callable]) -> np.ndarray:
        '''Rounding estimation of the contingency vector.
        
        This method creates a Gurobi model to estimate the non negative discrete contingency vector.

        Args:
            x_tilde (np.ndarray): The contingency vector with the solution of the previous optimization step.
            id_node (int): The ID of the node for which the estimation is being performed.
            constraints (List[Callable], optional): List of additional constraints to apply to the model. Defaults to None.
        
        Returns:
            np.ndarray: Estimated contingency vector with non-negative integer values.
        '''
        # Same logic as previous method
        self.model = gp.Model(f'RoundingEstimation. NodeID: {id_node}')
        self.model.setParam('OutputFlag', 0)
        n = len(x_tilde)
        
        # Rounding problem
        # We want to find which values of the vector x should be rounded up (1) or down (0)
        x_floor = np.floor(x_tilde)
        # We obtain the decimal part of each value
        residual_round = x_tilde - x_floor
        
        # Decision variable (binary vector indicating rounding up or down)
        y = self.model.addMVar(shape=n, vtype=gp.GRB.BINARY, name="y")

        # Objective function: minimize the sum of squared differences (L2 norm)
        self.model.setObjective(gp.quicksum((residual_round[i] - y[i]) * (residual_round[i] - y[i]) for i in range(n)), gp.GRB.MINIMIZE)

        # The rounded solution will be x_floor + y, we want constraints over this vector
        x_rounded = x_floor + y

        # Additional constraints
        for i, constraint in enumerate(constraints):
            self.model.addConstr(constraint(x_rounded), name=f"GivenConstraint_{i}")
        
        # Run the model
        self.model.optimize()

        # Check for infeasibility
        if self.model.status == gp.GRB.INFEASIBLE:
            self.model.write("infeasible_model.lp")
            raise ValueError(f'Model is infeasible for node {id_node}. See infeasible_model.lp file for debugging.')
        
        # Rounded solution
        return np.asarray(x_floor + y.X)

