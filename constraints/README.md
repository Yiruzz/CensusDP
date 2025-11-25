# Constraints

This package provides a small DSL to express logical and aggregate constraints over contingency vectors used by the TopDown algorithm.

## Overview
- Constraints are represented as Python objects (classes) and composed to express domain rules.
- There are two main families:
  - Logical Expressions (filter-like predicates) — classes live in `constraints/logical_expressions/` (e.g. `Equal`, `GreaterThan`, `And`, `Or`, `Implies`).
  - Aggregate constraints — classes in `constraints/aggregate_constraints.py` (e.g. `SumEqual`, `SumEqualRealTotal`). Additionally there are **Contextual (dynamic)** constraints can compute values from the node's data (see `ContextualAggregateConstraint`).

## Typical workflow
1. Build a logical expression (atomic and/or compound).
2. Wrap it into an aggregate constraint if you want counts/sums (e.g. `SumEqual`, `SumEqualRealTotal`). Use just the logical expression if you want to express logical relationships only. (e.g. Value a in column A implies Value b in column B)
3. Register the constraint on `TopDown` (before calling `run`) using `set_constraint_to_level` or `set_constraint_to_tree`.
4. `DataHandler.build_hierarchical_tree` will convert these constraint objects into callables usable by the optimizer:
   - For `ContextualAggregateConstraint`, `apply_aggregation_function(filtered_df)` is called to compute the numeric `value` for that node that will be consider in the optimization constraint.
   - Then `to_constraint(contingency_df)` is called to return a callable that accepts a contingency vector (the optimization variable) and returns a boolean expression that the optimizer can add as a constraint.

## Minimal example
```python
# Example (conceptual)
from constraints.logical_expressions.atomic import Equal, TrueExpression
from constraints.logical_expressions.compound import And, Implies, LessThan
from constraints.contextual_constraints import SumEqualRealTotal
from topdown import TopDown

# (age < 15) AND (gender == 'F')  => (children == None)
left_expr = And(LessThan('age', 15), Equal('gender', 'F'))
right_expr = Equal('children', None)
constraint = Implies(left_expr, right_expr)

# Create TopDown and register constraint before initialize
top = TopDown(data_path='data/microdata.csv', hierarchy=['region','state'], queries=['age','gender', 'children'])

# Set constraint to entire tree
top.set_constraint_to_tree(constraint)

# Aggregate Constraint. Sum of the data on each node must equal the real total for region or higher levels
aggregate_constraint = SumEqualRealTotal(expression=TrueExpression())
top.set_constraint_to_level(1, aggregate_constraint)

# Other variable initializations ...

# Run algorithm (measurement + estimation + reconstruction)
top.run()
```

Here the resulting data will satisfy that the population at the level of region and higher levels (e.g., country) will equal the real total population, and that for each node, if the age is less than 15 and gender is 'F', then the number of children will be None.

## Important notes
- The optimizer expects each node's `constraints` to contain *callable* constraints (functions that accept the contingency vector). The data handler currently converts objects into callables during tree construction.
- For contextual constraints (e.g. `SumEqualRealTotal`) the node-level numeric value must be computed with `apply_aggregation_function` (done by `DataHandler`), otherwise `value` will not be initialized correctly and may cause infeasibility.
- If the solver reports infeasible models, check `infeasible_model.lp`. These files are written by the optimizer when a model has no feasible solution. Check for debugging.
- `TrueExpression` can be used as a no-op expression that always evaluates to true. This is useful when you want to create aggregate constraints that apply to all records without filtering.

## Extending the DSL
- Add a new atomic or compound constraint class under `constraints/logical_expressions/` by subclassing `LogicalExpression` and implementing `reduce(contingency_df)`.
- Add new aggregates by extending `AggregateConstraint` and implementing `to_constraint(contingency_df)`.
- For dynamic per-node behavior, extend `ContextualAggregateConstraint` and implement an aggregation function.

## Where to look in the code
- `constraints/logical_expressions/` — atomic & compound logical building expressions
- `constraints/aggregate_constraints.py` — aggregate/sum-style constraints
- `constraints/contextual_constraints.py` — contextual aggregate constraints that compute node-specific values
- `data_handler.py` — tree construction and conversion of constraint objects to callables
- `topdown.py` — how users register constraints with `TopDown.set_constraint_to_level` / `set_constraint_to_tree`
- `optimizer.py` — how callables are used to add Gurobi constraints (see debugging/infeasibility handling)