# CensusDP (TopDown)
This repository explores the TopDown differentially-private microdata synthesis algorithm developed by the US Census Bureau in order to have a better understanding and to have a more versatile implementation that can be applied to other hierarchical datasets.

The algorithm aims to generate synthetic microdata that preserves the statistical properties of the original dataset while ensuring differential privacy through noise addition. Additionally, thorough constraint enforcement, it provides guarantees in the released data. 

One important aspect of this algorithm is that it considers hierarchical data structures, allowing it to maintain consistency across different levels of aggregation and apply differential privacy with different budgets at different levels.

The implementation reads raw microdata, creates a hierarchical tree, constructs hierarchical contingency vectors, applies discrete DP noise, solves constrained optimization problems to restore consistency, and reconstructs synthetic private microdata.

**Quick start**

- Clone & enter project: `git clone <repo>`
- Create a Python virtual environment and install dependencies.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

The main algorithm is implemented in the `TopDown` class located in `topdown.py` (see `main.py` as an example). A simple usage pattern is:

```python
from topdown import TopDown
from constraints.logical_expressions.atomic import GreaterThan
from constraints.aggregate_constraints import SumEqual

# Instantiate with path to raw microdata, hierarchy columns, query columns and optional output path
td = TopDown(data_path='data/microdata.csv', hierarchy=['REGION','STATE'], queries=['AGE','SEX'], out_path='noisy_microdata.csv')
# Note that the hierarchy does not include the root level (NATIONAL), which is implicit.

# Set privacy parameters per level (list length must match number of levels of the tree)
td.set_privacy_parameters([0.5, 0.3, 0.2])

# Choose mechanism: 'discrete_laplace' or 'discrete_gaussian'
td.set_mechanism('discrete_laplace')

# Add constraints (see Constraints section below)
my_constraint = SumEqual(GreaterThan('AGE', 99), 0)
td.set_constraint_to_tree(my_constraint)
# td.set_constraint_to_level(1, my_constraint)

# Run end-to-end
noisy_df = td.run()
```

Notes:
- The hierarchy is defined as a list of column names from the raw data, ordered from root to leaves. The root **should not** be in the list given to the algorithm. This is because it is implicitly defined as the aggregation of all data in the hierarchy.

- The tree can have an arbitrary number of children, in this case the tree could be:
  ```
        NATIONAL
       /        \
     REGION     REGION
     /   \       /   \
   STATE STATE STATE STATE
  ```
- Input file parsing uses `pandas.read_csv`; the default separator is `,`. In `TopDown.initialize()` the `DataHandler.read_data` call may use `sep=';'` depending on how you call it.
- The pipeline is: read -> generate contingency dataframe -> build hierarchical tree -> measurement (noise) -> estimation (optimization) -> microdata construction.

## Parameter configuration

- **Data path & columns**: Provided when instantiating `TopDown(data_path, hierarchy, queries, out_path)` or using `DataHandler` directly.
- **Privacy parameters**: Use `td.set_privacy_parameters([...])`. The list index corresponds to tree level (level 0 = root).
- **Noise mechanism**: Use `td.set_mechanism('discrete_laplace')` or `td.set_mechanism('discrete_gaussian')`. Under the hood these call `sample_dlaplace` / `sample_dgauss` from `discretegauss.py`.
- **Constraints**: Constraints are objects conforming to the `Constraint` interface (see `constraints/constraint.py`). Use `td.set_constraint_to_tree(constraint)` to add a constraint to all levels, or `td.set_constraint_to_level(level, constraint)` to apply to a particular level (applies to that level and higher levels during building). Constraints are evaluated and converted to callable constraint functions during tree construction in `DataHandler.build_hierarchical_tree()`.

## Constraints

Constraints are represented as objects in the `constraints` package. The API is:

- `Constraint` (base interface): implement `to_constraint(contingency_df)` which returns a callable used by the optimizer.
- `constraints.logical_expressions` contains logical expressions that can be combined to select subsets of the contingency domain:
  - Atomic expressions: `Equal`, `GreaterThan`, `LessThan`, `NotEqual`, etc.
  - Compound expressions: `And`, `Or`, `Not`, `Implies`.
  - `TrueExpression` and `FalseExpression` helpers.
- `constraints.aggregate_constraints` contains aggregate constraints (combine a `LogicalExpression` with a value):
  - `SumEqual(expression, value)` ensures the sum of counts where `expression` is True equals `value`.
- `constraints.contextual_constraints` provides contextual aggregate constraints whose `value` is calculated from the node's data:
  - `ContextualAggregateConstraint(expression, aggregation_function)` computes `value` dynamically for each node using `aggregation_function(node_dataframe)`.
  - `SumEqualRealTotal(expression)` is a convenience constraint that sets the aggregate value to the real number of rows in the node's context (i.e., enforces the real total for that node).

See the documentation in the `constraints/` folder for more details.

How constraints are applied:

- Build or create expression objects:

```python
from constraints.logical_expressions.atomic import Equal
from constraints.logical_expressions.compound import And
from constraints.aggregate_constraints import SumEqual
from constraints.contextual_constraints import SumEqualRealTotal

# Example: count of persons with SEX == 'M'
expr = Equal('SEX', 'M')
sum_eq = SumEqual(expr, 100)  # Sum(expr) == 100

# Example: enforce that the real total in each node equals reported total
real_total = SumEqualRealTotal(TrueExpression())

td.set_constraint_to_level(0, real_total)  # apply real total constraint to root (and effectively root-only)
```

- When `DataHandler.build_hierarchical_tree(constraints)` runs, it will:
  - Generate the global `contingency_df` (domain/order used to build vectors).
  - For each contextual constraint (subclass of `ContextualAggregateConstraint`), call `apply_aggregation_function(node_dataframe)` so the constraint value is computed for that node before converting to a callable via `to_constraint(contingency_df)`.
  - Append the resulting callable to each `HierarchicalNode.constraints` list; the optimizer consumes these callables when solving level-wise problems.

It is important that the constraints keep the optimization problem feasible. Adding incompatible constraints (or too many constraints) may make the optimization infeasible. In that case, the optimizer will raise an exception and write logs indicating the model that could not be solved.

## Initialization

`TopDown.initialize()` (or running `TopDown.run()`) will:

- Read the raw data using `DataHandler.read_data()`.
- Generate a global `contingency_df` that enumerates all possible query combinations using `DataHandler.generate_contingency_dataframe()`.
- Build the hierarchical tree with `DataHandler.build_hierarchical_tree(constraints)`, creating `HierarchicalNode` objects, assigning contingency vectors for each node, and converting configured constraint objects into callable constraints for the optimizer.

## Measurement phase

Implemented in `TopDown.measurement_phase()`. For each level the algorithm:

- Retrieves the privacy budget from `TopDown.privacy_parameters[level]`.
- Calls `TopDown.add_noise(node.contingency_vector, budget)` which applies the configured noise mechanism element-wise to the contingency vector.

Noise functions live in `discretegauss.py` (`sample_dgauss`, `sample_dlaplace`).

## Estimation phase

Implemented in `TopDown.estimation_phase()`. Key points:

- Root estimation uses `OptimizationModel.non_negative_real_estimation(...)` and `OptimizationModel.rounding_estimation(...)` from `optimizer.py`.
- For non-root levels the implementation builds joint vectors for children, converts each child's constraints to work over slices of the joint vector, adds parent-child consistency constraints (sum(children) == parent), and solves a joint optimization for all children.

Currently the optimzer uses Gurobi via `gurobipy`. If you don't have a Gurobi license feel free to modify `optimizer.py` to use another solver or implement a fallback method. Consider that the constraints will be given as callables that accept contingency vectors as numpy arrays.

## Microdata generation

`TopDown.construct_microdata()` and `DataHandler.construct_microdata(tree)` reconstruct synthetic microdata from leaf contingency vectors by replicating combinations according to counts in the contingency vector and adding hierarchical columns.

## Resume / checkpoints

TODO

## Validation and correctness

- `TopDown.check_correctness()` verify that each parent's contingency totals equal the sum of its children (basic consistency check).
- TODO: Add further validation (distance metrics or statistical checks) as needed for benchmarking privacy-utility trade-offs.

## Project layout

- `topdown.py`: main `TopDown` implementation and pipeline control.
- `data_handler.py`: reading data, generating contingency domain, building hierarchical tree, and microdata reconstruction.
- `hierarchical_node.py`, `hierarchical_tree.py` — hierarchical tree data structures.
- `optimizer.py`: optimization model wrappers (non-negative estimation, rounding).
- `discretegauss.py`: implementations of discrete Laplace / discrete Gaussian samplers.
- `constraints/`: constraint API (logical expressions, aggregate constraints, contextual constraints).
- `main.py`: optional script (if present) showing example runs.

