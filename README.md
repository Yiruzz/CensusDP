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
from privacy import PureDP, ZCDP, ApproximateDP, RenyiDP
from constraints.logical_expressions.atomic import GreaterThan
from constraints.aggregate_constraints import SumEqual

# Build a privacy mechanism with one parameter per tree level (root + one per hierarchical column).
# Choose one of: PureDP (ε via discrete Laplace), ZCDP (ρ via discrete Gaussian),
# ApproximateDP (ρ + global δ, reports (ε, δ)-DP) or RenyiDP (ε + δ, joint-α calibration).
mechanism = PureDP([0.5, 0.3, 0.2])

# Instantiate with path to raw microdata, hierarchy columns, query columns and the mechanism.
td = TopDown(data_path='data/microdata.csv',
             hierarchy=['REGION', 'STATE'],
             query_columns=['AGE', 'SEX'],
             privacy_mechanism=mechanism,
             out_path='noisy_microdata.csv')
# Note that the hierarchy does not include the root level (NATIONAL). The root is implicitly defined as the aggregation of all data.

# Add constraints (see Constraints section below)
my_constraint = SumEqual(GreaterThan('AGE', 99), 0)
td.set_constraint_to_tree(my_constraint)
# Or set constraint to specific level:
# td.set_constraint_to_level(1, my_constraint)

# Run
td.run()
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

- **Data path & columns**: Provided when instantiating `TopDown(data_path, hierarchy, query_columns, privacy_mechanism, out_path)` or using `DataHandler` directly.
- **Privacy mechanism and budget**: Pass an instance of a `PrivacyMechanism` subclass to `TopDown(..., privacy_mechanism=...)`. The mechanism carries one budget parameter per tree level (index 0 = root, last = leaves), and its length must equal `len(hierarchy) + 1`. Available variants in `privacy.variants`:
  - `PureDP(epsilons)` — ε-DP per level via the discrete Laplace mechanism.
  - `ZCDP(rhos)` — ρ-zCDP per level via the discrete Gaussian mechanism.
  - `ApproximateDP(rhos, delta)` — ρ-zCDP under the hood; reports an equivalent (ε, δ)-DP guarantee via the Bun–Steinke inequality.
  - `RenyiDP(epsilons, delta)` — (ε, δ)-DP via joint-α Rényi-DP composition over the discrete Gaussian (the δ→ε conversion cost is paid once for the whole tree).
  Call `td.privacy_mechanism.report_guarantee()` after `td.run()` to print the resulting guarantee in human-readable form.
- **Query workload**: Use `td.set_query_workload(QueryWorkload()...)` (see `queries.py`) or pass a binary numpy matrix directly. The sensitivity Δ is computed once as twice the max column sum of the resulting binary `Q` (see "Privacy model" below).
- **Constraints**: Constraints are objects conforming to the `Constraint` interface (see `constraints/constraint.py`). Use `td.set_constraint_to_tree(constraint)` to add a constraint to all levels, or `td.set_constraint_to_level(level, constraint)` to apply to a particular level. Constraints are evaluated and converted to callable constraint functions during tree construction in `DataHandler.build_hierarchical_tree()`.

## Privacy model: bounded differential privacy

Two datasets are neighbours when they have the same number of records and differ in the values of exactly one of them. This is *bounded* differential privacy (Kifer and Machanavajjhala, *No Free Lunch in Data Privacy*, SIGMOD 2011), the definition the US Census Bureau adopted for its 2020 TopDown Algorithm because the total population is invariant (Abowd et al., *The 2020 Census Disclosure Avoidance System TopDown Algorithm*, Harvard Data Science Review, 2022). The alternative, *unbounded* DP, builds neighbours by adding or removing a record.

Replacing a record takes it out of one cell and puts it in another, so every sensitivity is twice the one for adding or removing a record:

| Measurement | Sensitivity Δ |
|---|---|
| Identity workload (every cell of the full joint) | 2 |
| Binary query workload `Q` | 2 × max column sum of `Q` (an upper bound) |
| Factored pipeline (one marginal per junction-tree bag) | 2 × number of bags |
| Private marginal selection (every 2-way marginal) | 2 × number of column pairs |

The doubling holds for the L1 and the squared L2 sensitivity alike, so one Δ calibrates every mechanism: `PureDP` adds discrete Laplace noise with scale Δ / ε, and `ZCDP` discrete Gaussian noise with σ = √(Δ / 2ρ). The factor is the `bounded_dp_factor` attribute of `TopDown`.

Since neighbours share their number of records, that number is public under this definition (Kifer et al., *Bayesian and Frequentist Semantics for Common Variations of Differential Privacy: Applications to the 2020 Census*, 2022, Definition 4.1), and enforcing it with `SumEqualRealTotal` at the root spends no budget.

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

td.set_constraint_to_level(0, real_total)  # root and level 1: level L covers levels 0 to L + 1
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

- Delegates to `self.privacy_mechanism.add_noise(noisy_vector, level, sensitivity)`, which selects the discrete-noise distribution (Laplace or Gaussian) and the scale derived from the per-level privacy parameter and the workload sensitivity.
- Distributes nodes across worker processes via `ProcessPoolExecutor` to parallelise the per-node noise draws.

The actual integer-noise samplers live in `privacy/noisy.py` (`sample_dgauss_optimized`, `sample_dlaplace_optimized`) and are thin, cached wrappers around the [OpenDP](https://docs.opendp.org/) `make_gaussian` / `make_geometric` mechanisms — caching avoids rebuilding an OpenDP mechanism object on every node.

## Estimation phase

Implemented in `TopDown.estimation_phase()`. Key points:

- Root estimation uses `OptimizationModel.non_negative_real_estimation(...)` and `OptimizationModel.rounding_estimation(...)` from `optimizer.py`.
- For non-root levels the implementation builds joint vectors for children, converts each child's constraints to work over slices of the joint vector, adds parent-child consistency constraints (sum(children) == parent), and solves a joint optimization for all children.

The optimizer uses Pyomo as an interface over optimization solvers (defaulting to Gurobi). Pyomo provides AbstractModels for efficient reuse of the same problem structure across multiple solves. If you don't have a Gurobi license, you can modify the solver in `OptimizationModel.__init__()` to use another supported solver. The constraints are given as callables that accept contingency vectors as numpy arrays or Pyomo expressions.

## Microdata generation

`TopDown.construct_microdata()` and `DataHandler.construct_microdata(tree)` reconstruct synthetic microdata from leaf contingency vectors by replicating combinations according to counts in the contingency vector and adding hierarchical columns.

## Validation and correctness

- `TopDown.check_correctness()` verify that each parent's contingency totals equal the sum of its children (basic consistency check).
- TODO: Add further validation (distance metrics or statistical checks) as needed for benchmarking privacy-utility trade-offs.

## Project layout

- `topdown.py`: main `TopDown` implementation and pipeline control.
- `data_handler.py`: reading data, generating contingency domain, building hierarchical tree, and microdata reconstruction.
- `hierarchical_node.py`, `hierarchical_tree.py`: hierarchical tree data structures.
- `optimizer.py`: optimization model wrappers (non-negative estimation, rounding).
- `privacy/`: DP variants (`variants.py` — `PureDP`, `ZCDP`, `ApproximateDP`, `RenyiDP`) and the OpenDP-backed integer-noise samplers (`noisy.py`).
- `queries.py`: small DSL (`QueryWorkload`, `col`) for building the binary query matrix `Q`.
- `constraints/`: constraint API (logical expressions, aggregate constraints, contextual constraints).
- `parallel_utils.py`: process-pool helpers used during the estimation phase.
- `main.py`: example driver showing a full configuration on the 2017 Chilean Census data.

