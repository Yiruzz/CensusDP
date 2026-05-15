# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the algorithm

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --process_until COMUNA --queries P01 P02 P03A P03B
```

`main.py` accepts `--process_until <hierarchy column>`, `--queries <cols...>`, and `--user_constraints` (optional flag for the Implies edit constraint). Inside `main.py` there is still a hard-coded data path, solver name, and budget allocation; treat the file as the runnable example for one configuration rather than a generic CLI.

Requires a working Gurobi license. The optimizer builds the Gurobi model in-process by default (matrix back end, see Estimation phase below). On infeasibility the model is written to `infeasible_model_node_{id}.lp` in the working directory.

There is no test suite, linter, or build step configured. The benchmark for the estimation phase lives in `bench_estimation.py`; see `bench_results.json` for stored runs and use `--compare` to diff them.

## Architecture

End-to-end pipeline lives in `TopDown.run()` (`topdown.py`):

1. **Initialize** (`TopDown.initialize` → `DataHandler`)
   - `read_data` loads the CSV. Note: `initialize()` hardcodes `sep=';'`, not `,`, regardless of file extension.
   - `generate_contingency_dataframe` builds the global query domain as the Cartesian product of unique values **seen in the loaded data**, one column at a time. A domain value absent from the input CSV silently drops out of the domain and therefore out of every contingency vector — if a query column has a known fixed domain you want to preserve, ensure at least one row exhibits each value. This domain ordering is the canonical index used by every contingency vector in the tree.
   - `build_hierarchical_tree` recursively splits the dataframe by `hierarchical_columns` and creates a `HierarchicalNode` per unique value. Root is implicit (whole dataset), so `hierarchy` should not include a root column. The tree is then flattened into a numpy contingency-vector array placed in `multiprocessing.shared_memory` so workers can read/write nodes without IPC overhead.

2. **Measurement phase** (`TopDown.measurement_phase`)
   - For each level, adds discrete noise to every entry of every node's contingency vector using `privacy_parameters[level]`. Mechanism is `discrete_laplace` or `discrete_gaussian`, both from `noisy.py` (vectorized samplers).
   - Parallelized with a `ThreadPoolExecutor` chunked across `self.workers` (default 4).

3. **Estimation phase** (`TopDown.estimation_phase`)
   - Root is solved alone via `OptimizationModel.non_negative_real_estimation` (L2 projection onto non-negative reals) followed by `rounding_estimation` (binary rounding step).
   - Non-root levels are solved **per parent**, jointly over all children: the joint vector is `concatenate(child_vectors)`, each child's constraints are translated to the right slice via `SparseRow.offset`, and `sum(children) == parent[i]` consistency constraints are added per index. Scheduling is adaptive — `subtree_estimation_phase` submits child tasks to a `ProcessPoolExecutor` (spawn) as soon as the parent finishes, so per-level wall times overlap.
   - Constraints flow as `SparseRow(indices, coefs, sense, rhs)` objects, not callables. `parallel_utils.generate_constraints` builds the joint row set; `optimizer.OptimizationModel` consumes it.
   - **Optimizer back end** (`optimizer.py`): three back ends selectable by the `_backend` key in `solver_options` or the `TOPDOWN_OPTIMIZER_BACKEND` env var. Default is `matrix`.
     - `matrix` (default): builds the Gurobi model with `addMVar` + `setMObjective` + `addMConstr` from a `scipy.sparse.csr_matrix`. No file involved.
     - `lp`: writes a Gurobi LP file to `$TMPDIR/topdown_*.lp` per node and feeds it to `gurobipy.read`. Slower than matrix but the file is human-readable, so this is the right back end when debugging.
     - `mps`: writes an MPS file. Slower than `lp` here because the column-oriented writer is heavier in Python; kept for completeness.
   - The empirical comparison at COMUNA scale (P01 P02 P03A P03B, 3 runs each) is in `bench_results.json`:
     - `baseline-pyomo-nl`: 15.05 s (the previous Pyomo NL path; preserved on the `shared-memory` branch).
     - `lp-writer-python`: 3.69 s.
     - `mps-writer-python`: 4.32 s.
     - `matrix-direct`: 2.02 s.
   - Pyomo is no longer a runtime dependency of the estimation phase on this branch.

4. **Microdata reconstruction** (`DataHandler.construct_microdata`)
   - Walks leaves, repeats each row of `contingency_df` `count[index]` times, and attaches the hierarchical path. Output CSV is written to `out_path`.
   - The inner loop is `contingency_df.iterrows()` + per-column `np.repeat` per leaf. There is a `TODO` in the source about vectorizing it. If you are profiling wall time after the matrix optimizer landed, this is now a meaningful share of total runtime.

`TopDown.check_correctness()` verifies `sum(children) == parent` post-estimation. Run it after `run()` whenever you touch `estimation_phase`, the constraint-to-`SparseRow` plumbing, or `parallel_utils.generate_constraints` — it is the cheapest smoke test for both consistency-constraint regressions and slice/index bugs in the joint-children optimization.

### Optional `TopDown` features

`main.py` references two scaffolded helpers that aren't part of the core pipeline:

- `topdown.read_processed_data(path, sep=';')` — load a previously-run partial output to resume from a higher level instead of restarting at the root. Marked TODO in `README.md`; treat as work-in-progress.
- `topdown.set_distance_metric('manhattan' | 'euclidean' | 'cosine')` — analysis-only hook to compare noisy vs. original contingency vectors. Does not affect the algorithm; leave unset for production runs.

### Constraints DSL (`constraints/`)

User-facing constraints are **objects**. `DataHandler.build_hierarchical_tree` converts each one to a `SparseRow` (`indices`, `coefs`, `sense`, `rhs`) once per node and stores those rows on `HierarchicalNode.constraints`. The optimizer never sees Python callables.

- **Logical expressions** (`constraints/logical_expressions/`): `Equal`, `NotEqual`, `GreaterThan`, `LessThan`, `TrueExpression`, etc., composed with `And`, `Or`, `Not`, `Implies`. `reduce(contingency_df)` returns a boolean Series over the domain. A standalone logical expression added as a constraint forces the count of rows that violate it to be zero (encoded as `sum_{i in negated} x[i] == 0`).
- **Aggregate constraints** (`constraints/aggregate_constraints.py`, `constraints/contextual_constraints.py`): wrap a logical expression with an aggregate target. `SumEqual(expr, value)` is static; `ContextualAggregateConstraint` computes `value` per node via `apply_aggregation_function(node_df)` at tree-build time (e.g. `SumEqualRealTotal` snapshots the node's row count). New aggregates subclass `AggregateConstraint` / `ContextualAggregateConstraint` and implement `to_sparse_row(contingency_df)`.
- `SparseRow.offset(k)` shifts every index by `k`; the joint-children solver uses this to translate per-child rows into the concatenated variable space without rewriting any logic.

`TopDown.set_constraint_to_level(level, c)` applies `c` to that level **and every level above it** — this is the documented behavior in `README.md`, not a bug. Targeting a single level in isolation is not exposed. `set_constraint_to_tree(c)` applies to all levels.

Consistency constraints (`sum(children) == parent`) are added automatically by `parallel_utils.generate_constraints`; do not also add them as user constraints — that will overdetermine the model.

## Repo layout notes

- `pyomo/` is a full clone of the upstream Pyomo project (its own `.git`); kept for studying the NL writer and the Gurobi plugins. Not used at runtime — do not edit or import from it.
- `censo2024/` and `data/` hold large raw census CSVs and are gitignored — never commit data.
- `discretegauss.py` is third-party (US Census Bureau implementation of exact discrete Gaussian/Laplace samplers); avoid editing it.
- `noisy.py` exposes vectorized wrappers (`sample_dgauss_optimized`, `sample_dlaplace_optimized`) used by `TopDown.discrete_laplace` / `discrete_gaussian`. Numpy-shape API: `(privacy_parameter, samples)` → array of samples.
- `bench_estimation.py` requires `psutil` (not in `requirements.txt`); install with `.venv/bin/pip install psutil`. Use `--backend {lp|mps|matrix}` to override the optimizer back end for a single bench run.
