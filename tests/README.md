# Empirical evaluation framework

Parameterized experiment scripts that quantify CensusDP's runtime and utility behaviour
under realistic configurations. Outputs are publication-quality plots + raw CSVs + JSON
config snapshots for reproducibility.

## Prerequisites

1. **Dataset.** The shipped `tests/config.toml` declares two datasets pointing at the 2017 Chilean census CSVs. The repo's `.gitignore` excludes `data/`, so you supply the actual CSVs yourself — either at the configured paths, or register your own dataset (see *Configuration* below).

2. **Python environment.** From the project root:

   ```powershell
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

   Requires Python 3.11+ (stdlib `tomllib` is used to read the config).

3. **Solver.** The default is Gurobi (commercial license required). Free alternatives that handle both QP and the rounding MIP: GLPK, CBC, HiGHS — switch via `solver_name` in `tests/config.toml`. IPOPT is **not** compatible because it can't handle the binary variables in `rounding_estimation`.

4. **Outputs land in `tests/out/`,** which is gitignored — each experiment writes its own subdirectory (e.g. `tests/out/exp1_runtime/`). Inspect them locally; nothing in this folder is shipped with the repo.

5. **Discoverability.** Every script has `--help`. Start with `python -m tests.exp1_runtime --help` for the common-flag reference.

## Configuration

The framework reads `tests/config.toml` at import time. Two top-level tables:

* **`[defaults]`** — values used when a CLI flag is omitted (dataset, process_until, mechanism, delta, allocation, trials, extension, solver_name, solver_options).
* **`[defaults.budgets]`** — per-mechanism default budget. PureDP / RenyiDP use ε natively; ZCDP / ApproximateDP use ρ. Pick values so each row encodes the **same** (ε, δ)-DP guarantee — the shipped config has all four at iso-(ε=10, δ=1e-10).
* **`[datasets.NAME]`** — one table per dataset, with `path`, `hierarchy`, `sep`, and `default_queries`.

**Precedence at runtime:** CLI flag > value in the config file. There are no built-in fallbacks for required fields; if the config is missing or malformed the framework refuses to start with a message that names the config file path.

### Registering a new dataset

Open `tests/config.toml` and append a table. Example for the US ACS public-use microdata:

```toml
[datasets.us_acs]
path             = "data/acs/microdata.csv"
hierarchy        = ["STATE", "COUNTY", "TRACT"]
sep              = ","
default_queries  = ["AGE_BAND", "SEX"]
```

Then run any experiment against it:

```powershell
python -m tests.exp2_utility_per_level --dataset us_acs --process_until COUNTY --queries AGE_BAND SEX
```

That's the only file you have to touch — no Python edit needed.

### Using a different config file

Pass `--config path/to/other.toml` to any script. Useful for sharing dataset-specific configs without mutating the repo's checked-in default:

```powershell
python -m tests.exp1_runtime --config configs/team_acs.toml --dataset us_acs --trials 3
```

### When something is wrong

Errors quote the active config file path so you know where to fix them:

* `dataset 'foo' not found in <path>/tests/config.toml (defined: ['personas', 'viviendas'])`
* `process_until='X' not in hierarchy [...] for dataset 'Y' (configured in <path>/tests/config.toml)`
* `<path>/tests/config.toml: [datasets.us_acs] is missing required key(s): ['sep']. Required: ['path', 'hierarchy', 'sep', 'default_queries'].`

## Layout

```
tests/
├── config.toml                  # Datasets registry + experiment defaults (edit this)
├── common.py                    # Shared utilities (config loader, truth pass, workloads, metrics, plots)
├── exp1_runtime.py              # Runtime vs query-vector size & tree depth
├── exp2_utility_per_level.py    # Per-level TVD / L1 / MAE vs truth
├── exp3_query_error.py          # Per-query percentage error vs tree level
├── exp4_budget_sweep.py         # Utility vs target ε at iso-(ε, δ), overlay mechanisms
├── exp5_query_matrix.py         # Compare query matrix designs (identity, k-marginals, random, mixed)
├── exp6_budget_allocation.py    # Compare budget allocation strategies across tree levels
├── exp7_mechanisms.py           # Compare PureDP / ZCDP / ApproximateDP / RenyiDP at iso-(ε, δ)
├── exp8_correlation.py          # Real vs private Cramér's V / Pearson heatmaps + diff
├── exp9_matrix_size.py          # Runtime vs n_queries (Q rows) at fixed query columns
├── _replot.py                   # Regenerate plots from existing CSVs without re-running
└── out/                         # All outputs (graphs, CSVs, JSON config snapshots)
    └── ...
```

Run any experiment from the project root:

```powershell
python -m tests.exp1_runtime [options]
python -m tests.exp2_utility_per_level [options]
# … etc.
```

## Common CLI flags

Every script accepts the flags below. Defaults shown are those in the shipped `tests/config.toml`; if you point `--config` at a different file, defaults shift accordingly.

| Flag             | Default        | Meaning                                                                          |
|------------------|----------------|----------------------------------------------------------------------------------|
| `--config`       | `tests/config.toml` | Path to a TOML config file (datasets + experiment defaults)                  |
| `--dataset`      | `viviendas`    | Name of a `[datasets.X]` table in `--config`                                     |
| `--process_until`| `COMUNA`       | Deepest hierarchy column to process; must appear in the selected dataset's `hierarchy` |
| `--queries`      | dataset-defaults | Query columns. Defaults to the dataset's `default_queries`                     |
| `--trials`       | `5`            | Independent trials per configuration                                             |
| `--mechanism`    | `ZCDP`         | `PureDP` / `ZCDP` / `ApproximateDP` / `RenyiDP`                                  |
| `--budget`       | per-mechanism (see `[defaults.budgets]`) | Total privacy budget in the **mechanism's native unit** (ε for PureDP/RenyiDP, ρ for ZCDP/ApproximateDP). Default is looked up from `[defaults.budgets][mechanism]`, so omitting `--budget` gives the iso-(ε, δ)-DP value for whichever `--mechanism` you picked. exp4/exp7 interpret it as **target ε** for their iso comparison. |
| `--delta`        | `1e-10`        | Required for `ApproximateDP` / `RenyiDP`; also used by exp4/exp7 to convert ε ↔ ρ |
| `--allocation`   | `exp_leaves`   | `equal` / `exp_leaves` / `exp_root` / `square_leaves`                            |
| `--ext`          | `png`          | Plot extension: `png` (light) / `pdf` (vector) / `svg`                           |
| `--out`          | auto           | Output directory (default `tests/out/<exp_name>`)                                |

## Workload selection (every experiment)

Every script (exp1 through exp8) also accepts:

| Flag                  | Default     | Meaning                                                                              |
|-----------------------|-------------|--------------------------------------------------------------------------------------|
| `--workload`          | `identity`  | One of: `identity`, `marginals1`, `marginals2`, ... (up to `marginals<n_query_cols>`), `random`, `mixed` |
| `--n_random_queries`  | `20`        | Number of rows when `--workload random` or `--workload mixed`                        |
| `--random_density`    | `0.1`       | Per-cell Bernoulli probability for `random` / `mixed`                                |

Workload semantics:

* `identity` — one query per contingency cell (`Q = I`). Sensitivity Δ = 1.
* `marginals<k>` — all k-way marginals over the `--queries` columns. Sensitivity Δ = `C(n_query_cols, k)`.
* `random` — random binary matrix of shape `(n_random_queries, n_cells)` with the given density. Empty columns are padded so every cell appears in at least one query. Sensitivity Δ ≈ `n_random_queries · random_density`.
* `mixed` — concatenation of `marginals1`, `marginals2` and `random`. Heterogeneous Δ.

Because the results vary a lot with the workload (especially when comparing mechanisms — Laplace's noise scales as Δ, Gaussian's as √Δ), every experiment now exposes the same workload knobs so you can dial sensitivity up or down. In exp1 (`marginals<k>` requires at least k query columns) any sweep iteration that can't build the chosen workload is skipped with a warning.

Exception: exp5 has its own `--workloads` (plural) flag — that script's whole point is to iterate over several workloads at once.

## Per-experiment extra flags

* **exp1**: `--query_progression` (cumulative prefix list), `--depth_sweep` (levels to test).
* **exp3**: `--top_queries N` (limits the plotted queries to the top-N by truth magnitude).
* **exp4**: `--budgets 0.1 0.5 1 2 5 10 20` — list of **target ε** values; `--mechanisms` to overlay several mechanisms on the same x-axis.
* **exp5**: `--workloads identity marginals1 marginals2 random mixed` (multiple workloads — overrides the single `--workload`).
* **exp6**: `--strategies equal exp_leaves exp_root square_leaves inv_nodes`.
* **exp7**: `--mechanisms`, plus `--budgets_pure / _zcdp / _approx / _renyi` to override the auto iso-(ε, δ) budget for a specific mechanism. The override unit is ε for PureDP/RenyiDP, ρ for ZCDP/ApproximateDP.
* **exp8**: `--metric {cramer,pearson}` (default `cramer` for categorical census data), `--include_geo` (add hierarchical geographic columns to the matrix).
* **exp9**: `--n_queries_sweep 10 50 100 250 500 1000` (rows of Q to sweep), `--random_density 0.1`. When `--process_until` is omitted, exp9 uses the **root-most column** of the selected dataset's hierarchy (2 tree levels) so the matrix-size effect is isolated from tree-traversal cost. Pass `--process_until <deeper_level>` if you want to see both effects together. The workload is fixed to random binary; the `--workload` flag is intentionally absent.

## Example invocations

Quick smoke (fastest possible):

```powershell
python -m tests.exp1_runtime --dataset viviendas --process_until REGION --queries P02 --trials 1
python -m tests.exp2_utility_per_level --dataset viviendas --process_until REGION --queries P02 --trials 1
```

Thesis-style per-level TVD on identity workload:

```powershell
python -m tests.exp2_utility_per_level --dataset viviendas --process_until COMUNA --queries P02 P03A --trials 5 --workload identity
```

Same experiment with a high-sensitivity random workload (Δ ≈ 60) — exposes the Gaussian advantage:

```powershell
python -m tests.exp2_utility_per_level --dataset viviendas --process_until COMUNA --queries P02 P03A --trials 5 --workload random --n_random_queries 200 --random_density 0.2
```

Budget sweep with all four mechanisms automatically calibrated to the same target ε at δ=1e-10:

```powershell
python -m tests.exp4_budget_sweep --dataset viviendas --process_until COMUNA --queries P02 --budgets 0.1 0.5 1 2 5 10 --mechanisms ZCDP PureDP ApproximateDP RenyiDP --delta 1e-10
```

PureDP-vs-Gaussian crossover (run exp7 twice with different workloads):

```powershell
# Δ=1: PureDP wins
python -m tests.exp7_mechanisms --queries P02 P03A P03B --workload identity --budget 10 --delta 1e-10 --out tests/out/exp7_iso_identity

# Δ ≈ 60: Gaussian variants win
python -m tests.exp7_mechanisms --queries P02 P03A P03B --workload random --n_random_queries 200 --random_density 0.2 --budget 10 --delta 1e-10 --out tests/out/exp7_iso_highsens
```

Compare allocation strategies under a marginals workload:

```powershell
python -m tests.exp6_budget_allocation --dataset viviendas --process_until COMUNA --queries P02 P03A P03B --workload marginals2 --trials 5
```

Cramér's V heatmaps including geography:

```powershell
python -m tests.exp8_correlation --queries P02 P03A P03B --include_geo --metric cramer
```

Runtime vs query-matrix size (rows of Q only, REGION depth by default):

```powershell
python -m tests.exp9_matrix_size --queries P02 P03A --n_queries_sweep 10 50 100 250 500 1000 --trials 3
```

## Truth cache

The per-cell truth tensor for each `(dataset, hierarchy, queries, process_until)` combo is
built once and cached at `tests/out/_truth_cache/<hash>.npz`. Delete that folder to force a
rebuild (e.g. after the upstream contingency-construction logic changes).

The cache is shared across workloads — the truth tree is built with `Q = I` and reused
regardless of which workload the experiment actually runs.

## Regenerating plots without re-running

After tweaking labels, units, or footers, regenerate every plot from the saved CSVs:

```powershell
python -m tests._replot
```

This reads each `tests/out/<expN>/expN_*.csv` + `*_config.json` and rebuilds the figures
using the current plotting functions. No mechanism calls, no Pyomo solves.

## Notes

* The OpenDP noise samplers in `privacy/noisy.py` do not accept a seed, so trials are
  statistically independent but not bit-reproducible. Random workloads (`random` / `mixed`)
  also draw fresh entropy each run for consistency.
* Each experiment skips `construct_microdata()` except for **exp8**, which needs the
  reconstructed microdata to compute pairwise correlation. exp8 is therefore noticeably
  slower than the others.
* For exp4 and exp7, `--budget` is the **target ε** at the given `--delta`. The script
  converts to ρ via inverse Bun–Steinke (ρ = ε² / (c + √(c² + ε))², c = √log(1/δ)) for
  ZCDP and ApproximateDP, so all mechanisms operate under the same (ε, δ)-DP guarantee
  and the resulting bars / curves are directly comparable.
