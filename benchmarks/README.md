# Benchmarks

The evaluation of the paper, in two parts that never share a budget:

1. **Comparison with the DAS** (experiments 1 and 2). IPUMS 1940 only. Both the DAS and our
   implementation run with the DAS's own published configuration, not ours.
2. **Utility of our implementation** (experiments 3 to 7). All five datasets, marginal pipeline,
   ρ-zCDP. Experiment 3 runs first and fixes the composition and the budget of the rest.

Part 1 runs first. Everything is run from the repository root; data lives in `data/<dataset>/`
and results in `data/out/<dataset>/`, both outside git.

| File | What it does |
|---|---|
| `<dataset>/prepare.py` | raw files → `data/<dataset>/<dataset>.parquet` and its `_sample` subset |
| `<dataset>/columns.py` | checks every column against its domain and saves the node counts (`_nodes.json`) |
| `<dataset>/domains.py`, `constraints.py`, `marginals.py` | declared domains (with their source), edit rules, marginal structures |
| `<dataset>/driver.py` | one TopDown run; flags in `runner.py` |
| `metrics.py` | utility of a finished run, in its own process |
| `experiments.py` | the configurations each experiment needs, and the decided budget |
| `orchestrate.py` | runs experiments round by round, resumably, into `runs.jsonl` |

---

## 0. Setup on the server

### Our pipeline

```bash
git checkout benchmarks && git pull
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

Gurobi needs a full licence (`GRB_LICENSE_FILE`). Copy the data, keeping the paths:

| Dataset | Files |
|---|---|
| adult | `data/adult/adult.parquet`, `adult_nodes.json` |
| sinasc | `data/sinasc/sinasc.parquet`, `sinasc_nodes.json` |
| spanish_census | `data/spanish_census/spanish_census.parquet`, `spanish_census_nodes.json`, `dr_CensoPersonas_2021.json` |
| chilean_census | `data/chilean_census/chilean_census.parquet`, `chilean_census_nodes.json`, `data/csv-personas-censo-2017/etiquetas_persona_comuna_16r.csv` |
| ipums_1940 | `data/ipums_1940/EXT1940USCB.dat.gz`, `EXT1940USCB_AK.dat` |

Also copy every `*_sample.parquet` and `*_sample_nodes.json`. Then build IPUMS and smoke-test:

```bash
python -m benchmarks.ipums_1940.prepare --full    # 132M persons
python -m benchmarks.ipums_1940.columns           # writes ipums_1940_nodes.json
python -m benchmarks.orchestrate exp3_composition --sample --reps 1
```

The smoke test takes minutes. Check that `memory_mb` is no longer `null` in
`data/out/<dataset>/runs_sample.jsonl`: peak memory is only recorded on Linux.

### The DAS

The runnable 1940 path lives on the branch `make-1940-path-runnable` of the clone of
`DAS_2020_DHC_Production_Code`. This is the public repository of the original US Census DAS, so carry it as a bundle:

```bash
# local
git -C ~/Research/DAS_2020_DHC_Production_Code bundle create das1940.bundle make-1940-path-runnable
# server
git clone -b make-1940-path-runnable das1940.bundle ~/das/repo
```

Then follow `RUNNING_1940.md`, section 4, of that branch: Java 17, Python 3.11 with `numpy<2`,
PySpark 3.5, Gurobi, and `~/das/env.sh`. The `dprng.py` shim is `das1940/winshim/dprng.py` on
the local machine (not in git). Install GNU time too (`apt install time`) for the memory measurement.

---

## How runs are executed and recorded

- **One run at a time.** A run is the driver (the only thing timed; it records seconds per phase
  and peak memory) followed by `benchmarks.metrics` in a separate process. Never launch two
  orchestrators, or the DAS, at the same time: they would contaminate each other's time and memory.
- **Round by round.** Round r runs every configuration of every dataset once before round r + 1
  starts, so medians over the first rounds are available long before the fifth.
- **Results.** Every run, failed ones included, appends one line to
  `data/out/<dataset>/runs.jsonl`: the configuration, the round, `status` (`ok`, `error`,
  `timeout`, `out_of_memory`, `metrics_error`), the runner record (width, sensitivity, bags, ρ per level, seconds,
  memory) and the metrics per tree level. Next to it stay `<name>.json`, `<name>_metrics.json`
  (every pair and triple) and `<name>.log` (live TopDown output). The synthetic CSV is deleted.
- **Shared runs.** A configuration has one name wherever it appears, so a run several experiments
  need is run once. Example: `blocks_rho1_sqrt_sweep_r3`.
- **Stop and resume.** `Ctrl+C` or `kill <orchestrator pid>` (never `kill -9`) kills the current
  run, which leaves no line and is repeated by relaunching the same command. `--retry` repeats
  what failed or timed out; `--dry-run` lists what is pending; `--timeout H` kills a run after H
  hours and records it as `timeout`.

Early analysis, while a step is still running:

```python
import pandas as pd
runs = pd.read_json('data/out/sinasc/runs.jsonl', lines=True)
runs = runs[runs.status == 'ok']
runs['configuration'] = runs.name.str.replace(r'_r\d+(_sample)?$', '', regex=True)
runs['estimation'] = runs.seconds.str.get('estimation')
runs['tvd2_leaves'] = runs.levels.str.get(-1).str.get('tvd2')
runs.groupby('configuration')[['estimation', 'tvd2_leaves']].agg(['count', 'median'])
```

---

## Part 1 — Comparison with the DAS (experiments 1 and 2)

### The DAS configuration

Transcribed from `configs/Census1940/DDP2010_Update/ipums_1940.ini`, `ipums_1940_local.ini` and
`programs/strategies/strategies_1940.py` on that branch:

| | |
|---|---|
| Privacy | pure ε-DP, geometric mechanism, bounded DP (every query's sensitivity doubled) |
| Total budget | ε = 4 (`global_scale: 1/4`) |
| Per level | uniform, 1/5 each for National, State, County, Supdist, Enumdist. **Reconstructed**: the shipped line has seven values for five levels |
| Per query, at every level | hhgq 0.20 · age × hispanic × race × citizen 0.50 · age × sex 0.05 · ageGroups4 × sex 0.05 · ageGroups16 × sex 0.05 · ageGroups64 × sex 0.05 · detailed 0.10 |
| Invariants | state totals; household and group-quarters counts per enumeration district |
| Histogram | hhgq 8 × sex 2 × age 116 × hispanic 2 × race 6 × citizen 2 = 44,544 cells |

### Step 1.1 — Run the DAS

Alaska first (the config's default input), one output folder per repetition:

```bash
source ~/das/env.sh && cd $DAS_REPO
export DAS_1940_OUTPUT=~/das/out/alaska_r1 && mkdir -p $DAS_1940_OUTPUT
/usr/bin/time -v spark-submit --driver-memory 8g --master 'local[*]' \
    --conf spark.driver.maxResultSize=0 \
    das_framework/driver.py configs/Census1940/DDP2010_Update/ipums_1940_local.ini --loglevel INFO \
    2>&1 | tee $DAS_1940_OUTPUT/run.log
```

The log should reach 239 optimisation problems and `Run completed in ... seconds`
(`RUNNING_1940.md` §4.8). For the national run, decompress the input (`gunzip -k
EXT1940USCB.dat.gz`, 39.5 GB) and add `configs/Census1940/DDP2010_Update/ipums_1940_national.ini`
(not tested yet):

```ini
[DEFAULT]
INCLUDE=ipums_1940_local.ini

[reader]
PersonData.path: $DAS_1940_INPUT/EXT1940USCB.dat
UnitData.path: $DAS_1940_INPUT/EXT1940USCB.dat
```

It is 158,374 optimisation problems against Alaska's 239: time the Alaska run first and
extrapolate before committing the server (`RUNNING_1940.md` §5).

**Where the results are.**

- `$DAS_1940_OUTPUT/person/part-*`: the synthetic persons, `|`-separated, no header, columns
  `SCHEMA_TYPE_CODE SCHEMA_BUILD_ID TABBLKST TABBLKCOU SUPDIST ENUMDIST EPNUM RTYPE GQTYPE RELSHIP
  QSEX QAGE CENHISP CENRACE LIVE_ALONE`. **`CITIZEN` is not written.**
- `$DAS_1940_OUTPUT/run.log`: the DAS log, and at its end the `/usr/bin/time -v` report
  (`Elapsed (wall clock) time`, `Maximum resident set size`).

**To build: `benchmarks/ipums_1940/das_output.py`**, turning that output into our schema so
`python -m benchmarks.metrics ipums_1940 das_eps4_r1` evaluates it like one of our runs:

| DAS | Ours |
|---|---|
| `TABBLKST`, `TABBLKCOU`, `SUPDIST`, `ENUMDIST` | `STATEFIP`, `COUNTY`, `SUPDIST`, `ENUMDIST` |
| `GQTYPE` 000, 101, 201, …, 701 | `hhgq` 0 … 7 |
| `QSEX` 1, 2 · `CENHISP` 1, 2 | `sex` 0, 1 · `hispanic` 0, 1 |
| `QAGE` · `CENRACE` 01 … 06 | `age` · `race` 0 … 5 |
| — | `citizen` (missing: see the decisions below) |

### Step 1.2 — Run ours with the DAS configuration

**To build in `runner.py`** before `exp1` and `exp2` can list their runs (they refuse today):

1. `--epsilon`: `PureDP` (discrete Laplace, the same distribution as the geometric mechanism)
   instead of zCDP, split with `--composition uniform`.
2. The neighbouring relation is already aligned: TopDown uses bounded DP (sensitivity × 2), like
   the DAS.
3. The per-query proportions: ours measures one workload per level with one ε.
4. The invariants: the DAS holds the state totals and, per enumeration district, the occupied
   households and group quarters of each type. Settled: `--unit-bounds` reproduces them, after
   `python -m benchmarks.ipums_1940.prepare --units --full` has written the unit counts once.

How 3 and 4 were settled is in the decisions table. Both arms run through the campaign script,
which also measures time and machine-wide memory:

```bash
bash benchmarks/ipums_1940/campaign.sh
```

- **Experiment 1:** utility at the DAS configuration: ours full-joint against the DAS, TVD per
  level and L1 of the DAS queries (`marginals.WORKLOAD`).
- **Experiment 2, performance:** time and memory of the three systems on IPUMS 1940 (Table 3).
  Its scaling sweep moved to ESCENSUS, step 1.3.
- **Results:** ours in `data/out/ipums_1940/runs.jsonl`; the DAS in `~/das/out/*`, evaluated into
  `data/out/ipums_1940/das_*_metrics.json`.

### Step 1.3 — Scaling with the number of attributes (experiment 2): 100 runs

Width, time and peak memory (session PSS) against the first d of the 78 ESCENSUS columns, both
pipelines, at the Part 2 configuration (ρ = 1, sqrt, 20 workers, `BarConvTol` 1e-12): the sweep is
not compared with the DAS. IPUMS 1940 cannot show it, since its full joint over 6 attributes is
44,544 cells and never breaks.

```bash
unset BAR_CONV_TOL
python -m benchmarks.orchestrate exp2 --reps 5 --timeout 4 --min-available-gb 2 2>&1 | tee -a data/out/orchestrate.log
```

`--min-available-gb 2` kills a run as soon as the host's `MemAvailable` falls below 2 GB, that is,
once the RAM is full, and records it as `out_of_memory`. Not fitting in RAM is the finding, and
waiting it out in swap would only load the shared host for up to the timeout. The trigger is the
RAM, not the run's swap: the kernel moves idle pages to swap long before the RAM is full, so a swap
trigger would kill runs that fit. The kill lands within one sampling interval (1 s), so some pages
may reach swap first; `peak_session_swap_gb` and `lowest_available_gb` record both.

- **Marginal:** d ∈ {2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 78}, each declared bag cut to the
  kept columns. d = 78 is the `blocks` row of `exp6_memory` and is skipped as recorded.
- **Full-joint:** d = 2 … 8. The joint is 7.4·10⁶ cells at d = 6 (502 s on the laptop sample with
  4 workers), 1.5·10⁹ at d = 7 (12 GB per dense vector) and 8.5·10¹⁰ at d = 8, where one vector
  alone exceeds the server's RAM. Running out of memory is the expected result: the run is
  recorded as `out_of_memory` (or `error`, if an allocation fails before anything swaps) with its
  sampled memory peak, and the figure draws the width of every larger d as computed, not run.
- No metrics: the TVD of different attribute sets is not comparable, and the `memory` flag skips them.

---

## Part 2 — Utility of our implementation (experiments 3 to 7)

Marginal pipeline, ρ-zCDP, the five datasets, 5 rounds per configuration. Per run at ρ = 1 on
the laptop: Adult seconds, SINASC ~2.5 min, Spain ~9 min, Chile ~80 min, IPUMS not measured
(time one `python -m benchmarks.ipums_1940.driver --name calibration` first). One round of a step
is therefore ~6 h plus IPUMS.

### Step 2.1 — Composition (experiment 3, first half): 100 runs

The four compositions (`exponential`, `uniform`, `proportional`, `sqrt`) at ρ = 1, run once:

```bash
python -m benchmarks.orchestrate exp3_composition --reps 5 --timeout 12 2>&1 | tee -a data/out/orchestrate.log
```

Compare the median TVD per tree level of each composition. Write the winner in the decisions
table and in `COMPOSITION` in `experiments.py`.

### Step 2.2 — Budget (experiment 3, second half): 100 runs

ρ ∈ {0.1, 0.5, 1, 2, 5} with the chosen composition. ρ = 1 is already done in step 2.1:

```bash
python -m benchmarks.orchestrate exp3_budget --reps 5 --timeout 12 2>&1 | tee -a data/out/orchestrate.log
```

Round 1 is one run of every ρ on every dataset, then round 2, and so on. Write the chosen ρ in
the decisions table and in `RHO` in `experiments.py`.

### Step 2.3 — Experiments 4, 5 and 7: 50 runs

The reference run of each dataset (its default structure, `RHO`, `COMPOSITION`, sweep) is already
in step 2.2, and every experiment below reuses it:

```bash
python -m benchmarks.orchestrate exp4 exp5 exp7 --reps 5 --timeout 12 2>&1 | tee -a data/out/orchestrate.log
```

- **Experiment 4** (3-way TVD by granularity) reruns the reference configuration with
  `triples=100` (25 runs, named `..._sweep_triples100_rN`): `benchmarks.metrics` then evaluates
  every triple contained in a bag plus up to 100 of the rest per number of measured pairs (0, 1,
  2). It has to rerun because the synthetic CSV of the reference runs is already deleted. The key
  goes to the metrics, not the driver, so the runs are otherwise identical to the reference.
- **Experiment 5** (utility by marginal coverage, MI heatmap) runs nothing: it reads `levels` and
  `<name>_metrics.json` of the reference run.
- **Experiment 6** (marginal size) has its own step, 2.5, with a generated family of structures.
- **Experiment 7** (global MIP against the sweep): 5 new configurations. The MIP may not finish at
  full depth; a `timeout` in round 1 is itself the result, so drop that configuration from
  `experiments.py` before the next rounds instead of paying 12 h per round.

### Step 2.4 — Depth (experiment 3, third part): 20 runs

The reference run of every dataset cut to **3 tree levels** (national + 2), so that all five spend
the same total budget over the same depth and their per-level utility becomes comparable:

```bash
python -m benchmarks.orchestrate exp3_depth --reps 5 --timeout 12 2>&1 | tee -a data/out/orchestrate.log
```

Spain is excluded from `exp3_depth`: it already has 3 levels, so its step 2.2 reference runs are
its arm. `read_nodes` slices the saved counts to the truncated hierarchy
(`stored['nodes'][:len(hierarchy) + 1]`, with a prefix check), so the budget is split over the
three levels that exist and not over the five the file records.

### Step 2.5 — Structure (experiment 6): 115 runs

A family of structures derived mechanically from each dataset's declared one, by
`benchmarks/structures.py`: **merge** the two bags sharing an attribute whose union has the fewest
cells (rungs `_m1`, `_m2`, ... toward the joint), or **split** every bag of three or more attributes
by dropping the pair with the largest cardinality product (rungs `_s1`, `_s2`, ... away from it).
Both rules are width-extremal, so they are deterministic with no tuning constant, and neither reads
the data, so no budget goes to selection. Each `marginals.py` registers its family as `DERIVED`.

```bash
python -m benchmarks.orchestrate exp6 --reps 5 --timeout 4 2>&1 | tee -a data/out/orchestrate.log
```

27 configurations x 5 rounds = 135 runs, of which the 20 declared anchors are the step 2.2
reference runs and are skipped. `chilean_census` is excluded: its 21 bags come from 8 declared
marginals plus 30 constraint scopes, every scope is a mandatory clique (`topdown.py:285`), and
merging or splitting the marginals moves its width by 2% and its bag count by 1.

The analysis plots the bags the junction tree **returned** (`bags` in the run record), never the
ones requested: a split can be re-fused by the triangulation.

---

## Decisions and findings

Fill in as the steps finish. The values that change runs also go in `experiments.py`.

| Decision | Value | Evidence | Date |
|---|---|---|---|
| Part 1 budget | the DAS configuration above | `ipums_1940.ini`, `strategies_1940.py` | 2026-09-14 |
| Neighbouring relation, both parts | bounded DP: every sensitivity × 2 (`TopDown.bounded_dp_factor`), like the DAS | DAS `programs/engine/primitives.py` | 2026-09-14 |
| Part 1: ours, per-query proportions | the DAS's own `.2 .5 .05 .05 .05 .05 .1`, via `--workload das` + `set_query_budget` | step 1.1: `queriesprop` in `ipums_1940.ini` | 2026-09-29 |
| Part 1: invariants of ours | the DAS's: exact totals of the nation and the states, plus `hhgq_total_lb/ub` from the per-enumdist unit counts (`--unit-bounds`) | step 1.2 | 2026-10-01 |
| Part 1: budget axis of experiment 1 (ε = 4 only, or scaled) | _open_ | | |
| Part 1: DAS national, or state by state | _open_ | | |
| Part 1: `citizen`, not written by the DAS | score both sides over the five columns its MDF publishes (`score_five.py`); `citizen` is measured by neither | step 1.2 | 2026-09-29 |
| Part 1: DAS repetitions | 5 (reps 2-6 timed; rep1 utility only), and 5 per arm of ours | step 1.1 | 2026-09-29 |
| Part 2 composition (`COMPOSITION`) | `sqrt` | step 2.1 below: best leaf TVD on 4 of 5 datasets, and better than `exponential` on all 5 | 2026-09-22 |
| Part 2 budget (`RHO`) | `1` | step 2.2 below: no budget wins on utility, so this is the operating point of experiments 4 to 7, not a privacy claim; the sweep is the result | 2026-09-23 |
| Experiment 4: split contained and other triples | every contained triple, plus up to 100 of the rest per number of measured pairs (0, 1, 2), weighted by stratum size; own runs with `triples=100` | step 2.3 below: the uniform sample of 100 reached 0 contained triples on Chile and Spain | 2026-10-04 |
| Experiment 6: which structures to compare | the family generated from each declaration (`structures.py`), Chile excluded | step 2.5 below | 2026-09-30 |

### Step 1.1 — DAS

Six national runs on the shared server (24 cores, 123.5 GB RAM, 119 GB swap), 20 local Spark
executors, all `ok`. `das_rep1` predates the campaign script and has utility only; reps 2 to 6 have
the DFXML and the stage times. `das_rep6` is the one with the swap instrumentation (step 1.1b).

| rep | total s | reader | engine | writer | peak resident GB |
|---|---|---|---|---|---|
| rep2 | 16,140.5 | 3,082.5 | 11,978.7 | 1,077.4 | 120.32 |
| rep3 | 16,229.7 | 3,092.3 | 12,049.2 | 1,080.0 | 119.98 |
| rep4 | 16,279.8 | 3,099.3 | 12,088.1 | 1,082.8 | 120.05 |
| rep5 | 16,310.1 | 3,103.7 | 12,100.7 | 1,093.6 | 120.52 |
| rep6 | 16,231.1 | 3,107.4 | 12,033.0 | 1,082.9 | 119.06 |
| **median** | **16,231.1** | 3,099.3 | 12,049.2 | 1,082.8 | 120.05 |

Spread 1.05% from fastest to slowest, and the engine is 74% of the time in every replicate. The reps
ran back to back and the totals rise monotonically, 16,140.5 to 16,310.1 — see step 1.1b.

**The configuration is matched to ours, verified line by line in `ipums_1940_local.ini`.** This
matters more than it sounds: a reviewer will ask, and the file that ships with the release
(`ipums_1940.ini`) does not run.

| | DAS | ours |
|---|---|---|
| framework | `privacy_framework: pure_dp` | `PureDP` |
| mechanism | `dp_mechanism: geometric_mechanism` | discrete Laplace, same family |
| total budget | `global_scale: 1/4`, and under pure DP the engine derives epsilon = 1 / global_scale (`budget.py:224`) | `--epsilon 4` |
| per level | `geolevel_budget_prop: 1/5, 1/5, 1/5, 1/5, 1/5` | `--composition uniform` |
| neighbouring | `bounded_dp_multiplier = 2.0` | `TopDown.bounded_dp_factor = 2` |
| per query | `queriesprop = .2, .5, .05, .05, .05, .05, .1` | `set_query_budget(...)`, the same seven |

So the budget is epsilon = 4 flat on both sides, with no zCDP conversion and no delta anywhere in
Part 1. What remains different is the **estimator** and the **invariant set**: the DAS holds
`theinvariants.state = tot`, `theinvariants.enumdist = gqhh_vect, gqhh_tot` and
`theconstraints.enumdist = hhgq_total_lb, hhgq_total_ub`, where we hold the exact total at levels 0
and 1 only. Invariants are published for free, so the DAS holds strictly more free information than
we do.

> `epsilon_budget_total = 4.0` is still in `ipums_1940.ini`, but this release no longer reads it
> (only `programs/experiment/experiment.py:41` does), which is why the local config restates the
> budget as `global_scale`. And `ipums_1940.ini` gives **seven** `geolevel_budget_prop` values for
> the five geolevels it declares, which trips the assert at `budget.py:420`; the uniform 1/5 line is
> a reconstruction, documented in `programs/strategies/strategies_1940.py`.

### Step 1.1b — does the DAS page, and does it matter?

During the campaign the host sat at 122 of 123 GiB with 45-60 GiB of swap in use, which put two
questions on the table. `das_rep6` was run with the sampler extended to answer both.

| | rep6 |
|---|---|
| peak **resident** (its JVM and pyspark workers, `ps -o rss=`) | 119.06 GB |
| peak **footprint** (resident plus its own swapped pages, `smaps_rollup`) | **214.89 GB** |
| footprint / resident | **1.80x** |
| host RAM | 123.5 GB, so **oversubscribed 1.74x** |
| paged out over the run | 175.8 GiB |
| paged in over the run | 96.8 GiB |
| iowait | 443 s of 390,048 cpu-seconds = **0.114%** |

**The DAS does not fit in 123 GB of RAM.** Its working set on this instance is 215 GB; it completes
only because the host has 119 GB of swap. The `peak rss 120G` reported for reps 2 to 5 understates
the requirement by 80%, because `ps -o rss=` excludes pages that are in swap. **Quote 215 GB, and
say which instrument produced it.**

**But paging cost it nothing measurable.** iowait was 0.114% of available CPU, and rep6's 16,231.1 s
is the exact median of the five timed replicates, so the instrumented run sits dead centre of the
band rather than at its slow end. The monotone drift across reps 2 to 5 is 1.05% in total, which
bounds the cost of paging even on the most pessimistic reading: attribute all of the drift to swap
and swap cost the DAS 1%. Consistent with the `vmstat` spot check taken during the campaign, where
`wa` was 0 in every sample and `b` was 0 or 1 against 20-22 runnable.

The two readings are not in tension: 176 GiB went out and only 97 GiB came back, so most of what was
evicted was cold and never faulted in again.

> Instrument note for the comparison below. The DAS figure is its own processes' resident plus
> swapped memory. Ours is machine-wide `MemTotal - MemAvailable`, which includes page cache from
> reading the 40 GB input and anything else on the host. Ours therefore over-counts us, so every
> memory ratio in step 1.2 is a **lower bound** on our advantage. Neither of our arms paged: the
> full-joint peak is 59.9 GB machine-wide on a 123.5 GB host.

### Step 1.2 — ours with the DAS configuration

Ten national runs, 5 per arm, 20 workers, all `ok` (`benchmarks/ipums_1940/campaign.sh`), on the
same host as the DAS. Since 2026-10-01 both arms also hold the DAS's invariants (`--unit-bounds`,
below), so the arms differ in the measurement structure and nothing else — same epsilon, same
per-level split, same bounded DP, same invariants, both pure DP:

```
fj:  --full-joint --workload das --epsilon 4 --composition uniform --unit-bounds
mg:  --structure das --epsilon 4 --rounding mip --composition uniform --unit-bounds
```

The ten runs without the unit bounds (2026-09-27) are kept locally in
`data/out/ipums_1940/exp1_sin_unidades/` and summarised in the box at the end of this step.

**Cost**, median of 5, on the same host as the DAS:

| | DAS | ours full-joint | ours marginal |
|---|---|---|---|
| wall, median | 270.5 min | **63.4 min** | **30.0 min** |
| speedup vs DAS | 1.0x | **4.3x** | **9.0x** |
| min-max | 16,140-16,310 s (1.05%) | 3,782-3,823 s (1.1%) | 1,802 s, all five |
| peak memory | 214.9 GB | 76.0 GB (72.8-80.2) | 22.8 GB (22.0-23.1) |
| net of the host baseline | — | 70.4 GB | 17.4 GB |
| times less than the DAS | — | **2.8x** (3.1x net) | **9.4x** (12.4x net) |
| did it page? | yes, 176 GiB out | no | no |
| cores busy of 24 | — | 18.3 | 16.8 |
| width | 44,544 | 44,544 | 4,640 |
| sensitivity | 14 | 14 | 4 |

`wall_seconds` is quantised to the 20 s sampling period of `campaign.sh`, which is why the marginal
arm reads 1,802 s five times; its estimation phase in the run records is 1,522-1,530 s.

**Utility**, 2-way TVD over the five columns the DAS MDF publishes (`citizen` is written by neither),
median of 5 each:

| level | nodes | DAS | ours full-joint | fj / DAS | ours marginal | mg / DAS |
|---|---|---|---|---|---|---|
| national | 1 | 0.000643 | 0.000067 | **0.10** | 0.000239 | **0.37** |
| STATEFIP | 51 | 0.004323 | 0.001925 | **0.45** | 0.001474 | **0.34** |
| COUNTY | 3,108 | 0.025564 | 0.025352 | 0.99 | 0.019973 | **0.78** |
| SUPDIST | 3,205 | 0.025157 | 0.024955 | 0.99 | 0.019720 | **0.78** |
| ENUMDIST | 152,009 | 0.178726 | 0.178259 | 1.00 | 0.147493 | **0.83** |

L1 on the DAS's own seven query groups:

| level | DAS | ours full-joint | fj / DAS | ours marginal | mg / DAS |
|---|---|---|---|---|---|
| national | 7,196.0 | 4,037.0 | 0.56 | 2,017.0 | **0.28** |
| STATEFIP | 2,816.7 | 2,721.9 | 0.97 | 1,408.0 | **0.50** |
| COUNTY | 1,457.0 | 1,458.5 | 1.00 | 633.2 | **0.43** |
| SUPDIST | 1,455.9 | 1,459.2 | 1.00 | 629.1 | **0.43** |
| ENUMDIST | 381.7 | 381.8 | 1.00 | 280.2 | **0.73** |

**The full-joint arm is the controlled replication, and with the same invariants it reaches parity
below the states**: 0.99-1.00x at County, Supdist and Enumdist, with disjoint ranges against the
DAS in our favour, and the same L1 on the Bureau's queries (381.8 against 381.7 at the leaves).

**The marginal arm is the result.** Measuring two bags instead of seven query groups over the full
joint, at the same epsilon = 4 and the same invariants, it is better at every level — 0.83x at the
leaves, where the released microdata actually lives — while being 9.0x faster and needing 9.4x less
memory. It halves the L1 on the Bureau's own queries, or better, at four of the five levels.

**What the unit bounds changed.** DAS / ours over the pairs with and without `hhgq`:

| | COUNTY | SUPDIST | ENUMDIST |
|---|---|---|---|
| fj, pairs with `hhgq`, before | 0.919 | 0.919 | 0.883 |
| fj, pairs with `hhgq`, **now** | **1.005** | **1.006** | **1.004** |
| fj, pairs without `hhgq`, before -> now | 1.006 -> 1.008 | 1.006 -> 1.008 | 1.007 -> 1.002 |

The DAS's 2-3% lead at the fine levels was entirely in the pairs with `hhgq`, and it closed when ours
got the same invariants; the other pairs did not move. Per pair at Enumdist, `hhgq x sex` 0.06014
against the DAS's 0.06008, `hhgq x race` 0.04780 against 0.04850. The marginal arm gained more —
Enumdist 0.1593 -> 0.1475, County 0.0238 -> 0.0200, disjoint at every level — mostly in the `hhgq`
pairs its two bags split apart (`hhgq x hispanic` 0.0604 -> 0.0234).

The price is memory in the full-joint arm, 59.9 -> 76.0 GB (+27%); the marginal arm goes 22.3 ->
22.8 GB and neither arm got slower. Every active cell now sits in one upper-bound row of its type and,
for a group-quarters type, in one lower-bound row, which at the leaves adds about 1.5x the nonzeros
of the geographic rows (measured on the true supports). Most of them are redundant — the upper rows
where the child has units of that type (`units x 99999`) and the lower rows where it has none
(`>= 0`) — and a third of the leaf variables exist only to be pinned at zero, because the pruning
gives every child its parent's support. Dropping the redundant rows, or giving each child its own
support, would recover it; neither is implemented (decided 2026-10-01).

**The DAS's invariants, and how ours reproduces them.** `theinvariants.enumdist = gqhh_vect,
gqhh_tot` publishes the occupied households and group-quarters facilities of each type per
enumeration district, read from the H records; `hhgq_total_lb/ub` (`querybase_constraints.py:123`)
turns them into persons of type g >= units of type g for every group-quarters type, and <= units x
99999, i.e. none at all where there is no such unit. The DAS computes them at the enumeration
districts and sums them upwards (`nodes.py`, `addInReduce`), recomputing at the states with the
exact total and keeping the stricter. Ours: `prepare.py --units` writes the unit counts with the same
recode as the persons, `ipums_1940/constraints.py::unit_bounds` builds one `SumBoundOfNode` per type
and sense, and `--unit-bounds` registers them at every level. What the DAS adds at the states (<=
total minus the minimum of the other types; = total when a type is alone) is implied by the exact
state total plus the lower bounds, so the feasible sets are identical everywhere except one national
bound: summing the state bounds excludes the population of states with no facility of a type, which
happens for types 2, 4 and 5 (in 2, 3 and 15 states). That bound is 308x, 711x and 12,727x above the
true count, so it never binds. On the full data the true counts satisfy the bounds at every level
(no violation, server, 2026-10-01). On the Alaska sample a release without them puts persons in 380
enumeration-district x type combinations with no facility of that type.

**Why we are better above the states, first half: the solver tolerance.** The DAS solves every model at
`BarConvTol` 1e-8 (`configs/default.ini`, read at `optimization/optimizer.py:340`); ours sets 1e-12
(`optimizers.DEFAULT_SOLVER_OPTIONS`, a correctness default, see `CLAUDE.md`). The national node
solved alone, same noise, weights and constraints, median of 5 seeds: at 1e-8 the mean 2-way TVD is
0.000187 and the support 44,536 of 44,544 cells; at 1e-12, 0.000070 and 9,746, against a true
support of 15,633. On the server, the full-joint arm rerun at the DAS's tolerance, 5 replicates
with the unit bounds (`BAR_CONV_TOL=1e-8 ARMS=fj TAG=1e8 bash benchmarks/ipums_1940/campaign.sh`,
`ours_fj1e8_rep1..5`, 2026-10-03), reproduces the DAS on the six pairs without `hispanic` at every
level, 0.96 / 0.99 / 1.00 / 1.00 / 1.00x from the root down, and its workload L1 is 7,250 against
7,196 at the root and 2,773 against 2,817 at the states. Over all ten pairs it is 0.27x at the root
and 0.47x at the states; the rest is `hispanic`, below. Cost: 3,903 s (3,902-3,923) and 83.1 GB, 3%
and 9% above the same arm at 1e-12. **This is the full-joint arm the paper reports**
(`docs/figures/campaign.py::FULL_JOINT`), so that it replicates the DAS; the tables above are the same
arm at our 1e-12. `BAR_CONV_TOL` in the environment sets it (`benchmarks/common.py`).

**`hispanic`: a defect in the DAS's reader, not in its algorithm** (found 2026-10-04, full write-up
in `das1940/HANDOFF.md` §5.18). The four pairs with `hispanic` sit at 0.00127-0.00130 at the root in
every DAS run, against 30-60% spread run to run on the other pairs. The reader keeps IPUMS `RACE` as
1..6 in an axis declared 0..5, and the histogram index is computed by `nprmiSQLString`
(`programs/reader/sql_spar_table.py:47`), a ravel_multi_index in SQL with no range check. Race comes
right after `hispanic` in the histogram, so race 6, Other Asian or Pacific Islander, carries into it:
those persons land in race index 0 with `hispanic` inverted, and the Hispanic ones also one year
older. What looked like a rotation of the race axis is the whole histogram shifted by two cells.
From the truth, 168,885 non-Hispanic minus 4,509 Hispanic Other Asian/PI gives **164,376 Hispanics
too many (+7.46%)**, a TVD of 0.00124 in each `hispanic` pair at the root by itself. The releases left
on the server confirm it: `CENRACE 01` with `CENHISP 2` holds 646 in Oregon and 404 in Alaska, against
558 and 406 non-Hispanic Other Asian/PI in the truth; a rotation would have put them under
`CENHISP 1`. `das_release.py` undoes the race labels but not the carry. **Not corrected, by
decision** (2026-10-04): it is the release's implementation, the comparison reports the DAS as it
runs, and the paper describes the defect with a `% TODO` to settle with the advisors. Correcting it
would need the national releases, which are deleted after scoring, so it means rerunning the DAS
with the exact inverse in `das_release.py`.

> **Before the unit bounds** (2026-09-27, `exp1_sin_unidades/`). Cost: full-joint 65.4 min (4.1x),
> 59.9 GB; marginal 29.4 min (9.2x), 22.3 GB. 2-way TVD national / STATEFIP / COUNTY / SUPDIST /
> ENUMDIST: full-joint 0.000068 / 0.001933 / 0.026026 / 0.025614 / 0.183567 (1.02-1.03x the DAS
> below the states); marginal 0.000266 / 0.001625 / 0.023803 / 0.023473 / 0.159334 (0.89x at the
> leaves). L1 at the leaves: full-joint 386.0, marginal 296.3.

> **The generalised-least-squares fix, `6f53051`.** Before it, the full-joint arm was at **2x** the
> DAS's TVD at County, Supdist and Enumdist (0.0527 / 0.0517 / 0.2887 against 0.0256 / 0.0251 /
> 0.1787). The DAS weights its L2 objective by 1 / variance
> (`programs/optimization/geo_optimizers.py:125`); ours weighted every query term at 1.0, which is
> correct only when every block is equally noisy — and this workload splits the level's budget
> .2 / .5 / .05 / .05 / .05 / .05 / .1 across seven blocks. `TopDown._build_query_blocks` now derives
> `query_weights` by asking the mechanism for the variance, because PureDP has Var proportional to
> Delta^2 (b = Delta / epsilon) while zCDP has Var proportional to Delta (sigma = sqrt(Delta / 2
> rho)); hardcoding the square would have squared the weight spread again under zCDP. Reproduced on
> two machines: a 12-worker laptop run gave 0.000069 / 0.001925 / 0.026039 / 0.025645 / 0.183594
> against the server's median of 5 at 0.000068 / 0.001933 / 0.026026 / 0.025614 / 0.183567.

### Step 2.1 — composition

100 runs (5 datasets x 4 compositions x 5 rounds), rho = 1, 20 workers, all `ok`. Median 2-way
TVD at the **leaf** level, which is where the released microdata lives:

| dataset | exponential | uniform | proportional | **sqrt** |
|---|---|---|---|---|
| adult | 0.17911 | **0.16243** | 0.19009 | 0.17464 |
| chilean_census | 0.23719 | 0.25565 | 0.24980 | **0.23633** |
| ipums_1940 | 0.07720 | 0.09626 | 0.07306 | **0.07081** |
| sinasc | 0.12894 | 0.15289 | 0.12936 | **0.12335** |
| spanish_census | 0.08542 | 0.09092 | 0.08724 | **0.08336** |

**`sqrt` beats `exponential`, the runner's default, on all five**, and beats everything on four.
`proportional` is the worst almost everywhere. The spread across the 5 rounds is 0.1-0.5% on the
four large datasets and the min-max ranges of the compositions are **disjoint**, so the ordering
is not the noise draw: leaf TVD averages over thousands of nodes, which cancels it. Adult is the
only one with real spread (3-5%), consistent with its 15 leaf nodes and 48,842 records. Time and
memory are within 3% across compositions, so this decision is utility only.

The trade-off `sqrt` makes is explicit: **`uniform` wins at the top levels and loses at the
leaves.** Worst case each way, IPUMS: `uniform` is 93% better at STATEFIP (51 nodes) and 26%
worse at ENUMDIST (152,009 nodes). That is exactly what `level_rhos` predicts — `sqrt` minimizes
sum_i nodes_i / rho_i, and the leaves hold almost every node. On "worst level" `sqrt` wins 4 of
5, on the unweighted mean over levels 3 of 5.

**Adult is the exception and it is informative, not noise**: `uniform` wins there at *every*
level, leaves included, although `sqrt` gives the leaves 40% of the budget against uniform's 25%.
The consistency chain is the likely reason — adult already carries TVD 0.039 at the national node
(against 0.0004 for IPUMS), and a noisy parent is inherited downwards however much rho the child
gets. Adult is the degenerate case: 48,842 records, an artificial age-group hierarchy, 15 leaf
nodes. If that reading is right, the optimal composition tracks the records-per-node ratio.

**Where the error lives (`tvd2_inside` / `tvd2_outside`, with `sqrt`).** The ratio decays
monotonically with depth on all five datasets, from 16-35x at the national node to ~1x or less at
the leaves:

| level | records/node (Chile) | inside | outside | ratio |
|---|---|---|---|---|
| national | 17,574,003 | 0.001964 | 0.032301 | 16.4x |
| REGION | 1,098,375 | 0.010984 | 0.037852 | 3.4x |
| PROVINCIA | 313,821 | 0.035391 | 0.057967 | 1.6x |
| COMUNA | 50,792 | 0.062273 | 0.073445 | 1.2x |
| DC | 5,734 | 0.182167 | 0.174040 | 1.0x |
| ZC_LOC | 1,157 | 0.259453 | 0.233624 | 0.9x |

Two different error sources: **outside** a bag the error is structural (unmeasured association)
and nearly independent of the noise, so it dominates where the noise is small; **inside** it is
noise alone, and it grows as rho and records per node fall. They cross at roughly 5,000
records/node. Below that the inside error can be the *worse* of the two (IPUMS ENUMDIST: 0.0887
inside against 0.0440 outside), because bags are high-dimensional while the outside pairs are
2-way and average out. This qualifies the earlier "the error falls outside the bags" finding,
which was measured at the national node.

`sqrt` wins in **both** classes on 4 of 5 datasets, so the choice is not trading one against the
other.

### Step 2.2 — budget

125 runs (5 datasets x 5 budgets x 5 rounds) with `sqrt`, 20 workers, all `ok`. 25 of them are
the rho = 1 runs of step 2.1, reused untouched. Median 2-way TVD at the **leaf** level:

| dataset | rho 0.1 | rho 0.5 | rho 1 | rho 2 | rho 5 | over the 50x | exponent | per doubling |
|---|---|---|---|---|---|---|---|---|
| adult | 0.25214 | 0.19444 | 0.17464 | 0.15326 | 0.13299 | 1.90x | -0.164 | -10.7% |
| chilean_census | 0.29657 | 0.25348 | 0.23633 | 0.22004 | 0.19942 | 1.49x | -0.101 | -6.8% |
| ipums_1940 | 0.12363 | 0.08417 | 0.07081 | 0.05928 | 0.04609 | 2.68x | -0.252 | -16.0% |
| sinasc | 0.19287 | 0.14108 | 0.12335 | 0.10812 | 0.09192 | 2.10x | -0.189 | -12.3% |
| spanish_census | 0.11712 | 0.09127 | 0.08336 | 0.07711 | 0.07096 | 1.65x | -0.128 | -8.5% |

The exponent is the log-log slope of TVD against rho between 0.1 and 5; "per doubling" is
2^exponent - 1. **Noise alone would give -0.5, i.e. -29% per doubling.** Every dataset is between
a third and a half of that, so most of the leaf error is not noise. Spread across the 5 rounds is
0.05-0.6% on the four large datasets and 2.6-7% on adult, and the min-max ranges of neighbouring
budgets are **disjoint** everywhere, so the slope is not the noise draw.

**There is no knee.** Within a level the improvement per doubling is the same from 0.1 to 0.5 as
from 2 to 5 — the curves are straight in log-log. So the budget cannot be chosen by where the
curve bends, because it does not bend.

**The elasticity is not monotone in depth.** Per-doubling drop in 2-way TVD, level by level (the
best level of each dataset in bold):

| dataset | national | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| adult | -12.3% | **-12.5%** (AGE_20) | -10.3% | -10.7% | | |
| chilean_census | -1.8% | -5.4% | -8.7% | **-9.6%** (COMUNA) | -7.2% | -6.8% |
| ipums_1940 | -6.3% | **-20.7%** (STATEFIP) | -19.9% | -19.8% | -16.0% | |
| sinasc | -1.8% | -3.0% | -6.9% | **-12.5%** (REGSAUD) | -12.3% | |
| spanish_census | -0.7% | -4.5% | **-8.5%** (CMUN) | | | |

It rises from the root, peaks in the middle of the tree and falls again at the leaves. Above,
the error is structural — pairs of columns no bag measures, which rho does not touch: at Spain's
national node 50x the budget moves TVD from 0.042046 to 0.040455, **4%**. Below, the nodes are so
small (1,156 records per node at Chile's leaves, 459 at SINASC's) that the noise has already
destroyed the marginals and more budget does not rescue them. The peak sits where the node still
has signal and is already small enough for the noise to matter. Same depth story as the
inside/outside split in step 2.1, measured a different way.

IPUMS is the counter-example that confirms the reading: 6 columns in 2 bags, almost no
out-of-bag error to cap it, and at STATEFIP it reaches -20.7% per doubling — the closest to the
theoretical -29% anywhere in the study. Spain, with 42 bags, is the opposite end at -0.7%.

**Cost rises with the budget.** From rho 0.1 to rho 5 the estimation phase grows 7.9% on Chile
(1,489.5 s -> 1,607.0 s) and the largest worker's RSS grows 43% on IPUMS (2,195 MB -> 3,141 MB):
less noise leaves a cleaner support, but more cells survive the pruning in each node. So the
budget is not free in compute, although the effect is far smaller than the utility one.

**Decision: `RHO = 1`, and the argument is not utility.** With a flat, constant elasticity no
budget in the sweep wins: going from 1 to 2 buys 7-16% of leaf TVD for twice the privacy loss,
and dropping to 0.5 costs the same the other way. What `RHO` actually is, is the **operating
point at which experiments 4 to 7 hold the budget fixed** so that structure, coupling and
rounding are what vary; the paper's answer to "which budget" is this sweep, not this constant.
rho = 1 is the centre of the grid, leaves both ends reportable as a sensitivity analysis, and is
the value step 2.1 already ran, so its 25 runs are reused as the reference runs of step 2.3. The
robustness claim to make in the paper is the measured one: the conclusions do not depend on the
operating point, because the elasticity is flat and has no knee.

For stating the guarantee in (epsilon, delta) terms, the generic zCDP conversion
(`cdp_eps` in McKenna's `mechanisms/cdp2adp.py`) gives:

| rho | 0.1 | 0.5 | **1** | 2 | 5 |
|---|---|---|---|---|---|
| epsilon at delta 1e-9 | 2.72 | 6.47 | **9.52** | 14.15 | 24.41 |
| epsilon at delta 1e-10 | 2.88 | 6.84 | **10.03** | 14.87 | 25.54 |

That is the generic bound. The DAS computes a tighter epsilon from the full per-geolevel and
per-query allocation (`zCDPEpsDeltaCurve`, `programs/engine/budget.py:191`), so its published
epsilon is lower than this table would give for the same rho; say which bound is used before
comparing. For reference, the DHC persons production config carries `global_scale = 254/485`,
and rho = 1 / global_scale^2, so **the DAS ran at rho = 3.646** — between our 2 and our 5.

> **None of this applies to the 1940 comparison of Part 1.** That run sets
> `privacy_framework: pure_dp`, where the engine reads `global_scale` as 1 / epsilon rather than
> 1 / sqrt(rho) (`budget.py:224`), so `global_scale: 1/4` is epsilon = 4 flat, with no conversion
> and no delta. Verified in `ipums_1940_local.ini`; see step 1.1.

### Step 2.3 — experiments 4, 5 and 7

**Experiment 7 (global rounding MIP against the junction-tree sweep).** 50 runs planned, 5 datasets
x 2 rounding methods x 5 rounds, all at rho = 1 with `sqrt`, 20 workers. The 25 sweep runs are the
reference runs of step 2.2, reused untouched. Of the 25 MIP runs, **20 are `ok` and the 5 of Spain
are not**. Seconds are initialize + estimation, median of the rounds with min-max underneath:

| dataset | bags | sweep | MIP | MIP / sweep | leaf TVD, MIP / sweep | worker RSS |
|---|---|---|---|---|---|---|
| ipums_1940 | 2 | 1,932 s (1,928-1,938) | 1,954 s (1,950-1,959) | 1.01x | 0.9938 | 2,766 -> 1,719 MB |
| adult | 7 | 3 s (3-3) | 3 s (3-3) | 1.01x | 0.9758 | 249 -> 254 MB |
| sinasc | 9 | 44 s (44-44) | 155 s (92-211) | 3.54x | 0.9968 | 331 -> 682 MB |
| chilean_census | 21 | 1,564 s (1,559-1,565) | 2,820 s (**2,520-15,317**) | 1.80x | 0.9980 | 1,023 -> 1,903 MB |
| spanish_census | 41 | 311 s (297-316) | **does not finish** | - | - | 1,871 -> - |

**The MIP is slightly better and hugely less predictable, and that is the finding.** On utility it
wins everywhere, by 0.2% to 2.4% of leaf TVD — the same order as the 1.2% measured earlier on the
27-column instance, and far below the 7-16% that doubling rho buys (step 2.2). On time the median
ratio ranges from a tie (IPUMS, adult) to 3.5x (SINASC). But the medians hide the real difference:

- **The sweep's runtime is flat.** max/min across the 5 rounds is 1.0x on four datasets and 1.1x
  on Spain. Same configuration, different noise draw, same seconds.
- **The MIP's is not.** SINASC 2.3x (92 s to 211 s) and **Chile 6.1x: 2,520 s in one round and
  15,317 s in another**, same dataset, same budget, same structure. Round 4 spent over 4 h, which
  means it hit the 14,400 s `TimeLimit` on at least one node group and was saved only because that
  model had an incumbent to return.
- **Spain does not finish at all**, with 41 bags — the most complex junction tree of the five. Two
  attempts, both `error` with returncode 1: at `TimeLimit` 1,200 s (1,330 s wall) and again at
  14,400 s (14,532 s wall, from the orchestrator log — a run that dies before writing its
  `<name>.json` records no `seconds`). The message is the rounding MIP's, not the QP's: *"hit the
  TimeLimit ... with no incumbent at all"*, at node 60 of Chile in the first case.

The ordering by bags is suggestive and the extremes fit — IPUMS with 2 bags ties the sweep, Spain
with 41 does not finish — but SINASC (9 bags, 3.54x) is worse than Chile (21 bags, 1.80x), so bag
count alone does not predict it. That is consistent with what the rounding notes already record:
no metric of problem size predicts MIP runtime, and instance-level solver difficulty dominates
every structural trend. The mechanism is the one documented there: each variable sits in its
geographic row **plus one separator row per junction-tree edge incident to its bag**, so more bags
means more columns with 3+ nonzeros of mixed sign, and the matrix drifts further from totally
unimodular.

Memory is the one place the MIP sometimes wins: it doubles the largest worker on SINASC and Chile,
but on IPUMS it uses **38% less** than the sweep (1,719 against 2,766 MB) — 2 bags make a small
MIP, while the sweep still pays for its transports.

**What this buys the paper.** The argument for the sweep is not that it is faster on average: on
IPUMS and adult it is a tie. It is that it is **predictable and it finishes**. The MIP's price is
a 6x runtime spread on the same configuration and a dataset where no microdata comes out at all,
for 0.2-2.4% of TVD.

> `TimeLimit` was raised from 1,200 s to 14,400 s in `benchmarks/common.py` on 2026-09-23, to
> separate "does not finish in 20 min" from "does not finish". Chile needed it: it failed at 1,200 s
> and completed at 14,400 s. Spain did not. Round 1 of adult, SINASC and IPUMS ran before the
> change, at 1,200 s; the limit never bound for them (3 s, 155 s, 1,959 s), so those runs are
> equivalent, but `runs.jsonl` records two different `solver_options` for the same configuration.

**Experiment 4 (3-way TVD by how many pairs of a triple are measured).** The first version read
the reference runs, whose 100 triples were drawn uniformly. Almost no triple is inside a bag, so
the "contained" curve rested on 5 triples on IPUMS, 5 on SINASC, 4 on adult and **none** on Chile
or Spain. Redesigned on 2026-10-04: every contained triple is evaluated, and the rest are sampled
up to 100 per stratum of measured pairs, with a fixed seed and each triple weighted by its stratum
size, so `tvd3` and `tvd3_other` still estimate the mean over all triples. No dataset has a
non-contained triple with all three pairs measured, so the strata are clean:

| dataset | triples | contained | 2 pairs | 1 pair | 0 pairs |
|---|---|---|---|---|---|
| ipums_1940 | 20 | 5 | 6 | 9 | - |
| chilean_census | 2,925 | 14 | 87 | 709 | 2,115 |
| spanish_census | 76,076 | 87 | 333 | 9,561 | 66,095 |
| sinasc | 1,140 | 29 | 72 | 489 | 550 |
| adult | 364 | 10 | 53 | 116 | 185 |

25 runs (5 datasets x 5 rounds) of the reference configuration with `triples=100`, rho = 1,
`sqrt`, sweep, 20 workers, all `ok`. Copied from the server as `runs_exp4.jsonl` and appended to
each `runs.jsonl` locally. The rerun reproduces the reference runs: leaf 2-way TVD within 0.25%
on every dataset. The stratified metrics cost little: 1 s to 399 s per run, against 1 s to 227 s.
3-way TVD, median of 5, at the root and at the leaves:

| dataset | level | contained | 2 pairs | 1 pair | 0 pairs |
|---|---|---|---|---|---|
| ipums_1940 | national | 0.00012 | 0.00119 | 0.00191 | - |
| | ENUMDIST | 0.12314 | 0.16021 | **0.05784** | - |
| chilean_census | national | 0.00583 | 0.03746 | 0.04273 | 0.07117 |
| | ZC_LOC | 0.47103 | 0.43993 | 0.33690 | **0.33256** |
| spanish_census | national | 0.00136 | 0.02527 | 0.06282 | 0.08730 |
| | CMUN | **0.05151** | 0.08093 | 0.12244 | 0.14826 |
| sinasc | national | 0.00094 | 0.01040 | 0.02055 | 0.03038 |
| | CODMUNRES | **0.12667** | 0.15294 | 0.17927 | 0.23174 |
| adult | national | 0.03074 | 0.05480 | 0.09151 | 0.11403 |
| | AGE_5 | **0.16265** | 0.19427 | 0.26410 | 0.30635 |

**Near the root the error follows the coverage on all five datasets**: contained < 2 pairs <
1 pair < 0 pairs, with disjoint min-max ranges over the 5 rounds. **Measuring two of the three
pairs does not recover the triple**: at the root that stratum is still 1.8x (adult) to 19x (Spain)
worse than the contained one. The unmeasured pair's association given the third attribute is set
by the random coupling. The gaps are smallest on adult, where the noise already dominates at the
root (48,842 records), as with pairs in experiment 5.

**At the leaves the datasets split, and the split is experiment 5's.** On Spain, SINASC and adult
the ordering holds at every level, ranges disjoint throughout. On IPUMS and Chile it inverts at the
finest levels and the contained and 2-pair triples become the *worst*: from COUNTY down on IPUMS,
from COMUNA down on Chile (at REGION, 1 pair already edges out 2 pairs, 0.0548 against 0.0575).
Round-to-round spread there is 0.1-0.4%, so the inversion is not noise. These are exactly the two
datasets where experiment 5's unmeasured/measured ratio for pairs falls below 1 at the leaves (0.5
and 0.9). It is the same crossing seen in triples: the more of a triple lies inside the bags, the
more of its error is noise of high-dimensional bags, which dominates in small nodes. That
agreement between two independent metrics is what makes experiment 4 more than a copy of
experiment 5. Figure: `python -m docs.figures.fig_levels`, which also prints the full table.

**Experiment 5** ran nothing of its own, as designed: its 5 configurations are the reference runs,
already `ok`. It reads `<name>_metrics.json` (every pair with its U in both files), which
is **not** in `runs.jsonl` — `workload` is empty there for four of the five datasets — so those 5
files have to be copied from the server before that analysis can be written.

### Step 2.4 — depth

20 runs (4 datasets x 5 rounds) at rho = 1 with `sqrt` and the sweep, all `ok`, with **8 workers**:
the default of `orchestrate.py` until 2026-10-01, while the full-depth runs they are compared with
used 20. Utility is unaffected (step 2.5 reran the same situation and moved less than 0.5%); the
times below are not comparable, and the cut runs are the slower side of it. Spain's
arm is its own step 2.2 reference, which is already 3 levels. Median of 5 throughout, and every
full-against-cut comparison below has **disjoint min-max ranges** over the 5 rounds unless said
otherwise.

**Every dataset at the same depth.** Per-level TVD at 3 levels:

| dataset | records | W | bags | level | nodes | records/node | tvd2 | tvd3 |
|---|---|---|---|---|---|---|---|---|
| adult | 48,842 | 27,188 | 7 | national | 1 | 48,842 | 0.04223 | 0.09476 |
| | | | | AGE_20 | 4 | 12,210 | 0.13016 | 0.21552 |
| | | | | AGE_10 | 8 | 6,105 | 0.16750 | 0.26065 |
| sinasc | 2,561,858 | 6,844 | 9 | national | 1 | 2,561,858 | 0.01181 | 0.02625 |
| | | | | REGIAO | 5 | 512,372 | 0.01247 | 0.02768 |
| | | | | UF | 27 | 94,884 | 0.01505 | 0.03293 |
| spanish_census | 4,707,186 | 135,481 | 41 | national | 1 | 4,707,186 | 0.04076 | 0.08001 |
| | | | | CPRO | 52 | 90,523 | 0.04872 | 0.09362 |
| | | | | CMUN | 909 | 5,178 | 0.08336 | 0.14900 |
| chilean_census | 17,574,003 | 112,847 | 21 | national | 1 | 17,574,003 | 0.03023 | 0.06134 |
| | | | | REGION | 16 | 1,098,375 | 0.03365 | 0.06914 |
| | | | | PROVINCIA | 56 | 313,821 | 0.04485 | 0.08825 |
| ipums_1940 | 132,404,766 | 4,640 | 2 | national | 1 | 132,404,766 | 0.00046 | 0.00142 |
| | | | | STATEFIP | 51 | 2,596,172 | 0.00138 | 0.00333 |
| | | | | COUNTY | 3,108 | 42,601 | 0.00821 | 0.01342 |

**No dataset has a hump: TVD rises monotonically from the root to the leaves in all five.** That
corrects how step 2.2 gets stated. What peaks at mid-height is the **elasticity** — how much TVD
moves when rho doubles — which is not the same thing as where the error sits. This step is a single
rho and therefore cannot re-test elasticity at matched depth; that would need 80 more runs (4
datasets x the 4 remaining rho x 5 rounds).

**The finding: the budget only improves what the bags measure.** Cutting the tree hands every
surviving level far more rho. Splitting 2-way TVD into pairs **inside** some bag and pairs
**outside** every bag, full depth to 3 levels:

| dataset | level | rho share | inside | outside | total |
|---|---|---|---|---|---|
| adult | national | 10.3 -> 17.2% | 0.02093 -> 0.01633 (**0.78**) | 0.05249 -> 0.04998 (0.95) | 0.93 |
| | AGE_20 | 20.6 -> 34.3% | 0.09685 -> 0.08132 (**0.84**) | 0.15858 -> 0.14432 (0.91) | 0.90 |
| | AGE_10 | 29.2 -> 48.5% | 0.12899 -> 0.11229 (**0.87**) | 0.20313 -> 0.18407 (0.91) | 0.90 |
| sinasc | national | 1.0 -> 11.9% | 0.00060 -> 0.00018 (**0.31**) | 0.01346 -> 0.01491 (**1.11**) | **1.10** |
| | REGIAO | 2.1 -> 26.5% | 0.00116 -> 0.00038 (**0.33**) | 0.01403 -> 0.01570 (**1.12**) | **1.10** |
| | UF | 5.0 -> 61.6% | 0.00402 -> 0.00143 (**0.36**) | 0.01755 -> 0.01869 (**1.06**) | 1.02 |
| chilean_census | national | 0.5 -> 8.0% | 0.00196 -> 0.00057 (**0.29**) | 0.03230 -> 0.03372 (**1.04**) | **1.04** |
| | REGION | 1.9 -> 32.0% | 0.01098 -> 0.00363 (**0.33**) | 0.03785 -> 0.03719 (0.98) | 0.96 |
| | PROVINCIA | 3.6 -> 59.9% | 0.03539 -> 0.01462 (**0.41**) | 0.05797 -> 0.04842 (0.84) | 0.81 |
| ipums_1940 | national | 0.2 -> 1.6% | 0.00006 -> 0.00002 (**0.36**) | 0.00097 -> 0.00112 (**1.16**) | **1.09** |
| | STATEFIP | 1.4 -> 11.2% | 0.00183 -> 0.00075 (**0.41**) | 0.00226 -> 0.00231 (1.02) | 0.69 |
| | COUNTY | 10.9 -> 87.3% | 0.01753 -> 0.00844 (**0.48**) | 0.01362 -> 0.00787 (0.58) | 0.51 |

`tvd2_inside` improves in **all 15 cells**, 0.29x to 0.87x — the noise responding to the extra rho,
as it should. `tvd2_outside` does not follow: it worsens in 5 of the 12 cells and is flat in one
more. And because `outside` dominates `inside` above the leaves, the total follows `outside`:

- **SINASC national**: inside **3.3x better**, outside **11% worse**, total **10% worse** — outside
  starts 22x larger than inside, so it decides the sum.
- **IPUMS national**: inside 0.36x, outside 1.16x, total 1.09x worse, with outside 16x larger.
- The total improves only where `outside` also improves: adult everywhere, Chile at REGION and
  PROVINCIA, IPUMS at STATEFIP and COUNTY (0.51x, with 8x the rho share).

The one comparison to call unchanged is IPUMS STATEFIP `outside` (1.02x), the only overlapping pair
of ranges in the table.

**The mechanism is the coupling, and the leaf sizes say so.** Cutting the tree makes the leaves much
coarser — SINASC 5,570 to 27 nodes, Chile 15,195 to 56, IPUMS 152,009 to 3,108 — so each separator
group holds far more records and the uniform record-to-cell pairing has correspondingly more freedom
in the pairs no bag covers. That degree of freedom is the one `CLAUDE.md` describes as "the one
degree of freedom the marginals do not pin down". Adult is the control that fits: its leaves barely
move, 15 to 8 nodes, and it is the only dataset where `outside` improves at every level.

**What this buys the paper.** It sharpens "the utility is decided by the structure, not the noise"
into something stronger and testable: the two error sources respond to shortening the tree in
**opposite directions**. More budget per level buys the measured marginals and can still make the
released microdata worse overall, because the unmeasured cross-tabs are set by a coupling that gets
freer as the leaves get bigger. Above the leaf level, `tvd2_outside` is where essentially all of the
error is.

**Cost: time collapses, memory can go up.** Median seconds (initialize plus estimation) and
`memory_mb`, full depth to 3 levels:

| dataset | levels | leaf nodes | seconds | main MB | worker MB |
|---|---|---|---|---|---|
| adult | 4 -> 3 | 15 -> 8 | 3.5 -> 2.7 | 324 -> 311 | 249 -> 246 |
| sinasc | 5 -> 3 | 5,570 -> 27 | 43.8 -> **4.3** | 4,225 -> 1,913 | 331 -> **608** |
| chilean_census | 6 -> 3 | 15,195 -> 56 | 1,563.8 -> **69.8** | 4,188 -> **8,754** | 1,023 -> **5,484** |
| ipums_1940 | 5 -> 3 | 152,009 -> 3,108 | 1,932.2 -> **93.9** | 14,426 -> 12,438 | 2,766 -> 1,027 |

Chile and IPUMS are 22x and 21x faster, with 8 workers against 20, so a lower bound. But **Chile's peak memory doubles and its largest worker
grows 5.4x**, because the model per node group scales with `n_children` and a shallower tree gives
each group more children — REGION to PROVINCIA is 56 children of 313,821 records each. SINASC moves
the same way. IPUMS goes the other way, because its wide groups at full depth were Supdist to
Enumdist.

> **Measured and deliberately left out of the paper.** Define SNR as (mean true count per measured
> cell) / sigma at that level. Over the 15 (dataset, level) points, spanning six orders of magnitude
> in SNR, log `tvd2_inside` regresses on log SNR with slope **-0.63 and R2 = 0.80**. It is not
> quotable as it stands: **IPUMS COUNTY sits 6x above the trend** (SNR 6.06, inside 0.00844, against
> SINASC UF at SNR 3.63 with 0.00143), most plausibly because with only 2 bags of very unequal width
> the count per cell averaged over the concatenation is a poor summary. Redo it with a per-bag SNR
> before using it.

### Step 2.5 — structure

135 runs (27 configurations x 5 rounds) at rho = 1 with `sqrt` and the sweep, 20 workers, all
`ok`. Median of 5 throughout, and every comparison below has **disjoint min-max ranges** over the 5
rounds unless said otherwise. Coverage is the fraction of attribute pairs inside some bag.

> **Rerun 2026-10-03.** The first pass ran the derived structures with 8 workers, the default of
> `orchestrate.py` until 2026-10-01, while the declared ones are the 20-worker runs of steps 2.1-2.3,
> so its times compared 8 workers against 20. The 115 derived runs were redone at 20. Utility did not
> move: within 0.5% at every level on IPUMS, SINASC and Spain, 1-4% on adult, which is its normal
> round-to-round spread. Time did: the derived structures got 1.6-1.7x faster on IPUMS and SINASC,
> and unchanged on Spain and adult. Memory is unchanged, since main plus largest worker does not
> depend on the number of workers. The one conclusion that was an artefact is below, under cost.

**What the tree returned.** Merges are respected on all four datasets. Splits are exact on adult
and IPUMS only: SINASC 17 -> 14, 21 -> 17, 21 -> 18 bags; Spain 64 -> 62, 77 -> 71, 79 -> 70,
80 -> 70. Spain's `blocks_s3` and `blocks_s4` come out as the **same structure** (70 bags,
W 76,020 against 75,999, tvd2 0.06112 both): the family saturates there, and a figure should show
one of them. IPUMS `das_m1` is the full joint in one bag, identical to the declared `joint`.

**Whole-tree utility and cost.** Mean of `tvd2` over the levels; seconds are initialize plus
estimation; GB is `memory_mb` main plus largest worker:

| dataset | structure | bags | coverage | W | Delta | tvd2 | seconds | GB |
|---|---|---|---|---|---|---|---|---|
| adult | `blocks_s2` | 13 | 14.3% | 2,884 | 26 | **0.12477** | 1.5 | 0.54 |
| | `blocks_s1` | 11 | 18.7% | 3,384 | 22 | 0.12544 | 1.6 | 0.54 |
| | `blocks` | 7 | 23.1% | 27,188 | 14 | 0.13785 | 3.5 | 0.56 |
| | `blocks_m1` | 6 | 26.4% | 40,705 | 12 | 0.17435 | 4.7 | 0.59 |
| | `blocks_m2` | 5 | 28.6% | 82,085 | 10 | 0.19923 | 7.6 | 0.64 |
| | `blocks_m3` | 4 | 31.9% | 503,960 | 8 | 0.28125 | 37.5 | 1.19 |
| sinasc | `blocks_s3` | 18 | 11.6% | 1,919 | 36 | 0.04036 | 37.8 | 4.43 |
| | `blocks_s2` | 17 | 13.2% | 2,135 | 34 | 0.03990 | 37.6 | 4.46 |
| | `blocks_s1` | 14 | 17.9% | 3,680 | 28 | 0.03856 | 40.9 | 4.44 |
| | `blocks` | 9 | 21.1% | 6,844 | 18 | **0.03798** | 43.8 | 4.45 |
| | `blocks_m1` | 8 | 23.2% | 18,204 | 16 | 0.03849 | 79.6 | 4.38 |
| | `blocks_m2` | 7 | 24.7% | 31,536 | 14 | 0.03912 | 123.8 | 4.45 |
| | `blocks_m3` | 6 | 27.9% | 57,009 | 12 | 0.04174 | 198.2 | 4.55 |
| | `blocks_m4` | 5 | 29.5% | 81,160 | 10 | 0.04496 | 273.8 | 4.66 |
| spanish_census | `blocks_s4` | 70 | 3.2% | 75,999 | 140 | 0.06112 | 240.2 | 9.45 |
| | `blocks_s3` | 70 | 3.3% | 76,020 | 140 | 0.06112 | 254.7 | 9.47 |
| | `blocks_s2` | 71 | 3.3% | 76,029 | 142 | 0.06419 | 250.4 | 9.46 |
| | `blocks_s1` | 62 | 3.9% | 77,808 | 124 | 0.06191 | 213.1 | 9.40 |
| | `blocks` | 41 | 4.6% | 135,481 | 82 | **0.05762** | 310.6 | 9.51 |
| | `blocks_m1` | 40 | 4.6% | 135,567 | 80 | 0.05772 | 314.8 | 9.56 |
| | `blocks_m2` | 39 | 4.7% | 139,806 | 78 | 0.05780 | 320.0 | 9.57 |
| | `blocks_m3` | 38 | 4.8% | 155,871 | 76 | 0.05792 | 330.5 | 9.66 |
| | `blocks_m4` | 37 | 4.9% | 161,614 | 74 | 0.05804 | 347.4 | 9.61 |
| ipums_1940 | `das_s2` | 5 | 33.3% | 496 | 10 | 0.01805 | 2,930.0 | 14.59 |
| | `das_s1` | 4 | 46.7% | 736 | 8 | **0.01650** | 2,513.3 | 15.18 |
| | `das` | 2 | 60.0% | 4,640 | 4 | 0.02099 | 1,932.2 | 16.79 |
| | `das_m1` | 1 | 100.0% | 44,544 | 2 | 0.02416 | 5,625.2 | 22.16 |

**More coverage is not better.** On adult, splitting improves tvd2 9% and takes less than half the
time, while merging to 31.9% coverage doubles it and costs 11x the time. On IPUMS the optimum is
**interior**, at 46.7%: better than the declaration and better than the full joint. SINASC and
Spain are optimal at their declaration. Spain's merges are within 0.7% of each other, because its
41 bags are fixed by the constraint scopes much as Chile's are (coverage moves 4.6% -> 4.9%).

**The finding: the best structure changes down the tree, and it follows records per node.** Per
level, with records per node:

| dataset | level | records/node | best | its coverage | runner-up |
|---|---|---|---|---|---|
| ipums_1940 | national | 132,404,766 | `das_m1` 0.00037 | **100.0%** | `das` 0.00042 |
| | STATEFIP | 2,596,172 | `das_s1` 0.00196 | 46.7% | `das` 0.00200 |
| | COUNTY | 42,601 | `das_s1` 0.00951 | 46.7% | `das_s2` 0.01028 |
| | SUPDIST | 41,312 | `das_s1` 0.00942 | 46.7% | `das_s2` 0.01022 |
| | ENUMDIST | 871 | `das_s1` 0.06061 | 46.7% | `das_s2` 0.06364 |
| sinasc | national | 2,561,858 | `blocks` 0.01075 | **21.1%** | `blocks_m1` 0.01111 |
| | REGIAO | 512,372 | `blocks` 0.01132 | 21.1% | `blocks_m1` 0.01177 |
| | UF | 94,884 | `blocks` 0.01470 | 21.1% | `blocks_m1` 0.01537 |
| | REGSAUD | 5,693 | `blocks_s1` 0.02963 | **17.9%** | `blocks` 0.02996 |
| | CODMUNRES | 460 | `blocks_s1` 0.12254 | 17.9% | `blocks_s2` 0.12294 |
| spanish_census | national | 4,707,186 | `blocks_m2` 0.04068 | **4.7%** | `blocks_m4` 0.04069 |
| | CPRO | 90,523 | `blocks_m1` 0.04871 | 4.6% | `blocks_m2` 0.04871 |
| | CMUN | 5,178 | `blocks` 0.08336 | **4.6%** | `blocks_m1` 0.08368 |
| adult | national | 48,842 | `blocks_s1` 0.04284 | 18.7% | `blocks` 0.04521 |
| | AGE_20 -> AGE_5 | 12,210 -> 3,256 | `blocks_s2` | 14.3% | `blocks_s1`, tied at every level |

Within every dataset, the coverage of the best structure **never increases** going down the tree.
IPUMS crosses from the joint to `das_s1` between the national node and the states; SINASC crosses
from `blocks` to `blocks_s1` between UF (94,884 per node) and REGSAUD (5,693). That is not the
level where SINASC's measured and unmeasured pairs swap: in experiment 5 they never do (1.4 at the
leaves), so the two crossings are different quantities; Spain crosses
from a merge to its declaration between the nation and CMUN. Adult never has enough records for
coverage to pay: a split is the best at every level, `blocks_s1` at the root and `blocks_s2` below,
the two overlapping below the root. SINASC's two lowest levels overlap with their runner-up too
(REGSAUD 0.02955-0.02982 against 0.02974-0.03011). Two of the close ones are still disjoint: IPUMS
STATEFIP (0.00194-0.00196 against 0.00198-0.00202) and the Spain national merges against the
declaration (0.04066-0.04072 against 0.04075-0.04079). At CPRO, Spain's merges and its declaration
are tied.

The mechanism: the error has a term set by coverage (what no bag measures is filled in by the
coupling), which does not depend on the node's size, and a noise term that grows with the cells
per bag and with sqrt(bags) through Delta and shrinks with the records in the node. High up, the
noise is negligible and coverage decides; at the leaves the noise decides and the smallest tables
win. Records per node fall by five orders of magnitude on IPUMS, so **no single structure is best
at every level**.

**Cost follows the width in one direction only.** Merging is slower on every dataset: SINASC
`blocks_m4`, with 12x the cells, takes 6.3x the time of the declaration, and the IPUMS joint 2.9x.
Splitting is faster on SINASC (`blocks_s1` 0.93x), Spain (0.69x) and adult (0.46x), but not on
IPUMS, where `das_s1` has 6x fewer cells than the declaration and is **1.3x slower** (2,513 s
against 1,932 s), and `das_s2` 1.5x: many bags cost separator rows and transports in the sweep.
The first pass read "an interior minimum at the declaration on IPUMS and SINASC" (2.1x and 1.6x
slower splits); on SINASC that was the 8 workers. Memory barely moves except where W does: IPUMS
22.2 GB for the joint against 14.6 GB for `das_s2`, adult 1.19 GB against 0.54 GB.

**What this buys the paper.** It answers experiment 6 with a mechanism rather than a ranking: the
choice of marginals is a bias-variance trade-off whose balance point moves with the records per
node, so it is set by the depth of the hierarchy as much as by the data. The IPUMS family runs from
the marginal pipeline to the full-joint pipeline, so the same figure also says where each pipeline
wins.
