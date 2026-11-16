# Final server benchmarking

## Protocol

All 30 registered functions, 20 optimizer seeds (0 through 19), one common
reproducible function instance per function/dimension, instance seed 42:

| Dimension | Population | Exact objective evaluation budget per trial |
|---:|---:|---:|
| 30 | 30 | 30,000 |
| 300 | 90 | 300,000 |
| 600 | 180 | 600,000 |
| 1000 | 300 | 1,000,000 |

At 30D: Original PSO, PSO-m, PSO-RIW, HPPSO, GA-MPC, GWO, SHADE, CSA, CMA-ES.
At 300D/600D/1000D: the same list with Sep-CMA-ES replacing CMA-ES.
This gives 21,600 independent trials and at most 10,422,000,000 objective calls.
The supplied final JSON is the authoritative population and seed schedule.
All functions use the requested dimension, including Michalewicz.

Only the objective-call budget terminates successful runs. Initialization,
reevaluations inside selection/sorting, and every attempted objective call are
counted. The final finite candidate is saved, then a budget interrupt stops the
algorithm immediately. This handles a partial final population with no extra
objective calls. No fitness target, wall-time limit, or practical iteration
limit is used. Existing algorithm APIs receive an unreachable guard of B+1
iterations, since every iteration makes at least one evaluation. GWO and CSA
use evaluations consumed/B for their schedules instead of this guard. CMA
fitness targets are disabled and their internal evaluation cap is unreachable
before the wrapper's exact cap.

Numerical exceptions and non-finite objectives still terminate that trial as a
recorded failure. They do not count as successful budget completion. No automatic
restarts, fitness clipping, or replacement penalties hide such failures. The
previous full CMA-ES numerical issue on Powell Sum has not been silently patched.

Shifted bounds, objective scales, new function definitions and structured DCT
rotations are documented in additional_benchmarks.md and numerical_stability.md.
Schwefel 2.22 returns ln(1+f); those results must be labelled on that scale.
The classical 20 retain their current implemented definitions. This runner does
not independently establish the validity of every original manuscript formula.
The 20 seeds randomize optimizers; they are not 20 different function instances.

## Install on a Linux server

Transfer this working checkout, including the new uncommitted files, to the
server. A fresh GitHub clone alone will not contain local changes until pushed.
From the repository root, with Python >=3.10:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

If uv is already installed, `uv sync` also installs the project. The optional
CEC2011/minionpy extra is not required. NumPy, SciPy, pandas and matplotlib are
already declared dependencies. The runner works without a GUI or browser and
sets BLAS/OpenMP thread counts to one per worker before importing NumPy.

## Run locally on one server

First validate the installation with the separate tiny smoke configuration:

```bash
python -u -m experiments.final_benchmark --config experiments/configs/final_smoke.json --output final_smoke --workers 2
python -m experiments.final_benchmark --output final_smoke --report
```

This checks all 30 functions at 6D with two seeds and only 18 calls per trial.
It is a program check, not final scientific results. Use a different output
folder for the real run:

```bash
mkdir -p logs
nohup python -u -m experiments.final_benchmark --config experiments/configs/final_benchmark.json --output final_results --workers 8 > logs/final.log 2>&1 &
tail -f logs/final.log
```

Choose workers to match allocated physical CPU cores and memory. Each worker
runs independent trials. Wall times measure optimizer initialization, objective
evaluation and instrumentation, excluding instance construction and serialization.
They are elapsed times under the chosen concurrency/load, not pure optimizer
CPU overhead. Use the same worker/core allocation when comparing timing studies.

After the run finishes:

```bash
python -m experiments.final_benchmark --output final_results --report
```

No full benchmark is started by a report command. Report generation is a separate
headless process and can be repeated. Prefer running it after all shards finish.

## SLURM arrays / multiple nodes

A ready template is `experiments/final_benchmark.slurm`. It assigns 16 disjoint
shards with four worker processes per array task. Adjust `--partition`, memory,
account, and wall-time settings for your cluster before submission:

```bash
mkdir -p logs
sbatch experiments/final_benchmark.slurm
```

All array tasks must use the SAME shared output directory, identical config,
source files and Python/dependency environment. Keep the number of shards (16)
constant when resuming; worker count may change. Each trial's deterministic
ordinal assigns it to exactly one shard. Never run the same shard twice at once.
Per-shard leases reject duplicate concurrent launches. The shared filesystem
must support atomic rename and hard links (e.g. normal POSIX/Lustre/NFS setups).
A completed plan is published atomically so simultaneously starting array tasks
do not observe half-written configuration files.

To run shards manually on separate allocated nodes:

```bash
# On node/task 0:
python -u -m experiments.final_benchmark --output final_results --workers 4 --shards 16 --shard 0
# On another node/task, use its unique index 1, 2, ..., 15.
```

There is no web server or port to coordinate across nodes. Progress is printed
per completed trial and saved in progress-shard-N.json. Use `tail -f` or your
scheduler's log viewer. Very long trials produce their result only on completion.

## Resume, preemption and failures

Rerun exactly the same command/output/config/shard assignment. Successfully
saved trials are skipped; completed failed trials are also skipped by default
so resuming does not silently select a more favorable repeat. Each trial is
saved by atomic replacement in its own compressed JSON file. A killed partial
trial has no completed file and reruns from its seed; there is no mid-generation
optimizer checkpoint. Other completed trials survive a killed worker.

A clean exit removes the shard lease. After forced termination, a stale file
`.shard-N.lock` may remain. Inspect its host/PID, verify the old job and all its
workers have stopped on that host, then remove ONLY that stale lease before
resubmission. Do not remove leases from running jobs.

To explicitly repeat failures, add `--retry-failed`. Previous failed attempts
are preserved in an attempts subfolder. Record this choice in the analysis;
repeated stochastic seeds with changed code must instead use a NEW output folder.

The plan records a SHA-256 fingerprint of the config, runner, benchmarks and
algorithm source. Source/config changes are rejected for an existing output.
This prevents silently mixing different protocols. Do not use this runner to
resume the earlier smoke/iteration-based experiment directories.

## Files and convergence plots

* plan.json: configuration, functions, algorithm selection, shard layout,
  suite version, code/config fingerprint.
* definitions/{D}d.json: exact common shifts, bounds, rotations, centers,
  biases, objective scales and reference optima where supplied.
* environment-shard-N.json: Python/NumPy/SciPy versions, OS and hostname.
* trials/{D}/{function}/{algorithm}/seed_NNN.json.gz: outcome, actual FE count,
  best score/position, elapsed seconds, errors, and convergence trace.
* progress-shard-N.json: completed assigned trials and invocation failures.

The trace combines log-spaced early and uniform late checkpoints, always
including the first and final evaluations. With trace_points=1000 it stores at
most 2,000 points per trial. It records best-so-far values at actual evaluations,
not iteration numbers, without storing every vector of the population.

`--report` additionally saves:

* runs.csv: per-trial score, evaluation count, time, termination reason and error.
* summary.csv: mean final score, sample standard deviation, mean elapsed seconds,
  valid-seed count, expected-seed count and completeness flag.
* average_ranks.csv: within-function ranks of mean scores, averaged over complete
  function blocks only, with the number of included functions per dimension.
* convergence/{D}d_{function}.png and convergence/{D}d.pdf: all-algorithm
  median best-so-far curves across 20 seeds, shaded interquartile ranges,
  with actual evaluations on the x-axis.
* report_status.json: expected, saved, missing and failed trial counts.

Only function blocks with all required seeds and all applicable algorithms
successfully at the exact budget enter ranks and comparison plots. Missing or
failed blocks remain visible in the CSVs and counts; they are not assigned an
ordinary finite rank. Scores and timings from successful seeds of incomplete
blocks are not reported as complete means. Failure timing/error details remain
in runs.csv. No comparisons pool raw scores across unrelated functions.

Raw best-objective curves are used, with the transformed units explicitly
labelled for Schwefel 2.22. Functions with negative values use a linear y-axis;
positive functions use log axes, with a zero-safe axis where necessary. Shading
is the seed IQR, not a confidence interval. No curves extend beyond the budget.

For publication, also record CPU model, memory, node allocation, dependency
versions, and any scheduler throttling. Complete this run and inspect failures
before making statistical claims or replacing manuscript tables.
