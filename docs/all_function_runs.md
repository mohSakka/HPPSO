# Active function registry update

The all-function runner now uses `classical20-additional-v3`: classical f1–f20
and the ten functions in [additional_benchmarks.md](additional_benchmarks.md).
All old `scaling_*` cases are retired, including duplicate Sphere/Rosenbrock.
Resume checks reject the old suite's plan; full-budget results below remain
archival until revised-suite runs are performed in a new output directory.
The current 30D population/iteration settings and CMA exclusions are retained.

---

# Current 30D rerun: 3,000 iterations

The current experiment uses only 30D, population 30, 3,000 iterations, a maximum
of 90,030 actual objective evaluations including initialization, seeds 0 and 1,
all 25 mathematical functions, and nine algorithms excluding Sep-CMA-ES.
There are 450 full trials. Results are saved separately under
`experiments/results/all_functions_30d_3000iters`; earlier 1,500-iteration results
remain archived. The 1000D run remains stopped after Happy Cat.

Start/resume this run (only when no other run is serving port 8765):

    uv run python -m experiments.smoke --output-dir experiments/results/all_functions_30d_3000iters --dimensions 30 --workers 2 --port 8765 --phase full

Live results: http://127.0.0.1:8765/ .

---

# Earlier handoff and historical protocols

# Current experiment transition

The active 1000D run is allowed to finish all 18 Happy Cat trials, then the
controller stops the old process tree and launches a 30D-only full experiment.
No remaining 1000D functions are scheduled afterward. Transition state is saved
in the old run's `transition_status.json`.

The revised experiment uses population 30, 1,500 iterations, a 45,030-evaluation
cap including initialization, two seeds, all 25 functions, and nine algorithms:
full CMA-ES is included; Sep-CMA-ES is excluded at 30D. There are 450 full trials.
The new output directory is `experiments/results/all_functions_30d_1500iters`.
Earlier 500-iteration 30D results are archived in the old directory and are not
reused as full-budget results. The 450 matching short smoke trials are reused.

After the controller has completed the handoff, the new run can be resumed with:

    uv run python -m experiments.smoke --output-dir experiments/results/all_functions_30d_1500iters --dimensions 30 --workers 2 --port 8765 --phase full

Do not run this command while the controller is waiting or while a benchmark
already owns port 8765. The dashboard remains http://127.0.0.1:8765/ .

The controller is a local Windows helper and uses the installed psutil package
to verify process identity before terminating the old tree. It checks both the
PID creation time and command line, and stops only that experiment's process
tree. Optimizer implementations are not changed by the handoff.

---

# Historical experiment protocol (before the revision)

# All-function smoke and full-budget experiments

Start or resume:

    uv run python -m experiments.smoke --output-dir experiments/results/all_functions_30d_1000d --workers 2 --port 8765 --phase both

Live page: http://127.0.0.1:8765/ (localhost only; refreshes every three seconds).
Use `--phase smoke` or `--phase full` to select a phase. Keep the Python process
running. Ctrl+C stops it; rerunning the same command resumes completed trials.
Do not run two orchestrators against one output directory at the same time.

## Exact requested plan

| Dimension | Population | Maximum iterations | Evaluation cap |
|---|---:|---:|---:|
| 30 | 30 | 500 | 15,030 |
| 1000 | 300 | 3,333 | 1,000,200 |

Only 30D and 1000D are currently scheduled. The 250D, 500D, and 750D
experiments were stopped; their completed records remain in the older output
directory and are excluded from the current dashboard and CSVs.

The revised rule is approximately 1000*D at higher dimensions, counting
initialization. The explicit 30D baseline is retained. Both seeds 0 and 1 are
used for all problems. All 20 classical functions plus all five scaling-suite
functions run at the requested dimension, including Michalewicz (the older
classical benchmark configuration still retains its default 10D exception).
CEC2011 fixed-dimension real-world problems are outside this mathematical suite.

All ten registered algorithms run at 30D. Full CMA-ES is excluded at 250D and
above; Sep-CMA-ES and the other eight algorithms remain. This gives 950 trials
per phase. Full budgets sum to approximately 458 million evaluations before
any algorithm-dependent shortfall; this is a substantial run, not a quick smoke
check. Two worker processes each restrict BLAS threads to one.

The smoke phase uses population 30, two iterations, and a maximum of 90
objective calls per trial, regardless of dimension. It checks finite evaluations,
feasible queried points, initialization and basic updates. These are diagnostic
scores, not optimized benchmark outcomes or a proof of mathematical definitions.
Only a smoke phase with no failed trials automatically advances to the full phase.

## Counting and outputs

Every objective call is counted, including initialization and redundant calls
inside selection and sorting. The hard cap may interrupt a final generation;
reported best is the best finite evaluated point, including partial generations.
The iteration limit is also enforced, so algorithms can use fewer evaluations
than the cap. Actual counts and termination reason are always saved. CMA-ES
variants have their native absolute 1e-10 target disabled for this experiment:
that target prematurely terminates benchmarks having negative objective values.
Their other algorithm settings are unchanged.

The translated classical bounds and the original scaling-suite definitions are
saved exactly in `problem_definitions.json`. Schwefel 2.22 reports log1p units;
raw scores from different functions are not pooled or compared as common errors.
Known optimum values of the custom scaling suite are zero; this is not assumed
for all classical functions.

* `plan.json`: exact population, iterations, cap, seeds, algorithm selections.
* `progress.json` and `index.html`: active jobs and most recent completions.
* `trials.jsonl`: append-only completed-trial journal, including sampled traces.
* `runs.csv`: completed trial scores, counts, status, timing, and error details.
* `summary.csv`: per-function, per-algorithm means, sample standard deviations,
  minimum score, and completed-seed counts, with smoke and full phases separated.

Results update during execution. A one-seed mean is provisional. Failed trials
are retained in runs.csv and omitted from the summary; inspect failures before
using comparisons. Resume reruns failed trials. Reusing this output after changing
objective or algorithm implementations would mix incompatible results: choose a
new directory after scientific code changes. Stop this process before editing
algorithm code; a process pool keeps its already imported implementation.

The server remains running when phases complete so results stay accessible.
This run does not overwrite prior pilot results or published manuscript tables.
