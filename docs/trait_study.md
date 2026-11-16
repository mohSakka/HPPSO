> Corrected zero-velocity reruns: see [zero_velocity_rerun.md](zero_velocity_rerun.md). Earlier completed results below are historical.

# Four-trait experiments

## Thirteen-function extension

The user requested five further shifted functions, three further rotated
functions and two further compositions, at the same full budget on this machine.
The added classical functions were sampled without replacement using selection
seed **20261107** from ten functions with known nonnegative minima. The three
rotated cases were similarly sampled from five unused rotations. The exact pools,
draw order and final names are in `experiments/configs/trait_extension_selection.json`.
The choices were frozen before any new optimization results were evaluated.

| Category | Added functions |
|---|---|
| Five shifted | Alpine 1, Levy, Rosenbrock, Griewank, Zakharov |
| Three rotated | Ellipsoid, discus, Schaffer F7 |
| Two compositions | Multimodal composition; new ridge composition |

The original Sphere, rotated Rastrigin and original composition remain unchanged.
The expanded study thus has **13 distinct functions**, still 100D, population 50,
ten matched seeds and 100,000 evaluations per path. The extension adds **100 batch
jobs, 5,900 paths and 590 million evaluations**. With the preserved original batch,
the combined dataset contains **130 jobs, 7,670 paths and 767 million evaluations**.

Run the complete local continuation pipeline:

```bash
python -u -m experiments.run_trait_extension --workers 2
```

It resumes new trials from `experiments/results/trait_study_added10`, generates
the additional-function report, combines the completed original results with the
new raw data in `experiments/results/trait_study_expanded13`, renders all plots,
audits the results and creates `artifacts/trait_study_expanded13_results.zip`.
`progress-0.json` tracks completed batches; `pipeline_status.json` tracks the
optimization/report/combination/audit/package stages. Original trials are copied
unchanged, with their original fingerprints. An archived source version is
verified without executing archived code and retained in the combined provenance.
The optimizer update and budget tracker are byte-for-byte unchanged. All previously
used objective branches are retained, and the additional evaluator branches are
checked against independent scalar definitions.

For a new run of all 13 functions from scratch, use
`experiments/configs/trait_study_expanded13.json` with the ordinary trait_study
runner in a fresh output directory. The standalone additions config is
`trait_study_added10.json`. Old three-function records remain reproducible with
`artifacts/trait_study_local3_source_snapshot.zip`; the current auditor can verify
them with `--source-archive` or verify the archives recorded in a combined plan.
Reports and audits recognize each source's execution fingerprint rather than
relabeling old raw results as a new source version.

### Additional composition definitions

The added existing `f30_composition_multimodal` mixes separately rotated Rastrigin,
Griewank and Expanded Schaffer F6, with three centers, widths (1,2,3), and biases
(0,100,200). Its complete definition is in additional_benchmarks.md and the saved
instance metadata.

The new local `f31_composition_ridges` mixes three distinct components:

1. Centered Rosenbrock, `h0(y)=sum_i[100*((y[i+1]+1)-(y[i]+1)^2)^2+y[i]^2]`.
2. Discus, `h1(y)=10^6*y[0]^2+sum_{i>0} y[i]^2`.
3. Schaffer F7, `h2(y)=(mean_i sqrt(r_i)*(1+sin(50*r_i^0.2)^2))^2`, with
   `r_i=sqrt(y[i]^2+y[i+1]^2)`.

For `z=x-o`, centers c0=0 and c1/c2 sampled independently from [-4,4]^D,
component k evaluates `h_k(Q_k*(z-c_k))+b_k`. Each Q_k is an independent
orthogonal signed/permuted DCT-II with norm=ortho. For squared distances
`d_k=||z-c_k||^2`, unnormalized weights are
`d_k^(-1/2)*exp(-d_k/(2*D*sigma_k^2))`, normalized to sum to one, with widths
sigma=(1,2,3) and biases b=(0,100,200). At an exact center, that center's weight
is one. Log-space normalization avoids underflow; no epsilon changes the function.

The box is `[o-5,o+5]`, o is sampled from [-4,4]^D, and all component functions
are nonnegative and zero at their own origins. Therefore `F(o)=0` is a feasible
global minimum. The varied shift coordinates preserve the intended nonhomogeneous
solution. Shift, rotations and centers use SeedSequence([instance_seed,D,31]);
the full arrays are saved in problem_definitions.json. This is a documented
custom benchmark, not an official CEC/BBOB function. The standard 30-function
benchmark registry and its earlier experiment definitions are not modified.

The completed expanded results are in
`experiments/results/trait_study_expanded13/index.html`, with readable findings in
`results_summary.md` and all numerical checks in `audit.json`. The result archive is
`artifacts/trait_study_expanded13_results.zip`; the runnable code bundle is
`artifacts/trait_study_expanded13_code.zip`.

The earlier three-function results remain in
`experiments/results/trait_study_local3_final/index.html`. This is an unchanged
snapshot of all 30 raw batches plus the earlier report. Windows denied overwriting
an interim CSV in the original folder, so final reporting uses this fresh
directory; no optimization was rerun. `source_snapshot.json` records its origin.

The full local study selects **shifted Sphere, rotated Rastrigin and the existing
ellipsoid/Rastrigin/Ackley composition**. All use **100D**, **50 particles**,
**10 matched seeds (0–9)** and **100,000 evaluations per path**, including
initialization. Exact shifts, feasible domains and rotation/composition parameters
are in the saved problem_definitions.json. The objectives have feasible varied
minimizers and reference value zero. Function definitions are unchanged.

The user selected **one parameter at a time**, shared curiosity/conformity
levels **0.5**, and non-personalized openness as a common activation probability
**1−theta** each generation. No personalities are reinitialized in the isolated
PSO experiments. All four traits are independently enabled/disabled in the HPPSO
factorial experiment; it uses the existing stagnation policy except for the
explicit full-HPPSO arm without personality reinitialization.

## Vectorized paths

One job represents a seed and a function. Its one optimization loop updates an
array of **59 independent paths** with shape `(paths, particles, dimensions)`.
Each path has separate positions, velocities, personal/global bests, traits and
stagnation counters. Global bests and peers are selected **within each path**.
Initial positions and exogenous random draws are matched across paths. Different
paths never exchange solutions. NumPy and SciPy evaluate the populations together.

This produces **30 batch jobs and 1,770 path results**, corresponding to **177
million individual objective evaluations**. A batch call evaluating 59×50 points
costs 2,950 logical evaluations, not one evaluation. Identical baselines are
computed once and referenced by several groups. The theta=1 openness endpoint
is algebraically PSO and reuses that path. Every path stops at the exact budget,
including a partial final generation when needed; no iteration or target-fitness
stop is used.

The vectorized formulas are algebraically equivalent to the original nine
functions and checked against their scalar evaluators. Reduction/FFT rounding
can differ from scalar execution; no old ablation scores are imported. The
batched engine is checked against independent scalar paths and the production
HPPSO update with matched draws. This separates numerical batching from
scientific changes to the optimizer.

## Isolated operators and controls

All isolated arms use standard PSO: inertia 0.65, c1=c2=2, per-coordinate learning
draws, and initial velocities zero in every coordinate. All paths use the same velocity
cap (20% of coordinate width) and repeated reflecting boundaries.

* **Curiosity:** PSO, PSO-m, personalized PSO-curiosity and shared-level
  PSO-curiosity. For each coordinate, mutate with probability `alpha_mut*q_i`
  and Gaussian standard deviation `eta*(upper-lower)`, after velocity clipping
  and motion, before reflection. Personalized q_i is a fixed uniform draw;
  shared q_i=0.5. Eta is varied over 0.0001/0.001/0.01/0.1 with alpha_mut held
  at its existing default 0.05. Alpha_mut is varied over 0.1/0.3/0.5/0.7/1 with
  eta=0.001. Both personalized and shared variants use every setting.
* **PSO-m:** the repository's existing reference: mutate an entire selected
  particle with probability 0.05, Gaussian standard deviation 0.1×coordinate
  width. This is distinct from coordinate-wise curiosity. The shared-curiosity
  arm isolates personalization under the same coordinate-wise mechanism.
* **Aggressiveness:** PSO, PSO-aggressiveness and PSO-RIW. Personalized inertia
  is `0.4+0.5*a_i` for fixed a_i~U(0,1). RIW uses one fresh inertia U(0.4,0.9)
  shared by the swarm each generation. Ordinary PSO uses the mean inertia 0.65.
* **Openness:** PSO plus personalized and non-personalized variants. Sample one
  uniform non-self peer from all peers, update only when its **current** fitness
  is strictly worse, and attract toward its **personal best** with
  `lambda*(P_j-X_i)`. Personalized eligibility is fixed `o_i>theta`. The global
  control gives every particle probability `1-theta` independently each
  generation; it has no persistent personality roles. Theta and lambda are each
  swept over 0.1/0.3/0.5/0.7/1 while the other is fixed at theta=0.5 or lambda=0.4.
  Theta=1 is an explicit no-op control. Constant shared openness 0.5 is supported
  as an alternative configuration, but would disable all particles at theta>=0.5.
* **Conformity:** PSO, personalized PSO-conformity and shared-level
  PSO-conformity. This is the existing HPPSO confidence/conformity weight:
  `(1-f_i)*c1*r1*(P_i-X_i)+f_i*c2*r2*(g-X_i)`. Personalized f_i~U(0,1), shared
  f_i=0.5. Removing conformity restores **unweighted** ordinary PSO cognitive
  and social terms. Consequently conformity changes the total learning gain
  as well as its per-particle split; comparing personalized to shared 0.5
  isolates heterogeneity at the same expected gain. It is not equivalent to
  simply setting f_i=0, which would remove social learning.

Mutation, peer, activation, inertia and proposed-reinitialization draws use
separate seeded streams. Masking a trait or changing its parameter does not
consume another trait's draws. Curiosity masks and Gaussian samples are matched
across eta/alpha settings. A new full reset proposal is sampled each generation;
only particles whose actual stagnation counters reach the threshold apply it.

## HPPSO ablations and interactions

All **16 subsets** of curiosity/conformity/aggressiveness/openness are run on
one common HPPSO backbone. It starts with zero velocities and uses one cognitive
and social uniform draw per particle, matching HPPSO's equations. A missing
aggressiveness trait fixes inertia at 0.65; missing conformity restores ordinary
unweighted cognitive/social terms; missing curiosity/openness removes that term.
The empty subset is an internal factorial control. It uses per-particle learning draws, whereas standard PSO uses per-coordinate learning draws. It is excluded from the ablation comparison.

The ablation group includes full HPPSO, each of the four leave-one-trait-out
variants and **HPPSO without stagnation-triggered personality
reinitialization**. The factorial arms retain the threshold-10 reinitialization
policy; their interactions therefore include its adaptive consequences. Full
HPPSO without reinitialization changes only that policy. Default HPPSO parameters
remain eta=0.001, alpha_mut=0.05, theta=0.5 and lambda=0.4. Hyperparameter winners
are not selected after seeing results and inserted into the factorial study.

Interaction responses are `Y=final_error/initial_best_error`, using each path's
matched evaluated initial population. They do not use an extra random reference
sample or alter the optimized objective. If initial error is exactly zero, best
error must stay zero and the normalized response is defined as zero. This metric
permits equal-weight aggregation across the selected fixed functions.

For traits A and B, pair interactions are
`mean_other_traits(Y11-Y10-Y01+Y00)`, computed within each seed. Negative values
mean the joint effect reduces error beyond the sum of the separate effects on
this normalized-response scale; positive values mean less-than-additive gains.
Interaction matrices are symmetric and leave the diagonal blank. These are
factorial intervention contrasts, not correlations. Main effects and all triple
and four-way contrasts are also saved, with their subset identifiers.

Uncertainty for factorial contrasts uses percentile bootstrap intervals for the
**mean of paired seed contrasts**. Aggregate matrices average functions equally
within each seed, then resample whole seed blocks; this preserves matching and
describes optimizer randomness on these fixed instances, not uncertainty over a
population of benchmark functions. Ten seeds and unadjusted 95% intervals are
exploratory; no multiple-comparison significance claims are made.

## Run locally or on the server

From the repository root, after `python -m pip install -e .`:

```bash
python -u -m experiments.trait_study --config experiments/configs/trait_study.json --output experiments/results/trait_study_local3 --workers 2
python -m experiments.trait_report --output experiments/results/trait_study_local3 --sweep-curves
python -m experiments.audit_trait_study --output experiments/results/trait_study_local3
```

The first command executes experiments; the second creates all plots and tables.
The third audits budgets, feasible solutions, original scalar objective values
and independently reconstructed pairwise interactions. To regenerate just curves
from saved statistics without another bootstrap or optimization run, use
`python -m experiments.trait_curve_plots --output PATH`.
On one 24-core EPYC 7443, start with 22–24 workers if RAM permits. A two-socket
48-core system may use 44–48. Threads are not additional physical cores. Each
worker uses one BLAS/OpenMP thread. Each job holds all variant states in memory;
measure memory before increasing concurrency. A persistent shell/session or
server scheduler is appropriate for unattended execution.

`--dry-run` prints exact job/path/evaluation counts without running. Use
`--groups curiosity curiosity_grid`, `--groups aggressiveness`, `--groups
openness openness_grid`, `--groups conformity`, or `--groups hp_ablation`
to execute selected groups in a fresh output directory. Selecting the factorial
or HPPSO-ablation group includes both groups so the interaction design stays
complete. Changing code/configuration/groups/shard count requires a fresh output
directory. The default config uses the three selected functions; the broader
nine-function short validation is saved separately.

Completed function/seed batches are written atomically as gzip JSON and skipped
on resume. A killed partial batch is rerun as a whole with the same path layout.
Failures are preserved, and skipped by default; `--retry-failed` archives the
previous attempt before retrying. No invalid value is substituted into ranks.
Reports require all seeds of a function to have valid complete batches before
including that function. Source/configuration hashes prevent accidental mixing.

One writer per shard is enforced by an exclusive lock. After a forced kill,
remove a stale lock only after verifying the writer and all its workers exited.
Independent shards use the same output directory and `--shards N --shard I`;
each seed/function job belongs to exactly one shard. Keep N fixed on resume.
See `experiments/trait_study.slurm` for an array-job example.

## Saved outputs

* `plan.json`: exact variant specifications, group aliases, factorial bit masks,
  parameters and source/configuration fingerprint.
* `problem_definitions.json`: exact benchmark instance metadata.
* `trials/<function>/seed_NNN.json.gz`: every path's final score/position,
  initial best, sampled raw convergence, reset count, exact evaluation count,
  host, status/error and **whole-batch** elapsed seconds. Shared execution cannot
  identify independent algorithm wall-clock times; dividing batch time by the
  number of paths would not measure each algorithm's execution time.
* `batch_timings.csv`: one timing record per seed/function batch, with both logical
  evaluation counts and the number of vectorized objective calls.
* `runs.csv`, `summary.csv`: expanded path results and final mean/sample standard
  deviation, median/IQR, mean reset counts and 95% seed-bootstrap intervals for the median.
* `convergence_summary.csv`: sampled medians/IQRs and displayed curve confidence
  intervals. Grid-only paths omit curve CIs unless `--sweep-curves` is requested;
  final-score CIs and raw seed trajectories are always available.
* `paired_comparisons.csv`: matched-seed differences against each group's
  reference, median-difference CIs and win/tie/loss counts.
* `group_ranks.csv`, `average_group_ranks.csv`: descriptive ranks within the
  five default comparison groups. Parameter settings are not pooled into a
  global algorithm ranking.
* `parameter_results.csv`: every evaluated setting and shared/personalized
  comparison, including aliases for no-op endpoints. Response plots show
  one-parameter slices with the other parameter held at its declared default.
* `factorial_effects.csv`, `factorial_seed_contrasts.csv`: all 15 main and
  interaction contrasts per function and their paired uncertainty estimates.
* `figures/convergence.pdf`, `late_convergence_linear.pdf`: default groups on
  full budget axes and linear zooms of the final 20%, with pointwise median CIs.
* `figures/parameter_convergence.pdf`: with `--sweep-curves`, convergence for
  every parameter slice; per-setting raw traces are saved regardless.
* `figures/*_parameter_response.png`, `figures/interactions/*.png`: parameter
  response figures and per-function/aggregate four-trait interaction matrices.
* `index.html`, `report_status.json`, `progress-*.json`: gallery, completeness and
  live batch progress. A 1,000-evaluation smoke run is validation, not the final
  100,000-evaluation experiment.

Full-budget plots use raw objective error on log/zero-safe axes; linear late
views expose small differences. Curves summarize the median and pointwise 95%
seed-bootstrap intervals. Whole seeds are resampled with common bootstrap indices
across paths and checkpoints, preserving pairing. Intervals are not simultaneous
confidence bands or the spread of individual runs. Timing is batch timing under
concurrent load and includes optimizer setup, evaluation and tracing, but excludes
instance construction, serialization and report generation.
