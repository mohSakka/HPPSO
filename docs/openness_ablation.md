# Isolated PSO openness ablation

## Final selected study

The user-selected final study has **nine functions**: Sphere, Rosenbrock, Levy,
Ackley, Expanded Schaffer F6, rotated ellipsoid, rotated Rastrigin, rotated Ackley,
and **one composition function** (ellipsoid/Rastrigin/Ackley). It keeps 100D,
50 particles, 100,000 evaluations, all five arms and ten matched seeds: **450
trials and 45 million evaluations**. The second composition and remaining
challenging functions are excluded from the final selection.

To run all nine from scratch on the server:

```bash
python -u -m experiments.openness_ablation --config experiments/configs/openness_ablation_final9.json --output experiments/results/openness_ablation_final9_100d --workers 24
python -m experiments.plot_openness_detail --output experiments/results/openness_ablation_final9_100d
```

`openness_ablation_selected.json` contains just the five selected additions.
The local extension continues in `openness_ablation_hard_100d_selected`: it
reuses 228 valid trials from the interrupted ten-function plan and runs only
the remaining 22 composition trials. Original trial files are preserved. Copies
carry their original execution fingerprint and source under `reused_from`; their
top-level fingerprint identifies the new analysis plan. Reuse requires identical
source code and settings, allowing only a subset of the original function list.

The final combined local report is in `openness_ablation_combined_9functions`.
Its `plan.json` lists both source plans; `runs.csv` retains each source's raw
fingerprint. Generate it after both source reports are complete:

```bash
python -m experiments.combine_openness_reports --sources experiments/results/openness_ablation_100d experiments/results/openness_ablation_hard_100d_selected --output experiments/results/openness_ablation_combined_9functions
```

This combines medians, seed uncertainty estimates, paired comparisons and
per-function ranks without pooling objective magnitudes. It generates full
convergence, linear late-budget views and final-score plots for all nine.
The combined directory is an analysis product, not a resumable trial directory.

## Original protocol and optional broader configuration

Run all five arms on shifted Sphere, Rosenbrock, Levy and Ackley at **100D**,
with seeds **0–9**, **50 particles**, and **100,000 objective evaluations per
trial**, including initialization. This is 200 trials and 20 million evaluations.
Stopping uses the exact evaluation count, including a possible partial final
generation. There is no iteration or target-fitness stopping rule.

The challenging extension adds ten existing benchmark instances, using exactly
the same optimizer settings, 100D, ten seeds, population and evaluation budget:

| Function | Main challenge |
|---|---|
| Expanded Schaffer F6 | Cyclic interactions and many local minima |
| Rotated ellipsoid | Non-separability and condition number 1,000,000 |
| Rotated Rastrigin | Coupled multimodal landscape |
| Rotated Ackley | Coupling and a narrow optimum region |
| Composition | Rotated ellipsoid/Rastrigin/Ackley, distinct centers |
| Rotated Griewank | Coupled oscillatory landscape |
| Rotated discus | Coupling with one strongly weighted direction |
| Schwefel 1.2 | Cumulative cross-coordinate interactions |
| Rotated Schaffer F7 | Coupled oscillatory pair interactions |
| Multimodal composition | Rotated Rastrigin/Griewank/Expanded Schaffer F6 |

These add **500 trials and 50 million evaluations**, bringing the study to
**14 functions, 700 trials and 70 million evaluations**. These are the repository's
documented local instances, not an official CEC/BBOB suite. Rotations use an
orthogonal signed/permuted DCT. Exact shifts, rotations and composition parameters
are saved; formulas are in [additional_benchmarks.md](additional_benchmarks.md).
All ten additions have feasible varied minimizers x=o and objective minimum zero.

| Arm | Current-fitness gate | Attraction target |
|---|---|---|
| PSO | No openness term | None |
| PSO-open-worse-pbest (original openness) | f(X_j) > f(X_i) | P_j |
| PSO-open-better-pbest | f(X_j) < f(X_i) | P_j |
| PSO-open-any-pbest | All sampled peers, including equal fitness | P_j |
| PSO-open-worse-current | f(X_j) > f(X_i) | X_j |

All arms use standard PSO, with inertia **0.65**, **c1=c2=2**, per-coordinate
uniform cognitive/social draws, initial velocities uniform in [-1,1], velocity
clipping at **20% of each coordinate's domain width**, and reflecting boundaries.
The baseline equations match the repository's standard PSO with mutation disabled.
There is no confidence weighting, aggressive inertia, curiosity/mutation,
stagnation mechanism, or personality reinitialization.

For the four openness arms only, draw one fixed openness value o_i uniformly in
[0,1) for each particle. The particle is eligible when **o_i > 0.5**, preserving
the HPPSO openness threshold. On each generation sample **one random non-self
peer uniformly from all other particles**. Sampling is never conditioned on
fitness and a rejected peer is never replaced. For eligible particles satisfying
the arm's gate, add **0.4*(target_j-X_i)** to the standard PSO velocity before
clipping. Gates always use current scores, even when attraction targets a pbest.
Thus these are isolated openness variants of standard PSO, not the full HPPSO.

Initialization and base PSO random coefficients are matched across all five
arms for each seed. Separate seeded trait and peer streams ensure openness does
not consume base PSO draws. Trait values and peer choices are matched across the
four openness arms; all particles have peer draws, but only eligible particles
can be affected. These matching choices reduce avoidable comparison noise.

The four shifted instances are fixed using instance seed 42, with the same
construction as the final benchmark, and shared across algorithms and seeds.
Domains translate with the functions: F(x)=f(x-o), x in [L+o,U+o]. Sphere and
Ackley have minimizers x=o, while Rosenbrock and Levy have minimizers x=o+1.
Exact shifts, domains and minimizers are saved in problem_definitions.json.
All reference objective values are zero; optimal coordinates are varied.

## Run and resume

From the repository root, after `python -m pip install -e .`:

```bash
python -u -m experiments.openness_ablation --workers 4
```

For server transfer, `artifacts/openness_ablation_server.zip` contains the source,
configuration and instructions. Unzip it and enter the `hppso-server` directory.
On Linux, for example:

```bash
unzip openness_ablation_server.zip
cd hppso-server
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -u -m experiments.openness_ablation --workers 24 > openness.log 2>&1
```

Use a persistent shell/session or a server scheduler for unattended execution.

Run the challenging extension in its own output directory:

```bash
python -u -m experiments.openness_ablation --config experiments/configs/openness_ablation_challenging.json --output experiments/results/openness_ablation_hard_100d --workers 4
python -m experiments.plot_openness_detail --output experiments/results/openness_ablation_hard_100d
```

The original four-function results remain in `openness_ablation_100d`; this avoids
relabeling or rerunning their saved trials. The two batches share scientific
settings but retain separate source/configuration fingerprints. Reporting now
supports any number of selected functions, with four-panel pages. Multi-page
files use `_02.png`, `_03.png`, etc.; `overview_pages.pdf` contains all overview
pages, `all_functions.pdf` contains one curve page per function, and
`linear_detail.pdf` includes every late-curve and final-score page.

During a run, `--report --output PATH` creates a snapshot using only complete
function blocks. Follow it with `python -m experiments.plot_openness_detail
--output PATH --completed-only` for a linear-scale snapshot of those blocks.
The full report and default detail command should be regenerated after completion.

The reporting upgrade changes the runner fingerprint, so the old four-function
batch can be viewed using `--report`; to execute it again using this code version,
choose a fresh output directory. The optimizer and objective implementations
have not changed in the extension.

This prints progress after each trial and automatically generates reports.
The default output directory is `experiments/results/openness_ablation_100d`.
Use `--output PATH` and `--config PATH` to override. On a dual-socket 7443 server,
`--workers 44` is a possible starting point; each worker uses one BLAS thread.
Resume with the same command: saved trials, including recorded failures, are
skipped. Code/config fingerprints prevent mixing incompatible trials. A killed
partial trial is rerun from its seed. A stale `.shard-openness.lock` may be
removed only after verifying no process is still writing this output directory.
Investigate saved failures before choosing a fresh output directory for a rerun.

Regenerate reports without running any trial:

```bash
python -m experiments.openness_ablation --report
```

For a linear-scale view of the last 20% of evaluations, plus final-score plots
showing every seed and median confidence intervals, run:

```bash
python -m experiments.plot_openness_detail
```

This saves `convergence/late_convergence_linear.png`, `final_scores_linear.png`
and the two-page `linear_detail.pdf`, and adds them to the local HTML gallery.
Use `--start-fraction 0.5` to show the final half instead. These are supplementary
views of the same saved observations; they do not rerun or transform the scores.

## Results and uncertainty

* `trials/`: gzip JSON per seed, with final score, best position, exact evaluation
  count, sampled best-so-far trajectory, elapsed seconds, status/error and metadata.
* `runs.csv`: raw per-trial results and execution times.
* `summary.csv`: per-function/algorithm mean, sample standard deviation, median,
  IQR and a **95% percentile bootstrap interval for the median**.
* `paired_comparisons.csv`: matched-seed final-score differences against PSO and
  original openness, with 95% bootstrap intervals for the median paired difference
  and win/tie/loss counts. Negative differences favor the listed algorithm.
* `convergence_summary.csv`: every saved curve checkpoint's median, IQR and
  confidence limits. Raw seed trajectories are retained in the trial JSONs.
* `function_ranks.csv` and `average_ranks.csv`: descriptive ranks based on final
  median scores, with average ranks across complete functions only.
* `convergence/`: per-function PNGs, four-panel overview pages and a PDF of all functions;
  `index.html` opens the gallery locally.
* `plan.json`, `problem_definitions.json`, `progress.json`, `report_status.json`:
  reproducibility, progress and completeness information.

Convergence shows median best-so-far **objective error f(x)-f*=f(x)** against
actual evaluations, including initialization. The vertical scale is logarithmic
outside a tiny linear interval around zero (symlog, threshold 1e-12), preserving
exact zeros and small Ackley roundoff without altering stored values. Bands are
**pointwise 95% bootstrap confidence intervals for the seed median**, not the
spread of individual runs or simultaneous confidence bands. Bootstrap resamples
whole seeds 2,000 times, using the same resampled seed indices across arms and
checkpoints. Ten seeds give exploratory uncertainty estimates; no multiple-test
significance claim is made. Differences of paired seed scores are distinct from
differences between unpaired medians. The selected representative functions support
operator comparisons on these instances, not general claims across all problems.

All arms and seeds for a function must finish successfully at the exact budget
before its comparisons/plots are included. Failed/partial runs remain in the raw
results and are counted in report_status.json. Runtime wraps optimizer setup,
objective evaluations and tracing under concurrent load; instance construction,
result serialization, bootstrapping and plotting are excluded.
