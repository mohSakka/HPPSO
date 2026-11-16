# Quantitative swarm diversity experiment

The four algorithms are HPPSO, ordinary PSO, PSO-m and PSO-RIW. The protocol uses
the exact 13 shifted/rotated/composition instances of the corrected trait study:
100 dimensions, population 50, seeds 0–9 and 100,000 objective evaluations per
algorithm, including initialization. All initial velocities are zero. Parameters,
boundary reflection and velocity limits match the corrected ablation study.

This is 130 matched batches, 520 algorithm runs and 52 million optimization
evaluations. Each algorithm retains independent positions, fitnesses and bests;
matched random streams and starting populations pair the comparisons across
seeds. Instrumentation does not sample random numbers or evaluate objectives.

## Definitions

For the current evaluated population x_i, define z_ij=x_ij/(upper_j-lower_j).
Let zbar be the coordinate-wise population mean, N the particle count and D
the dimension. The metrics are:

* Population dispersion: sqrt(sum_i ||z_i-zbar||^2/(N D)). It is the root mean
  squared coordinate deviation from the swarm centroid (population denominator N).
* Exact mean pairwise distance: 2 sum_(i<j) ||z_i-z_j||/(N(N-1)sqrt(D)). This
  includes all distinct particle pairs, with no sampling or estimated distances.
* Mean particle movement: sum_i ||(x_i(t)-x_i(t-1))/width||/(N sqrt(D)). This
  measures displacement during the most recent generation, including mutation
  and boundary reflection. It is not displacement since the preceding saved
  checkpoint. The initial value is defined as zero.

The metrics use current positions, not personal-best archives. A uniform swarm
has expected dispersion near 1/sqrt(12). Spatial distances are dimensionless and
translation invariant. No division by the initial diversity or objective error
is involved. The two spatial summaries are related; movement gives complementary
trajectory information rather than a third independent spatial measure.

Observations include the evaluated initial population and the final population,
with a fixed union of logarithmic and linear generation checkpoints in between.
Only complete generations are accepted. At the final evaluation budget the engine
does not update its personal-best archive, so objective curves use the budget
tracker's best evaluated candidate, as in the original corrected trait study.

## Outputs and interpretation

Each function has a four-panel figure: dispersion, pairwise distance, movement
and objective error against evaluations. All trajectory x-axes use a logarithmic evaluation scale to expand early
optimization. Diversity y-axes are linear to preserve absolute differences;
objective error uses log scale when all values are positive
and zero-safe symlog otherwise. Curves show medians and 95% percentile bootstrap
intervals over whole seeds (2,000 resamples, fixed bootstrap seed).

The raw gzip records retain every seed's trajectories. CSV files save per-seed
final values, average diversity over the evaluated interval (trapezoidal AUC
divided by interval length), and average diversity over the final 20% of the
budget. Paired differences are HPPSO minus each comparator, with seed-bootstrap
intervals. Higher diversity is a positive difference; lower objective error is
a negative difference. These intervals are exploratory and unadjusted for the
multiple functions and metrics. The fixed instance suite does not establish
performance across all possible optimization problems.

Maintaining greater dispersion alone does not prove prevention of premature
convergence: mutation can retain spread while failing to improve solutions.
Interpret diversity jointly with the objective curves and paired final errors,
including any functions where HPPSO has lower diversity or worse error. This
experiment evaluates the reviewer's claim rather than assuming it will hold.

Every saved final score and solution is checked against its corresponding
completed corrected ablation path. A mismatch stops the run. This verifies that
the four-path replay and instrumentation preserve the previously evaluated
optimization trajectories. The analytical metric and instrumentation tests are
in tests/test_diversity_study.py.

## Run and resume

From the repository root:

```bash
python -m pip install -e .
python -u -m experiments.diversity_study --workers 2
```

The default reference is experiments/results/trait_study_zero_velocity13.
The runner checks its configuration and requires those raw reference results.
It resumes saved completed diversity batches, rejects source/config changes,
then generates reports, audit.json and a results ZIP automatically. To run on
another machine, also copy the reference plan.json and trials directory, and
use --reference PATH. Metric values cannot be reconstructed from best-only
convergence records; the replay is necessary to observe the full population.

Results: experiments/results/diversity_zero_velocity13/index.html.
Progress: experiments/results/diversity_zero_velocity13/progress.json.
Archive: artifacts/diversity_zero_velocity13_results.zip.
