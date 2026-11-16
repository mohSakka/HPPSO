# Current dimensional experiment functions

This harness studies dimensionality using the ten functions in
[additional_benchmarks.md](additional_benchmarks.md). The old duplicate Sphere
and Rosenbrock cases and all `scaling_*` function IDs have been removed from
active code. `experiments/configs/scaling.yaml` selects f21 through f30.
The evaluation budgets, actual-call accounting, reference normalization,
median/IQR plots and output files below still apply. New results carry version
`additional-v2`; use a fresh output directory. The function definitions in the
historical section below describe retired results only, not the active suite.

---

# Archived scaling-v1 experiment protocol

# Representative dimensional scaling study

Run `uv run hppso-run-experiment scaling`, or
`uv run python -m experiments.run scaling`.
Use `--plots-from <run-directory>` to regenerate the figures.

## Design and interpretation

The default pilot uses dimensions 100,200,...,1000, five functions, HPPSO,
Original PSO, SHADE, and Sep-CMA-ES, three independent optimizer seeds, and a
fixed population of 30. Change the YAML algorithm list to include PSO-m,
PSO-RIW, or other registered algorithms. Full CMA-ES is available but its dense
covariance cost is substantial at high dimension.

Two evaluation policies answer different questions:

* Proportional: B(D)=10D objective evaluations. This holds evaluations per
  variable constant, while wall time may grow much faster than D.
* Fixed: B(D)=1000 evaluations. This holds total objective resources constant,
  showing degradation as the same resources cover more variables.

These defaults are exploratory, intentionally small budgets, not claims that
10D evaluations are sufficient for convergence. For publication, increase
`evaluations_per_dimension`, independent runs and problem instances as resources
allow. A useful follow-up varies k in B(D)=kD. Avoid choosing k after seeing which
setting favors a particular algorithm. COCO describes proportional budgets and
anytime assessment: https://numbbo.github.io/coco-doc/experimental-setup/ .

A wrapper counts every objective call, including initialization, reevaluations,
and comparisons inside algorithms. It raises before any call beyond the cap.
The reported answer is the best finite point actually evaluated, including a
partially evaluated final generation. The iteration ceiling is sufficient to
reach the evaluation cap; algorithm-dependent schedules still use their native
iteration progress. This is evaluation fairness, not identical iteration counts.
Algorithms with their own termination conditions may stop early; actual counts,
status, and wall times are recorded. No unrequested restarts are introduced.
The trial timer includes algorithm initialization and excludes problem creation,
reference sampling, result serialization, and plotting.

All algorithms and seeds share the same function instance at each dimension.
Shifts and rotations come from a separate deterministic RNG and do not consume
optimizer RNG state. There is one function instance per dimension in this pilot;
three optimizer seeds do not substitute for multiple independent instances.

## Mathematical definitions (scaling-v1)

Let o have independently sampled coordinates from U[-4,4], z=x-o, and
x belong to [o-5,o+5]. Each objective is nonnegative, with a known feasible
global minimizer x=o and minimum zero. The same translated domains are passed
to all algorithms; shifting does not restrict the original feasible box.

The fixed orthogonal transformation is

    y = Qz = DCT-II(s * z[p]), norm='ortho',

where p is a fixed permutation and s has independent signs +/-1. The transform
mixes coordinates and costs O(D log D), avoiding a dense O(D^2) multiplication
per evaluation. This is a structured orthogonal rotation, not a uniformly random
orthogonal matrix. It is reproducible from the exact p and s in the manifest.

Define

    E(v) = sum_i (10^6)^(i/(m-1)) * v_i^2    (m=len(v), i=0,...,m-1)
    R(v) = sum_i [v_i^2 + 20 sin^2(pi v_i)]
    H(v) = sum_{i=1}^{m-1} [100(v_{i+1}-2v_i-v_i^2)^2 + v_i^2].

H(v) is classical Rosenbrock evaluated at v+1. The suite contains:

1. Sphere: sum_i z_i^2, a separable control.
2. Rosenbrock: H(z), a non-separable valley problem.
3. Rotated ellipsoid: E(Qz), non-separable and ill-conditioned (condition 10^6).
4. Rotated Rastrigin: R(Qz), non-separable and multimodal.
5. Composition: split y=Qz into three nearly equal contiguous blocks a,b,c;
   F(x)=D*[E(a)/(10^6*len(a)) + R(b)/(10*len(b)) + H(c)/(100*len(c))].

The last function is a custom additive hybrid composition with one shared global
minimizer; it is not a distance-weighted, multiple-center CEC composition.
These definitions are representative stress tests, not official CEC/BBOB
implementations or evidence of equivalent coverage. Original classical functions
remain available unchanged through the classical experiment configuration.

## Outputs and trend plots

* `problem_definitions.json`: exact shifts, bounds, signs, permutation,
  reference optimum, rotation convention, and suite version.
* `scaling_runs.csv`: per-run budget, actual evaluations, status/failure,
  wall time, best score, error/D, and normalized error.
* `scaling_summary.csv`: median and interquartile range for every group.
* `metrics.json`: all runs, sampled evaluation-count traces, and summaries.
* `scaling_checkpoint.json`: completed trials saved after each function.
  This is a recoverable record, not an automatic resume mechanism.
* Four figures: normalized-error and wall-time trends for each budget policy.

Normalized error is best objective error divided by the median error at 100
common uniform reference points from that exact problem instance. Reference
sampling is outside every algorithm's budget, is identical for all algorithms,
and its value is saved. This reduces changes in objective scale with dimension;
it does not make different landscapes equally difficult. Raw error/D is also
saved. Figures are per function; incomparable function values are not pooled.
The composition is scaled by D so error/D has a consistent block-mean meaning.

Bands are interquartile ranges across optimizer seeds, not confidence intervals.
Groups containing failed runs produce plot gaps. Wall time includes objective
cost and machine/load effects, so it is not pure optimizer overhead. Report
hardware and environment for publication. Do not infer a complexity exponent or
statistical significance from three-run pilot curves alone.
