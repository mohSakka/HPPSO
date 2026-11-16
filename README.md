# HPPSO

Human Personality based Particle Swarm Optimization — a PSO variant driven by
four personality traits (curiosity, confidence, aggressiveness, openness),
benchmarked against PSO, PSO-m, PSO-RIW, GA-MPC, GWO, SHADE, CMA-ES,
Sep-CMA-ES, and CSA on 20 shifted classical functions plus ten additional hard problems, and the CEC2011 suite.

## Install

```bash
uv sync                       # core dependencies
uv sync --extra cec2011       # add minionpy for the CEC2011 benchmark
uv sync --extra all           # everything (cec2011 + posthoc stats)
```

## Quick start

Run HPPSO on a single function:

```python
import numpy as np
from hppso import HPPSO

def sphere(x):
    return float(np.sum(x ** 2))

opt = HPPSO(sphere, n_pop=30, dimensions=30, max_it=500, bounds=(-100, 100))
best_score, history = opt.optimize()
print(f"best fitness: {best_score:.2e}")
```

HPPSO minimizes the objective. For each particle whose openness exceeds
`socialism_threshold`, it samples one uniformly random peer excluding itself.
Only if that peer has strictly worse **current** fitness does it add
`lam * (peer_personal_best - particle_position)` to velocity. Better or equal
peers produce no openness update; there is no resampling.

HPPSO samples independent random velocity coefficients for each particle and coordinate.
Curiosity is sampled independently per coordinate with probability
`alpha_curio * curiosity` (default `alpha_curio=0.05`), using Gaussian noise with
standard deviation `eta * (upper_bound - lower_bound)` (default `eta=0.001`). Velocity is clipped before
moving and mutation, and positions are reflected into the bounds afterward, reversing the velocity
components at each boundary crossing (including repeated crossings).
Set `v_max` to a scalar or a per-dimension array for finite velocity clipping;
the shared default for HPPSO, PSO, PSO-m, and PSO-RIW is 20% of each
dimension's search range. All four use the same reflecting boundaries and default
`c1=c2=2`. Pass `v_max=np.inf` to disable velocity clipping. The supplied
pseudocode does not specify numerical values for `alpha_curio` or `v_max`.
The existing `socialism_threshold` parameter name is retained for compatibility.

Run the algorithm regression tests with `python -m unittest discover -s tests -v`.

Run a full benchmark from a YAML config:

```bash
uv run hppso-run-experiment classical_30d
# or:
uv run python experiments/run.py experiments/configs/classical_30d.yaml
```

Each run writes a timestamped folder under `experiments/results/<name>/`:

```
experiments/results/classical_30d/20260604_164332/
├── config_used.yaml      # provenance snapshot
├── metrics.json          # full per-algorithm, per-problem results
├── results.csv           # long-form (classical / cec2011 only)
└── plots/                # avg ranks, wins, per-function convergence
```

Add `--no-plots` to skip plot generation, or regenerate plots later
without rerunning:

```bash
uv run hppso-run-experiment --plots-from experiments/results/classical_30d/<timestamp>
```

See [experiments/README.md](experiments/README.md) for the full config
schema and the four shipped configs (`classical_30d`, `classical_1000d`,
`cec2011`, `nn_diabetes`).

## Layout

```
src/hppso/        # Library: algorithms, benchmarks, runners, nn, utils
experiments/      # YAML-driven harness (configs/, run.py, plots.py)
reproduction/     # Paper-reproduction pipeline (reads pickles in results/)
notebooks/        # Cleaned tutorial notebooks
docs/             # Algorithm description + manuscript PDF
artifacts/        # Pre-generated outputs, originals, superseded files
```

## Reproducing paper figures and tables

Drop the original pickles in `results/` (gitignored) and run:

```bash
uv run python -m reproduction.run_reproduction
```

Figures and tables land in `artifacts/reproduced_figures/` and
`artifacts/reproduced_tables/`. See [artifacts/REPRODUCTION_REPORT.md](artifacts/REPRODUCTION_REPORT.md)
for the paper-vs-reproduced coverage matrix.

## Documentation

- Algorithm description: [docs/algorithm.md](docs/algorithm.md)
- Manuscript: [docs/papers/HPPSO_Manuscript.pdf](docs/papers/HPPSO_Manuscript.pdf)
- Experiment harness: [experiments/README.md](experiments/README.md)

## Numerical evaluation

Schwefel 2.22 now returns the stable, order-preserving `ln(1 + f)` objective.
Its scores use transformed units; historical results must be rerun. See
[the numerical evaluation and analysis policy](docs/numerical_stability.md).

## Shifted benchmark domains

Classical problems translate both the objective and its feasible domain:
`F(x) = f(x - o)` with per-dimension bounds `[L + o, U + o]`.
Thus `x - o` always lies in the original benchmark domain, and every original
feasible minimizer `z*` maps to the feasible minimizer `x* = z* + o`.
Independent coordinate shifts give varied solution coordinates; the translation
preserves domain widths and the optimum's relative location inside the box.

The suite generates one shift vector per problem and shares it across all
algorithms and runs. Both classical experiment entry points save
`problem_definitions.json`, containing the exact shift vector, base bounds,
translated bounds, dimension, shift convention, and objective scale. Explicit
`fixed_shifts` take precedence over random generation, allowing reconstruction
from saved vectors. `build_problem_suite(random_shift=False)` retains its existing
behavior of generating random vectors once per suite; it does not disable shifts.

Previously saved results used the old fixed-domain construction and must be
rerun for comparison with this corrected suite. Domain translation preserves
the implemented base function; it does not independently certify every formula
or claimed optimum in the manuscript. Schwefel 2.22 uses the log1p scale described
above. Schwefel 2.26 retains its rounded `418.9829` constant, so its reference
minimum is approximately, rather than exactly, zero.

## Additional hard benchmarks

The default experiment suite now contains the classical 20 plus ten additional hard functions:
Expanded Schaffer F6, rotated ellipsoid, rotated Rastrigin, rotated Ackley,
two distinct three-center composition functions, rotated Griewank, rotated discus,
Schwefel 1.2, and rotated Schaffer F7. The extra Sphere/Rosenbrock cases
and all `scaling_*` function IDs have been removed. Use `build_benchmark_suite`
for all 30 or `build_additional_suite` for just the additions. See the
[exact definitions, bounds, shifts and rotations](docs/additional_benchmarks.md).

Archived results containing retired variants remain historical records; their
scores are not relabelled as results for the new functions.

## Representative dimensional experiments

Run `uv run hppso-run-experiment scaling` for dimensions 100–1000 on the five
additional hard functions, including structured rotations and a three-center
composition. Both proportional and fixed evaluation budgets are supported, with
actual evaluation counting and median/IQR trend plots. See
[the experiment definitions and interpretation](docs/scaling_experiments.md).

## Final server benchmark

The final protocol uses dimensions 30/300/600/1000, exactly `1000*D` objective
evaluations, increasing populations, and 20 seeds across all 30 functions.
CMA-ES is used only at 30D, replaced by Sep-CMA-ES at higher dimensions.
See [server setup, resumable runs, SLURM and report commands](docs/final_benchmark.md).

## Isolated openness ablation

Compare standard PSO with four isolated openness rules at 100D on shifted Sphere,
Rosenbrock, Levy and Ackley: 10 matched seeds, 50 particles, 100,000 evaluations.
Run `python -u -m experiments.openness_ablation --workers 4` to save raw trials,
timings, paired comparisons and convergence plots with bootstrap confidence bands.
See [protocol, uncertainty and output definitions](docs/openness_ablation.md).
The final selection adds Expanded Schaffer F6, rotated ellipsoid, rotated
Rastrigin, rotated Ackley and one composition function. Use
`experiments/configs/openness_ablation_final9.json` for all nine functions with
the same 100D protocol. Reports paginate across every selected function.

## Four-trait studies

The curiosity, aggressiveness, openness, conformity and factorial HPPSO studies
use 59 independent vectorized paths per seed/function. The selected full study
uses shifted Sphere, rotated Rastrigin and one composition at 100D, ten seeds
and 100,000 evaluations per path. Parameters vary one at a time; all 16 trait
combinations support interaction matrices and leave-one-trait-out ablations.
See [protocol, local/server commands and interpretation](docs/trait_study.md).
The expanded trait study adds five reproducibly sampled shifted cases, three
rotated cases and two compositions. `python -m experiments.run_trait_extension`
runs the local continuation, combines all 13 functions, audits and packages the
results while preserving the completed original trials.
