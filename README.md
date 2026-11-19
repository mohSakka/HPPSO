# HPPSO

Human Personality based Particle Swarm Optimization — a PSO variant driven by
four personality traits (curiosity, conformity, aggressiveness, openness),
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

## Repository layout

- [src/hppso/](src/hppso/) contains the algorithms, benchmark definitions, and runners.
- [experiments/](experiments/) contains research protocols, configurations, and analysis scripts.
- [paper_results/](paper_results/) contains the revised paper results and their provenance.
- [notebooks/](notebooks/) contains standalone tutorials and the MAGD demonstration.
- [tests/](tests/) contains regression tests.

## Reproducing the revised results

Start with [the data and reproduction notes](paper_results/README.md). The
mathematical benchmark uses 30 functions at 30, 300, 600, and 1000 dimensions,
20 seeds per setting, and 1,000 objective evaluations per dimension. The
CEC2011 and neural-network studies use their own protocols. Run scripts retain
their experiment paths for working with the full trial archives.

For the algorithm and protocols, see [the HPPSO description](docs/algorithm.md),
[the mathematical benchmark guide](docs/final_benchmark.md), and
[the experiment harness](experiments/README.md).
The earlier pickle-based reproduction utilities are retained for historical
archives and regression tests.

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

## Research protocols

- [Final mathematical benchmark](docs/final_benchmark.md) covers the four dimensions and the resumable runner.
- [CEC2011 and neural-network data](paper_results/README.md) identify the final trial settings and result files.
- [Component and sensitivity configurations](experiments/configs/) and [diversity protocols](docs/diversity_study.md) document the supporting studies.
