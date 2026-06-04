# HPPSO

Human Personality based Particle Swarm Optimization — a PSO variant driven by
four personality traits (curiosity, confidence, aggressiveness, openness),
benchmarked against PSO, PSO-m, PSO-RIW, GA-MPC, GWO, SHADE, CMA-ES,
Sep-CMA-ES, and CSA on 20 shifted classical functions and the CEC2011 suite.

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

Run a full benchmark from a YAML config:

```bash
uv run hppso-run-experiment classical_30d
# or:
uv run python experiments/run.py experiments/configs/classical_30d.yaml
```

Outputs land in `experiments/results/<name>/<timestamp>/`.

## Layout

```
src/hppso/        # Library: algorithms, benchmarks, runners, nn, utils
experiments/      # YAML-driven harness (configs/, run.py)
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
