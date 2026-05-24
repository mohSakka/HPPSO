# HPPSO — Human Personality Based Particle Swarm Optimization

A research codebase for **HPPSO**, a human personality-based variant of Particle Swarm Optimization, with comprehensive benchmarking against state-of-the-art metaheuristics.

## Features

- **HPPSO algorithm** with human-inspired personality traits: curiosity, confidence, aggressiveness, and sociality/openness
- **10 competitor algorithms**: PSO, PSO-m, PSO-RIW, GA-MPC, GWO, SHADE, CMA-ES, Sep-CMA-ES, CSA
- **20 shifted classical benchmark functions** with fixed shift vectors for reproducibility
- **CEC2011 real-world problems** (22 test cases via `minionpy`)
- **Neural network weight training** using metaheuristics as optimizers

## Repository Structure

```
HPPSO/
├── src/hppso/              # Main Python package
│   ├── algorithms/         # PSO, HPPSO, GWO, SHADE, CMA-ES, GA-MPC, CSA
│   ├── benchmarks/         # 20 classical functions + CEC2011 loader
│   ├── experiments/        # Runnable benchmark and training scripts
│   ├── nn/                 # Simple MLP for metaheuristic weight training
│   └── utils/              # Plotting and statistical analysis
├── notebooks/
│   ├── original/           # Original research notebooks (preserved)
│   └── *.ipynb             # Clean tutorial notebooks
├── docs/
│   ├── algorithm.md        # Algorithm description
│   └── papers/             # Manuscript PDF
├── results/                # Benchmark outputs (gitignored)
├── requirements.txt
└── pyproject.toml
```

## Installation

```bash
cd HPPSO
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[all]"
```

For CEC2011 benchmarks only:

```bash
pip install -e ".[cec2011]"
```

## Quick Start

### Run HPPSO on Sphere function

```python
import numpy as np
from hppso import HPPSO

def sphere(x):
    return np.sum(x**2)

opt = HPPSO(sphere, n_pop=30, dimensions=30, max_it=500, bounds=(-100, 100))
best_score, history = opt.optimize()
print(f"Best fitness: {best_score:.2e}")
```

### Benchmark on 20 shifted functions

```bash
python -m hppso.experiments.benchmark_classical --dim 30 --runs 20 --max-iters 500
```

Results are saved to `results/classical/`.

### Benchmark on CEC2011

```bash
python -m hppso.experiments.benchmark_cec2011 --runs 10 --pop-size 200 --max-iters 1000
```

### Train neural network weights with HPPSO

```bash
python -m hppso.experiments.train_neural_network --algorithm HPPSO --dataset diabetes
```

## Benchmark Functions

Twenty classical functions with coordinate shifting (`BenchmarkFunctions`):

| Group | Functions |
|-------|-----------|
| Unimodal | Sphere, Schwefel 2.22, Zakharov, Rosenbrock, Powell Sum |
| Multimodal | Rastrigin, Ackley, Griewank, Weierstrass, Levy |
| Heterogeneous | Schwefel 2.26, Styblinski-Tang, Michalewicz, Alpine, Happy Cat, HGBat, Bent Cigar, Discus, Penalized, Zakharov-2 |

Each function uses a fixed shift vector stored in `results/classical/shift_vectors.json` for reproducible comparisons.

## Algorithms Compared

| Algorithm | Module | Key Idea |
|-----------|--------|----------|
| PSO | `algorithms/pso.py` | Canonical particle swarm |
| PSO-m | `algorithms/pso.py` | PSO + Gaussian mutation |
| PSO-RIW | `algorithms/pso.py` | Random inertia weight |
| **HPPSO** | `algorithms/hppso.py` | Human personality-based PSO |
| GA-MPC | `algorithms/ga_mpc.py` | Multi-parent crossover GA |
| GWO | `algorithms/gwo.py` | Grey Wolf Optimizer |
| SHADE | `algorithms/shade.py` | Adaptive differential evolution |
| CMA-ES | `algorithms/cmaes.py` | Covariance matrix adaptation |
| Sep-CMA-ES | `algorithms/sep_cmaes.py` | Separable CMA-ES |
| CSA | `algorithms/csa.py` | Circle Search Algorithm |

## Documentation

- Algorithm details: [`docs/algorithm.md`](docs/algorithm.md)
- Manuscript: [`docs/papers/HPPSO_Manuscript.pdf`](docs/papers/HPPSO_Manuscript.pdf)
- Original notebooks: [`notebooks/original/`](notebooks/original/)

## Original Notebooks

The original research notebooks are preserved in `notebooks/original/`:

| Notebook | Description |
|----------|-------------|
| `HPPSO_Final_Reviiew.ipynb` | 20-function shifted benchmark, Wilcoxon analysis, NN training |
| `CEC2011_HPPSO_Review_.ipynb` | CEC2011 real-world benchmark, Friedman/Nemenyi tests |

## Citation

If you use this code, please cite the accompanying manuscript in `docs/papers/`.

## License

Research and academic use. See manuscript for authorship details.
