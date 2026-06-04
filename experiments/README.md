# experiments/

Config-driven harness around `hppso.experiments.runners.run_benchmark`.
One YAML per experiment, one runner, timestamped output directories.

## Layout

```
experiments/
├── run.py                       # Dispatcher
├── plots.py                     # Plot generation (secondary)
├── configs/
│   ├── classical_30d.yaml       # 20-function shifted bench at 30D
│   ├── classical_1000d.yaml     # 20-function shifted bench at 1000D
│   ├── cec2011.yaml             # CEC2011 real-world bench (needs minionpy)
│   └── nn_diabetes.yaml         # MLP weight training on diabetes
└── results/                     # gitignored — populated at runtime
    └── <name>/<YYYYMMDD_HHMMSS>/
        ├── config_used.yaml
        ├── metrics.json
        ├── results.csv          # (classical / cec2011 only)
        └── plots/                # avg_ranks, wins, per-function convergence
```

## Run

```bash
uv run python experiments/run.py experiments/configs/classical_30d.yaml
uv run python experiments/run.py classical_30d       # shorthand
uv run hppso-run-experiment classical_30d            # console entry point
```

Override the output directory with `--output-dir <path>`.

## Plots (secondary)

Each finished run also gets a `plots/` subfolder with:

| Plot | Generated for |
|---|---|
| `avg_ranks_bar.png` | classical / cec2011 |
| `wins_per_algorithm.png` | classical / cec2011 |
| `convergence_<func>.png` (one per highlight function) | classical / cec2011 |
| `train_test_mse.png` | nn_training |

Plot generation is best-effort — a failure logs a warning instead of failing the run.

```bash
# Skip plots on a fresh run:
uv run hppso-run-experiment classical_30d --no-plots

# Regenerate plots only, without rerunning the experiment:
uv run hppso-run-experiment --plots-from experiments/results/classical_30d/<timestamp>
```

Disable plots permanently per-config by setting `output.save_plots: false` in the YAML.

## Config schema

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Used as the subfolder under `results/` |
| `experiment_type` | yes | `classical` \| `cec2011` \| `nn_training` |
| `algorithms` | yes | List of names from `hppso.experiments.runners.DEFAULT_ALGORITHMS` (or `PSO`/`PSO-m`/`PSO-RIW`/`HPPSO` for `nn_training`) |
| `runs`, `pop_size`, `max_iters` | yes | Per-algorithm settings |
| `dim` | classical only | Problem dimensionality (30 or 1000) |
| `dataset` | nn_training only | `diabetes` \| `wine` |
| `seed` | optional | Defaults to 42 |
| `output.save_csv`, `output.save_json`, `output.save_plots` | optional | Defaults true |

To add a new experiment: drop a YAML in `configs/`, reference one of the three
`experiment_type` values, and it's runnable.
