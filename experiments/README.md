# experiments/

Config-driven harness around `hppso.experiments.runners.run_benchmark`.
One YAML per experiment, one runner, timestamped output directories.

## Layout

```
experiments/
├── run.py                       # Dispatcher
├── configs/
│   ├── classical_30d.yaml       # 20-function shifted bench at 30D
│   ├── classical_1000d.yaml     # 20-function shifted bench at 1000D
│   ├── cec2011.yaml             # CEC2011 real-world bench (needs minionpy)
│   └── nn_diabetes.yaml         # MLP weight training on diabetes
└── results/                     # gitignored — populated at runtime
    └── <name>/<YYYYMMDD_HHMMSS>/
        ├── config_used.yaml
        ├── metrics.json
        └── results.csv          # (classical / cec2011 only)
```

## Run

```bash
uv run python experiments/run.py experiments/configs/classical_30d.yaml
uv run python experiments/run.py classical_30d       # shorthand
uv run hppso-run-experiment classical_30d            # console entry point
```

Override the output directory with `--output-dir <path>`.

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
| `output.save_csv`, `output.save_json` | optional | Defaults true |

To add a new experiment: drop a YAML in `configs/`, reference one of the three
`experiment_type` values, and it's runnable.
