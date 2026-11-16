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
| `dataset` | nn_training only | A dataset key from `hppso-train-nn --help` |
| `seed` | optional | Defaults to 42 |
| `output.save_csv`, `output.save_json`, `output.save_plots` | optional | Defaults true |

To add a new experiment: drop a YAML in `configs/`, reference one of the three
`experiment_type` values, and it's runnable.
# Neural network manuscript rerun

After installing the project with `python -m pip install -e .`, run the 20 real
datasets with five optimizer seeds and a fixed budget of 6,000 training evaluations:

```sh
hppso-train-nn --dataset all --algorithm all --num-seeds 5 --seed 0 --pop-size 30 --budget 6000 --output experiments/results/nn_predictive_rerun
```

Use `--tables-only` to generate the dataset and architecture CSVs without downloading
data or running optimizers. On first use, scikit-learn downloads California Housing
and the external datasets by their pinned OpenML IDs. California Housing uses the
first 2,000 rows of a seeded random permutation, sampled without replacement before
splitting. Each hidden width is the floor of two thirds of the preceding width.
The optimization vector includes all weights and biases. Classification uses one
output per class, and regression uses one output per target.

Each dataset uses a fixed 60/20/20 row-level train/validation/test split, shared
across optimizers and seeds. Classification splits are stratified. Feature
imputation and standardization are fitted on the training split, as is regression
target standardization. Both the split and California Housing subsampling use
seed 42. Classification uses one-hot targets, softmax probabilities and
cross-entropy. Regression uses linear outputs and MSE on standardized targets
for training, then reports held-out MSE, RMSE and MAE in original target units.
Accuracy, macro-F1 and cross-entropy are reported for held-out classification.
The returned model minimizes training loss among the evaluated candidates.
Validation and test metrics are measured on that frozen model. Repeated optimizer
seeds measure optimizer variability on a fixed split. Small datasets such as
Linnerud have correspondingly small held-out partitions.

One evaluation is one full-training-set loss computation for one parameter vector.
A fused loss and gradient counts as one evaluation, and its backward pass is
also recorded in a separate gradient-call counter. Adam uses full batches and
learning rate 0.001, with parameter bounds shared with the other optimizers.
Equal evaluation counts allow comparison under a common loss-call budget.
Runtime and gradient counts capture additional computational differences.

`--algorithm all` includes Adam and the adaptive neural-network MAGD from
`notebooks/magd_algorithm_demo.ipynb`. MAGD uses the demonstration learning rate
of 0.03, per-agent Adam moments, gradient norm clipping at 1, and the notebook's
two pruning rules. It starts with the shared population size and can shrink
during training. All candidate losses, including initialization, count toward
the same evaluation budget. A fused loss and gradient is computed once per
candidate, and the gradient is reused if that agent survives. The benchmark
uses the shared model, objective, and parameter bounds. The evaluation budget
replaces the notebook's epoch cap and early stopping rule.

The output includes settings, exact split row indices, dataset and architecture
tables, per-seed metrics, evaluation counts, elapsed time, and convergence
histories. The summary exports contain training-loss medians and held-out
metric medians and quartiles. Use a fresh output directory for each training run.
