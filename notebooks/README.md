# HPPSO Notebooks

## Tutorial notebooks

| Notebook | Cleaned version | Description |
|----------|-----------------|-------------|
| `01_benchmark_20_functions.ipynb` | `01_benchmark_20_functions_cleaned.ipynb` | 20 shifted classical benchmarks (30D) |
| `02_benchmark_cec2011.ipynb` | `02_benchmark_cec2011_cleaned.ipynb` | CEC2011 real-world problems |
| `03_neural_network_training.ipynb` | `03_neural_network_training_cleaned.ipynb` | MLP training with metaheuristics |

## Original research notebooks (preserved)

| Original | Cleaned refactor |
|----------|------------------|
| `original/HPPSO_Final_Reviiew.ipynb` | `original/HPPSO_Final_Reviiew_cleaned.ipynb` |
| `original/CEC2011_HPPSO_Review_.ipynb` | `original/CEC2011_HPPSO_Review_cleaned.ipynb` |

Original files are **not modified**. Cleaned copies reorganize the workflow for clarity and reproducibility.

## Shared utilities

- `nb_helpers.py` — path setup, merged data loading, plot style constants.

## Regenerating cleaned notebooks

```bash
python scripts/build_cleaned_notebooks.py
```

## Recommended workflow

1. **Paper reproduction (no re-runs):** open `original/HPPSO_Final_Reviiew_cleaned.ipynb` or run `python -m reproduction.run_reproduction`.
2. **Quick tutorial:** `01_benchmark_20_functions_cleaned.ipynb` with `USE_SAVED_RESULTS = True`.
3. **Live experiments:** set demo flags only when you intend to re-run optimizers.

## Result file dependencies

See `REPRODUCTION_REPORT.md` for the full list of pickles under `results/`.
