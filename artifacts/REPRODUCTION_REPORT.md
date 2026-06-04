# HPPSO Paper Reproduction Report

This report documents reproduction of figures and tables using **only** pre-generated files in `results/`.
No experiments were rerun.

## Coverage Summary

- **reproduced**: 5
- **partial**: 3
- **missing**: 3
- **unsupported**: 7

## Item-by-Item Coverage

| Paper Item | Status | Notes |
|---|---|---|
| Figure 1 | unsupported | HPPSO framework diagram; no ablation result files in results/ |
| Figure 2 | unsupported | Conformity ablation; no ablation result files in results/ |
| Figure 3 | unsupported | Aggressiveness ablation; no ablation result files in results/ |
| Figure 4 | unsupported | Curiosity ablation; no ablation result files in results/ |
| Figure 5 | unsupported | Openness ablation; no ablation result files in results/ |
| Figure 6 | unsupported | Openness hyperparameter sensitivity; no ablation result files in results/ |
| Figure 7 | reproduced | HPPSO median convergence, 30D, 15 panels |
| Figure 8 | reproduced | HPPSO median convergence, 1000D, 15 panels |
| Table 1 | unsupported | Function definitions; static content from paper |
| Table 2 | reproduced | 30D mean/std/rank from merged best costs |
| Table 4 | reproduced | 30D average ranks |
| Table 6 | reproduced | 30D Wilcoxon pairwise scores |
| Table 3 | partial | 1000D performance table; SEP-CMAES column missing from result files |
| Table 5 | partial | 1000D average ranks; SEP-CMAES missing |
| Table 7 | partial | 1000D Wilcoxon scores; SEP-CMAES absent; f2 excluded per notebook convention |
| Table 8 | missing | CEC2011 benchmark outputs not present in results/ |
| Table 9 | missing | CEC2011 benchmark outputs not present in results/ |
| Table 10 | missing | CEC2011 benchmark outputs not present in results/ |

## Validation Checks

- Table 2 sanity check f1 Sphere PSO mean: paper≈1.410E+04, reproduced=1.414E+04 (PASS)

## Paper vs Reproduced Highlights

| Metric | Paper | Reproduced | Match |
|---|---:|---:|:---:|
| Table 4 SHADE avg rank (30D) | 1.92 | 1.85 | ~ |
| Table 4 CMAES avg rank (30D) | 2.95 | 2.75 | ~ |
| Table 4 HPPSO avg rank (30D) | 4.12 | 4.20 | ~ |
| Table 6 CMAES Wilcoxon score | 93.75 | 93.75 | exact |
| Table 6 SHADE Wilcoxon score | 93.75 | 93.75 | exact |
| Table 6 HPPSO Wilcoxon score | 75.00 | 75.00 | exact |
| Table 2 f1 PSO mean (30D) | 1.41e4 | 1.41e4 | exact |
| Table 5 HPPSO avg rank (1000D) | 2.93 | 2.90 | ~ |
| Table 7 top score (1000D) | SEP-CMAES 100 | CMAES 100 | partial* |

*Paper ranks SEP-CMA-ES first at 1000D; repository stores CMAES only (no sep-CMA-ES file).

## Inferred Result Schema

| File pattern | Content | Key structure |
|---|---|---|
| `all_convergence_histories 30d.pkl` | 30D histories, 9 algos incl. stale CMAES | `(function, algorithm) -> list[20 runs][iterations]` |
| `all_convergence_histories 30d cmaes only.pkl` | 30D CMAES-only rerun | `(function, 'CMAES') -> list[20 runs][iterations]` |
| `all_convergence_histories1000.pkl` | 1000D histories, all algos | same tuple schema |
| `best_costs_30d.pkl` | 30D final costs per run | `(function, algorithm) -> list[20 floats]` |
| `all_svs30d.pkl` | **30D shift vectors** | `function -> list[20 seeds][30]` (identical across seeds) |
| `all_svs 1000.pkl` | **1000D shift vectors** | `function -> list[20 seeds][1000]` (identical across seeds) |

Normalized merged schema: `MergedDataset` with `best_costs`, `convergence_histories`, `shift_vectors`, `merge_report`.

## Assumptions

- `HPPSO_Modified` in pickles maps to paper label **HPPSO**.
- `CMAES` maps to paper **CMAES**; **SEP-CMA-ES** is absent from all result files.
- `all_convergence_histories.pkl` duplicates 1000D data; `all_convergence_histories1000.pkl` is preferred.
- 30D CMA-ES results are taken from the dedicated cmaes-only convergence file.
- Shift vectors are fixed per function at each dimension (see Shift Vectors section).
- 1000D Wilcoxon table excludes `f2_schwefel_2_22` (all algorithms return overflow/inf), matching the original notebook.
- Figures 7–8 plot **HPPSO only** (median over 20 runs), not all-algorithm overlays.

## Shift Vectors (Fixed Benchmark Offsets)

Each shifted benchmark function uses a fixed coordinate offset **o** (identical across all
20 random seeds). One row per function; all dimensions appear in the `shifting_vector` column.

| Dimension | Source pickle (`results/`) | Exported CSV (repo root) | Rows |
|---|---|---|---|
| 30D | `results/all_svs30d.pkl` | `shift_vectors_30D.csv` | 20 functions (vector length 30) |
| 1000D | `results/all_svs 1000.pkl` | `shift_vectors_1000D.csv` | 20 functions (vector length 1000) |

CSV columns: `function`, `shifting_vector` (space-separated values, one row per function).

These files define the shifted problem instances used in the benchmark experiments at each
dimension.

## Merge Decisions

### 30D
- CMA-ES replaced from separate file: **True**
- Stale embedded CMA-ES keys dropped: **20**
- CMA-ES keys merged from cmaes-only file: **20**
### 1000D
- CMA-ES replaced from separate file: **False**
- Stale embedded CMA-ES keys dropped: **0**
- CMA-ES keys merged from cmaes-only file: **0**
- Missing algorithms vs paper: **Sep-CMA-ES**
