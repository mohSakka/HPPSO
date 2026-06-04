#!/usr/bin/env python3
"""Build cleaned, refactored notebook copies (does not modify originals)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"


def _cell(cell_type: str, source: str, outputs=None) -> dict:
    c = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")[:-1]] + ([source.split("\n")[-1]] if source else []),
    }
    if cell_type == "code":
        c["outputs"] = outputs or []
        c["execution_count"] = None
    return c


def _md(text: str) -> dict:
    lines = text.strip() + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in lines.split("\n")[:-1]] + ([lines.split("\n")[-1]] if lines.strip() else []),
    }


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [l + "\n" for l in text.strip().split("\n")],
        "outputs": [],
        "execution_count": None,
    }


def _save(nb: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"Wrote {path.relative_to(REPO)} ({len(nb['cells'])} cells)")


def _nb(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def build_01_cleaned():
    cells = [
        _md(
            """# Benchmark: 20 Shifted Classical Functions (Cleaned)

## Purpose
Tutorial notebook comparing **HPPSO** with PSO variants, GA-MPC, GWO, SHADE, CMA-ES, Sep-CMA-ES, and CSA on 20 shifted benchmarks.

## Modes
- **`USE_SAVED_RESULTS = True`** (default): loads pre-computed pickles from `results/` — no optimization runs.
- **`USE_SAVED_RESULTS = False`**: runs a **small** demo benchmark (few runs/iterations). Increase settings only when you intend to re-run experiments.

## Inputs
- `results/all_convergence_histories 30d.pkl`, `best_costs_30d.pkl`, CMA-ES merge files (see reproduction pipeline).

## Outputs
- Summary tables and optional convergence plots in `reproduced_figures/`.
- Aligns with paper **Tables 2, 4, 6** and **Figure 7** (30D)."""
        ),
        _md("## 1. Imports and configuration"),
        _code(
            """import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NOTEBOOK_DIR = Path.cwd()
if (NOTEBOOK_DIR / "nb_helpers.py").exists():
    sys.path.insert(0, str(NOTEBOOK_DIR))
elif (NOTEBOOK_DIR.parent / "notebooks" / "nb_helpers.py").exists():
    sys.path.insert(0, str(NOTEBOOK_DIR.parent / "notebooks"))
REPO_ROOT = NOTEBOOK_DIR.parent if (NOTEBOOK_DIR / "nb_helpers.py").exists() else NOTEBOOK_DIR.parent.parent

import nb_helpers as nh
nh.ensure_reproduction_imports()

from reproduction.config import FUNCTION_NAMES, DIMENSIONS
from reproduction.visualize import plot_hppso_median_panels, plot_all_algorithms_convergence
from reproduction.tables import save_tables
from hppso.benchmarks.classical import build_problem_suite
from hppso.experiments.runners import DEFAULT_ALGORITHMS, run_benchmark
from hppso.utils.statistics import overall_average_ranks

plt.rcParams.update(nh.PLOT_STYLE)

# --- Experiment settings (paper: 30D, 20 runs, pop=30, 500 iters) ---
DIM = 30
USE_SAVED_RESULTS = True  # set False only to run a lightweight demo

# Demo-only settings (ignored when USE_SAVED_RESULTS=True)
NUM_RUNS = 5
POP_SIZE = 30
MAX_ITERS = 200"""
        ),
        _md("## 2. Load pre-computed results"),
        _code(
            """if USE_SAVED_RESULTS:
    dataset = nh.load_merged(DIM)
    print(f"Loaded merged {DIM}D data — algorithms: {', '.join(dataset.algorithms())}")
    print("Result pickles:", ", ".join(nh.list_result_files()[:4]), "...")
else:
    dataset = None
    print("Will run live benchmark in next section (demo settings).")"""
        ),
        _md("## 3. Live benchmark (optional demo only)"),
        _code(
            """if not USE_SAVED_RESULTS:
    np.random.seed(42)
    problems = build_problem_suite(dim=DIM, random_shift=False)
    results = run_benchmark(
        DEFAULT_ALGORITHMS, problems,
        num_runs=NUM_RUNS, pop_size=POP_SIZE, max_iters=MAX_ITERS,
    )
    rows = []
    for algo, prob_results in results.items():
        for func, data in prob_results.items():
            rows.append({
                "Function": func, "Algorithm": algo,
                "Mean Best Score": data["avg_final_score"],
                "Std Best Score": np.nanstd(data["all_final_scores"]),
            })
    df = pd.DataFrame(rows)
else:
    raw = nh.best_costs_to_dataframe(dataset)
    df = (
        raw.groupby(["function", "algorithm"], as_index=False)
        .agg(Mean_Best_Score=("best_cost", "mean"), Std_Best_Score=("best_cost", "std"))
        .rename(columns={"function": "Function", "algorithm": "Algorithm", "Mean_Best_Score": "Mean Best Score", "Std_Best_Score": "Std Best Score"})
    )"""
        ),
        _md("## 4. Analysis — performance table and ranks"),
        _code(
            """display_df = df.rename(columns={"function": "Function", "algorithm": "Algorithm"}) if "function" in df.columns else df
pivot = display_df.pivot_table(index="Function", columns="Algorithm", values="Mean Best Score")
display(pivot)

rank_input = display_df.rename(columns={"Function": "function", "Algorithm": "algorithm", "Mean Best Score": "Mean Best Score"})
print("\\nOverall average ranks (lower is better):")
print(overall_average_ranks(rank_input))"""
        ),
        _md("## 5. Export tables (reproduced_tables/)"),
        _code(
            """if USE_SAVED_RESULTS:
    paths = save_tables(dataset)
    for k, p in paths.items():
        print(f"  {k}: {p.name}")"""
        ),
        _md("## 6. Visualization"),
        _code(
            """if USE_SAVED_RESULTS:
    out = nh.REPRODUCED_FIGURES
    out.mkdir(exist_ok=True)
    fig7 = plot_hppso_median_panels(dataset, figure_number=7, output_dir=out)
    print(f"Figure 7: {fig7.name}")
    plot_all_algorithms_convergence(dataset, "f1_sphere", output_dir=out / "supplementary")
    print("Supplementary convergence plot for f1_sphere saved.")"""
        ),
        _md(
            """## Notes
- CMA-ES 30D results are merged from `all_convergence_histories 30d cmaes only.pkl` (see `reproduction/merge.py`).
- Shift vectors: `reproduced_tables/shift_vectors_30D.csv` or `results/all_svs30d.pkl`.
- Full pipeline: `python -m reproduction.run_reproduction`"""
        ),
    ]
    _save(_nb(cells), NOTEBOOKS / "01_benchmark_20_functions_cleaned.ipynb")


def build_02_cleaned():
    cells = [
        _md(
            """# CEC2011 Real-World Benchmark (Cleaned)

## Purpose
Benchmark HPPSO and competitors on **CEC2011** problems via `minionpy`.

## Important
CEC2011 result pickles are **not** in the repository. This notebook documents the workflow and can run a **short demo** only.

## Paper outputs (when data available)
- **Tables 8–10**, CEC2011 convergence figures."""
        ),
        _md("## 1. Imports and configuration"),
        _code(
            """import sys
from pathlib import Path

import pandas as pd

NOTEBOOK_DIR = Path.cwd()
sys.path.insert(0, str(NOTEBOOK_DIR if (NOTEBOOK_DIR / "nb_helpers.py").exists() else NOTEBOOK_DIR.parent / "notebooks"))

from hppso.benchmarks.cec2011 import load_cec2011_problems
from hppso.experiments.runners import DEFAULT_ALGORITHMS, run_benchmark

# Demo settings — paper uses pop=200, max_iters=1000, 10 runs
RUN_DEMO = False
NUM_RUNS = 2
POP_SIZE = 30
MAX_ITERS = 50"""
        ),
        _md("## 2. Load problems"),
        _code(
            """try:
    problems = load_cec2011_problems()
    for p in problems[:5]:
        print(f"{p['name']} | dim={p['dimension']}")
    print(f"... total {len(problems)} problems")
except ImportError as e:
    print("minionpy not installed:", e)
    problems = []"""
        ),
        _md("## 3. Run benchmark (optional — expensive)"),
        _code(
            """if RUN_DEMO and problems:
    results = run_benchmark(DEFAULT_ALGORITHMS, problems, num_runs=NUM_RUNS, pop_size=POP_SIZE, max_iters=MAX_ITERS)
    rows = []
    for algo, prob_results in results.items():
        for name, data in prob_results.items():
            rows.append({"Problem": name, "Algorithm": algo, "Mean Score": data["avg_final_score"]})
    display(pd.DataFrame(rows).pivot_table(index="Problem", columns="Algorithm", values="Mean Score"))
else:
    print("Skipping live CEC2011 runs. Add result files under results/ and extend this notebook to load them.")"""
        ),
    ]
    _save(_nb(cells), NOTEBOOKS / "02_benchmark_cec2011_cleaned.ipynb")


def build_03_cleaned():
    cells = [
        _md(
            """# Neural Network Training with Metaheuristics (Cleaned)

## Purpose
Demonstrate optimizing MLP weights with HPPSO / PSO variants on the diabetes dataset.

## Settings
- Small network, short training — suitable for teaching, not full paper experiments.
- Paper NN experiments used multiple datasets and longer budgets (see original notebook)."""
        ),
        _md("## 1. Imports and configuration"),
        _code(
            """import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hppso.algorithms import HPPSO, PSO
from hppso.nn.simple_mlp import SimpleNeuralNetwork, mean_squared_error, nn_objective_function

plt.rcParams.update({"figure.figsize": (8, 5), "axes.grid": True})

POP_SIZE = 30
MAX_ITERS = 100
BOUNDS = (-2, 2)
RANDOM_SEED = 42"""
        ),
        _md("## 2. Prepare data"),
        _code(
            """np.random.seed(RANDOM_SEED)
data = load_diabetes()
X = StandardScaler().fit_transform(data.data)
y = data.target.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
print(f"Train {X_train.shape}, test {X_test.shape}")"""
        ),
        _md("## 3. Training helper"),
        _code(
            '''def train_mlp(algorithm: str, pop: int = POP_SIZE, iters: int = MAX_ITERS):
    """Train one MLP with a given optimizer; returns train MSE, test MSE, history."""
    nn = SimpleNeuralNetwork(X_train.shape[1], 16, 8, 1)
    n_weights = len(nn.get_weights_flat())
    objective = lambda w: nn_objective_function(w, nn, X_train, y_train)

    if algorithm == "HPPSO":
        opt = HPPSO(objective, n_pop=pop, dimensions=n_weights, max_it=iters, bounds=BOUNDS)
        train_mse, history = opt.optimize()
        nn.set_weights_flat(opt.get_best_position())
    else:
        kwargs = {}
        if algorithm == "PSO-m":
            kwargs = {"mutation_rate": 0.05, "gaussian_mutation_strength": 0.1}
        if algorithm == "PSO-RIW":
            kwargs = {"w_random_range": (0.4, 0.9), "mutation_rate": 0}
        pso = PSO(objective, [(BOUNDS[0], BOUNDS[1])] * n_weights, pop, iters, **kwargs)
        weights, train_mse, history = pso.optimize()
        nn.set_weights_flat(weights)

    test_mse = float(mean_squared_error(y_test, nn.forward(X_test)))
    return float(train_mse), test_mse, history'''
        ),
        _md("## 4. Compare algorithms"),
        _code(
            """import pandas as pd

results = []
for algo in ["PSO", "PSO-m", "PSO-RIW", "HPPSO"]:
    train_mse, test_mse, _ = train_mlp(algo)
    results.append({"Algorithm": algo, "Train MSE": train_mse, "Test MSE": test_mse})
    print(f"{algo:8s} | train={train_mse:.4f} | test={test_mse:.4f}")

pd.DataFrame(results)"""
        ),
        _md("## 5. Optional convergence plot (HPPSO)"),
        _code(
            """_, _, hist = train_mlp("HPPSO")
plt.figure()
plt.plot(hist)
plt.yscale("log")
plt.title("HPPSO training MSE on diabetes")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.grid(True, alpha=0.35)"""
        ),
    ]
    _save(_nb(cells), NOTEBOOKS / "03_neural_network_training_cleaned.ipynb")


def build_hppso_final_cleaned():
    cells = [
        _md(
            """# HPPSO — 20 Shifted Benchmarks: Analysis & Reproduction (Cleaned)

## Purpose
Refactored version of `original/HPPSO_Final_Reviiew.ipynb`. This notebook **does not re-run** the full benchmark suite. It loads existing pickles, merges CMA-ES results, and reproduces paper tables/figures.

## Original notebook scope (preserved logic)
- 20 shifted functions at **30D** and **1000D**
- 20 independent runs per setting
- Algorithms: PSO, PSO-M, PSO-RIW, HPPSO, GA-MPC, GWO, SHADE, CSA, CMA-ES
- Wilcoxon pairwise scoring, average ranks, convergence plots

## Result files required (`results/`)
| File | Content |
|------|---------|
| `all_convergence_histories 30d.pkl` | Non-CMA 30D histories |
| `all_convergence_histories 30d cmaes only.pkl` | CMA-ES 30D histories |
| `all_convergence_histories1000.pkl` | 1000D histories |
| `best_costs_30d.pkl` | 30D final costs |
| `all_svs30d.pkl`, `all_svs 1000.pkl` | Shift vectors |

## Paper artifacts produced
- **Tables 2–7**, **Figures 7–8**
- Exports under `reproduced_tables/`, `reproduced_figures/`"""
        ),
        _md("## 1. Imports and paths"),
        _code(
            """import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Locate repo root from notebooks/ or notebooks/original/
_cwd = Path.cwd()
if (_cwd / "nb_helpers.py").exists():
    sys.path.insert(0, str(_cwd))
    REPO_ROOT = _cwd.parent
elif (_cwd.parent / "nb_helpers.py").exists():
    sys.path.insert(0, str(_cwd.parent))
    REPO_ROOT = _cwd.parent.parent
else:
    REPO_ROOT = _cwd

import nb_helpers as nh
nh.ensure_reproduction_imports()

from reproduction.merge import merge_dimension
from reproduction.aggregate import (
    summarize_performance,
    average_ranks,
    wilcoxon_pairwise_scores,
    median_convergence_curve,
    internal_to_paper,
)
from reproduction.config import FUNCTION_NAMES, FIGURE_PANEL_FUNCTIONS, HPPSO_ALGORITHM_KEY
from reproduction.tables import save_tables
from reproduction.visualize import (
    plot_hppso_median_panels,
    plot_all_algorithms_convergence,
    plot_average_rank_bar,
)
from reproduction.export_shift_vectors import export_all_shift_vectors

plt.rcParams.update(nh.PLOT_STYLE)

RESULTS_DIR = nh.RESULTS_DIR
OUT_TABLES = nh.REPRODUCED_TABLES
OUT_FIGURES = nh.REPRODUCED_FIGURES
OUT_TABLES.mkdir(exist_ok=True)
OUT_FIGURES.mkdir(exist_ok=True)"""
        ),
        _md(
            """## 2. Merge policy (CMA-ES)

30D: embedded CMA-ES entries in the main pickle are **replaced** by `all_convergence_histories 30d cmaes only.pkl`.

1000D: single file already contains all algorithms.

Dimensions are never mixed."""
        ),
        _md("## 3. Load and merge datasets"),
        _code(
            """datasets = {}
merge_reports = {}
for dim in (30, 1000):
    ds = merge_dimension(dim)
    datasets[dim] = ds
    merge_reports[dim] = ds.merge_report
    print(f"\\n{dim}D: {len(ds.algorithms())} algorithms, {len(ds.functions())} functions")
    print(f"  CMA-ES from separate file: {ds.merge_report.replaced_cmaes_from_separate_file}")
    if ds.merge_report.warnings:
        for w in ds.merge_report.warnings[:3]:
            print(f"  note: {w}")"""
        ),
        _md("## 4. Performance summaries (Tables 2 & 3)"),
        _code(
            """for dim, ds in datasets.items():
    perf = summarize_performance(ds)
    print(f"\\n=== {dim}D mean best cost (first 5 rows) ===")
    display(perf.head())
    pivot = perf.pivot(index="function", columns="algorithm", values="mean")
    display(pivot.iloc[:5, :5])"""
        ),
        _md("## 5. Average ranks (Tables 4 & 5)"),
        _code(
            """for dim, ds in datasets.items():
    ranks = average_ranks(ds)
    print(f"\\n{dim}D average ranks:")
    display(ranks)
    rank_fig = plot_average_rank_bar(ds, ranks, output_dir=OUT_FIGURES)
    print(f"  saved {rank_fig.name}")"""
        ),
        _md(
            """## 6. Wilcoxon pairwise scores (Tables 6 & 7)

Score = (wins + 0.5 × ties) / total_comparisons × 100, comparing pooled run-level costs across all functions."""
        ),
        _code(
            """for dim, ds in datasets.items():
    exclude = {"f2_schwefel_2_22"} if dim == 1000 else set()
    wdf = wilcoxon_pairwise_scores(ds, exclude_functions=exclude)
    print(f"\\n{dim}D Wilcoxon ranking:")
    display(wdf)"""
        ),
        _md("## 7. Export CSV tables"),
        _code(
            """for dim, ds in datasets.items():
    paths = save_tables(ds, output_dir=OUT_TABLES)
    print(f"{dim}D:", ", ".join(p.name for p in paths.values()))"""
        ),
        _md("## 8. Shift vectors"),
        _code(
            """from pathlib import Path

manifest = export_all_shift_vectors(REPO_ROOT)
for dim, meta in manifest.items():
    print(f"{dim}D shift vectors -> {Path(meta['exported_csv']).name}")"""
        ),
        _md("## 9. Figures 7 & 8 — HPPSO median convergence (15 panels each)"),
        _code(
            """for dim in (30, 1000):
    fig_num = 7 if dim == 30 else 8
    path = plot_hppso_median_panels(datasets[dim], figure_number=fig_num, output_dir=OUT_FIGURES)
    print(f"Figure {fig_num} ({dim}D): {path}")"""
        ),
        _md("## 10. Supplementary — all-algorithm convergence (selected functions)"),
        _code(
            """example_functions = ["f1_sphere", "f6_rastrigin", "f7_ackley"]
for dim in (30, 1000):
    for func in example_functions:
        p = plot_all_algorithms_convergence(
            datasets[dim], func, output_dir=OUT_FIGURES / "supplementary"
        )
        if p:
            print(f"  {dim}D {func}: {p.name}")"""
        ),
        _md("## 11. Export long-format merged CSV"),
        _code(
            """for dim, ds in datasets.items():
    csv_path = REPO_ROOT / "artifacts" / f"merged_results_{dim}D.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_long_dataframe().to_csv(csv_path, index=False)
    print(f"Saved {csv_path.name}")"""
        ),
        _md(
            """## Removed from original (exploratory / duplicate)
- Inline redefinitions of PSO, GWO, SHADE, CMA-ES (now in `src/hppso/`)
- 2000D sphere demo runs
- Parameter sensitivity sweeps
- Duplicate empty cells and repeated CSV exports

## Full automation
Run `python -m reproduction.run_reproduction` for the same pipeline without Jupyter."""
        ),
    ]
    _save(_nb(cells), NOTEBOOKS / "original" / "HPPSO_Final_Reviiew_cleaned.ipynb")


def build_cec2011_cleaned():
    cells = [
        _md(
            """# CEC2011 HPPSO Review (Cleaned)

## Purpose
Refactored from `original/CEC2011_HPPSO_Review_.ipynb`. Documents CEC2011 benchmarking workflow using the `hppso` package.

## Data status
CEC2011 result files are **not** bundled in `results/`. Enable `RUN_DEMO` for a minimal live test only.

## Original content preserved conceptually
- Standardized algorithm wrappers
- Friedman / Nemenyi tests (when sufficient result rows exist)
- Median convergence plots"""
        ),
        _md("## 1. Imports"),
        _code(
            """import sys
from pathlib import Path

import numpy as np
import pandas as pd

NOTEBOOK_DIR = Path.cwd()
if (NOTEBOOK_DIR / "nb_helpers.py").exists():
    sys.path.insert(0, str(NOTEBOOK_DIR))
elif (NOTEBOOK_DIR.parent / "nb_helpers.py").exists():
    sys.path.insert(0, str(NOTEBOOK_DIR.parent))

from hppso.benchmarks.cec2011 import load_cec2011_problems
from hppso.experiments.runners import DEFAULT_ALGORITHMS, run_benchmark

RUN_DEMO = False"""
        ),
        _md("## 2. Load CEC2011 suite"),
        _code(
            """try:
    problems = load_cec2011_problems()
    print(f"Loaded {len(problems)} CEC2011 problems")
except ImportError as exc:
    print("Install optional dependency: pip install -e '.[cec2011]'")
    print(exc)
    problems = []"""
        ),
        _md("## 3. Optional demo run"),
        _code(
            """if RUN_DEMO and problems:
    # Use first 2 problems only for smoke test
    subset = problems[:2]
    results = run_benchmark(DEFAULT_ALGORITHMS, subset, num_runs=2, pop_size=30, max_iters=30)
    rows = []
    for algo, res in results.items():
        for name, d in res.items():
            rows.append({"problem": name, "algorithm": algo, "mean": d["avg_final_score"]})
    pd.DataFrame(rows)"""
        ),
        _md(
            """## 4. Statistical tests (template)

When CEC2011 pickles are available, load them and apply Friedman/Nemenyi via `scikit-posthocs` as in the original notebook. See `hppso.utils.statistics` for Wilcoxon helpers used on classical benchmarks."""
        ),
        _code(
            """# Placeholder: load CEC2011 results when files exist
# import pickle
# with open("results/cec2011_results.pkl", "rb") as f:
#     cec_results = pickle.load(f)"""
        ),
    ]
    _save(_nb(cells), NOTEBOOKS / "original" / "CEC2011_HPPSO_Review_cleaned.ipynb")


def main():
    build_01_cleaned()
    build_02_cleaned()
    build_03_cleaned()
    build_hppso_final_cleaned()
    build_cec2011_cleaned()
    print("Done.")


if __name__ == "__main__":
    main()
