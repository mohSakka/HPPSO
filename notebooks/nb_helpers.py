"""
Shared helpers for cleaned HPPSO analysis notebooks.

These utilities load pre-computed result files only — they do not run benchmarks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve repository root (HPPSO package directory).
NOTEBOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NOTEBOOK_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPRODUCED_TABLES = ARTIFACTS_DIR / "reproduced_tables"
REPRODUCED_FIGURES = ARTIFACTS_DIR / "reproduced_figures"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def ensure_reproduction_imports():
    """Import reproduction package after sys.path is set."""
    from reproduction.merge import merge_dimension  # noqa: F401
    from reproduction.aggregate import (  # noqa: F401
        average_ranks,
        summarize_performance,
        wilcoxon_pairwise_scores,
        median_convergence_curve,
        internal_to_paper,
    )
    from reproduction.config import (  # noqa: F401
        FUNCTION_NAMES,
        FIGURE_PANEL_FUNCTIONS,
        HPPSO_ALGORITHM_KEY,
        PAPER_ALGORITHM_ORDER_30D,
        PAPER_ALGORITHM_ORDER_1000D,
    )
    from reproduction.visualize import (  # noqa: F401
        plot_hppso_median_panels,
        plot_all_algorithms_convergence,
        plot_average_rank_bar,
    )
    from reproduction.tables import save_tables  # noqa: F401
    from reproduction.export_shift_vectors import export_all_shift_vectors  # noqa: F401


def load_merged(dimension: int):
    """Load merged dataset for 30D or 1000D from existing pickles."""
    from reproduction.merge import merge_dimension

    return merge_dimension(dimension)


def best_costs_to_dataframe(dataset) -> pd.DataFrame:
    """Long-format best-cost table from a MergedDataset."""
    rows = []
    for (func, algo), scores in dataset.best_costs.items():
        for run_idx, score in enumerate(scores):
            rows.append(
                {
                    "function": func,
                    "algorithm_internal": algo,
                    "algorithm": internal_to_paper(algo),
                    "run": run_idx,
                    "best_cost": float(score),
                }
            )
    return pd.DataFrame(rows)


def internal_to_paper(algo: str) -> str:
    from reproduction.aggregate import internal_to_paper as _map

    return _map(algo)


def performance_pivot(dataset) -> pd.DataFrame:
    """Mean best cost per function × algorithm (paper display names)."""
    perf = summarize_performance_from_dataset(dataset)
    return perf.pivot(index="function", columns="algorithm", values="mean")


def summarize_performance_from_dataset(dataset) -> pd.DataFrame:
    from reproduction.aggregate import summarize_performance

    return summarize_performance(dataset)


def list_result_files() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(p.name for p in RESULTS_DIR.glob("*.pkl"))


# Plot style used across cleaned notebooks
PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.35,
    "legend.fontsize": 9,
}
