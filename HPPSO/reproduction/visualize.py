"""Generate publication-style figures from merged datasets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reproduction.aggregate import internal_to_paper, median_convergence_curve
from reproduction.config import (
    ALGORITHM_DISPLAY_NAMES,
    FIGURE_PANEL_FUNCTIONS,
    HPPSO_ALGORITHM_KEY,
    OUTPUT_FIGURES_DIR,
)
from reproduction.schema import MergedDataset

# Consistent colors for multi-algorithm supplementary plots.
ALGO_COLORS = {
    "Original PSO": "#1f77b4",
    "PSO-m": "#ff7f0e",
    "PSO-RIW": "#2ca02c",
    "HPPSO_Modified": "#d62728",
    "GA-MPC": "#9467bd",
    "GWO": "#8c564b",
    "SHADE": "#e377c2",
    "CSA": "#7f7f7f",
    "CMAES": "#bcbd22",
    "Sep-CMA-ES": "#17becf",
}


def _prepare_log_curve(curve: np.ndarray) -> np.ndarray:
    c = np.asarray(curve, dtype=float).copy()
    c[c <= 0] = np.finfo(float).eps
    return c


def plot_hppso_median_panels(
    dataset: MergedDataset,
    *,
    figure_number: int,
    output_dir: Path | None = None,
) -> Path:
    """
    Reproduce Figure 7 (30D) or Figure 8 (1000D):
    3x5 grid of median HPPSO convergence curves.
    """
    output_dir = output_dir or OUTPUT_FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 5, figsize=(18, 10), sharex=True)
    axes_flat = axes.flatten()

    for ax, (func_key, title) in zip(axes_flat, FIGURE_PANEL_FUNCTIONS):
        key = (func_key, HPPSO_ALGORITHM_KEY)
        runs = dataset.convergence_histories.get(key, [])
        median = median_convergence_curve(runs)
        if median.size:
            ax.plot(_prepare_log_curve(median), color=ALGO_COLORS[HPPSO_ALGORITHM_KEY], linewidth=1.8)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.set_xlabel("Iterations", fontsize=8)
        ax.set_ylabel("Global Best Fitness", fontsize=8)

    dim_label = f"{dataset.dimension}D"
    fig.suptitle(
        f"Figure {figure_number}: Median convergence curves of HPPSO on "
        f"{dim_label} benchmark functions over 20 independent runs",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out = output_dir / f"figure_{figure_number:02d}_hppso_median_convergence_{dataset.dimension}d.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_all_algorithms_convergence(
    dataset: MergedDataset,
    func_key: str,
    *,
    output_dir: Path | None = None,
) -> Path | None:
    """Supplementary per-function convergence plot with all available algorithms."""
    output_dir = output_dir or (OUTPUT_FIGURES_DIR / "supplementary")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for algo in sorted({k[1] for k in dataset.convergence_histories if k[0] == func_key}):
        key = (func_key, algo)
        runs = dataset.convergence_histories.get(key, [])
        median = median_convergence_curve(runs)
        if median.size == 0:
            continue
        label = internal_to_paper(algo)
        color = ALGO_COLORS.get(algo)
        ax.plot(_prepare_log_curve(median), label=label, color=color, linewidth=1.5)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_yscale("log")
    ax.set_title(f"Median convergence on {func_key} ({dataset.dimension}D)")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Global Best Fitness")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = output_dir / f"convergence_{func_key}_{dataset.dimension}d_all_algorithms.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_average_rank_bar(dataset: MergedDataset, avg_ranks: "pd.DataFrame", *, output_dir: Path | None = None) -> Path:
    import pandas as pd

    output_dir = output_dir or OUTPUT_FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    df = avg_ranks.sort_values("average_rank")
    ax.barh(df["algorithm"], df["average_rank"], color="#4c72b0")
    ax.invert_yaxis()
    ax.set_xlabel("Average Rank (lower is better)")
    ax.set_title(f"Average algorithm ranks ({dataset.dimension}D, 20 functions)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = output_dir / f"average_ranks_{dataset.dimension}d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
