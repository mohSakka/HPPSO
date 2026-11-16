"""Statistical analysis helpers for benchmark comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def rank_algorithms(results_df: pd.DataFrame, score_col: str = "Mean Best Score") -> pd.DataFrame:
    """Rank algorithms within each function group (lower is better)."""
    ranked = results_df.copy()
    valid = np.isfinite(ranked[score_col])
    complete = valid.groupby(ranked["Function"]).transform("all")
    ranked["Rank"] = ranked[score_col].where(complete).groupby(ranked["Function"]).rank(method="average")
    return ranked


def average_ranks_per_function(results_df: pd.DataFrame, score_col: str = "Mean Best Score") -> pd.DataFrame:
    ranked = rank_algorithms(results_df, score_col)
    pivot = ranked.pivot_table(index="Function", columns="Algorithm", values="Rank", aggfunc="mean")
    return pivot


def overall_average_ranks(results_df: pd.DataFrame, score_col: str = "Mean Best Score") -> pd.Series:
    return average_ranks_per_function(results_df, score_col).mean(axis=0).sort_values()


def wilcoxon_vs_baseline(
    scores_by_algo: dict[str, list[float]],
    baseline: str,
) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank test against a baseline algorithm."""
    rows = []
    base = scores_by_algo[baseline]
    for algo, scores in scores_by_algo.items():
        if algo == baseline:
            continue
        a, b = np.asarray(scores, dtype=float), np.asarray(base, dtype=float)
        if a.shape != b.shape:
            raise ValueError("Wilcoxon requires aligned paired runs")
        if a.size < 2 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            stat, p = np.nan, np.nan
        elif np.array_equal(a, b):
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(a, b, alternative="two-sided")
        rows.append({"Algorithm": algo, "Statistic": stat, "p-value": p})
    return pd.DataFrame(rows)
