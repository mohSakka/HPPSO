"""Statistical analysis helpers for benchmark comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def rank_algorithms(results_df: pd.DataFrame, score_col: str = "Mean Best Score") -> pd.DataFrame:
    """Rank algorithms within each function group (lower is better)."""
    ranked = results_df.copy()
    ranked["Rank"] = ranked.groupby("Function")[score_col].rank(method="average")
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
        stat, p = wilcoxon(scores, base, alternative="two-sided")
        rows.append({"Algorithm": algo, "Statistic": stat, "p-value": p})
    return pd.DataFrame(rows)
