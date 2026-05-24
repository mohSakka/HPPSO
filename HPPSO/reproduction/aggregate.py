"""Aggregate metrics, ranks, and statistical tests from merged datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from reproduction.config import (
    ALGORITHM_DISPLAY_NAMES,
    FUNCTION_NAMES,
    PAPER_ALGORITHM_ORDER_1000D,
    PAPER_ALGORITHM_ORDER_30D,
)
from reproduction.schema import MergedDataset


def internal_to_paper(algo: str) -> str:
    return ALGORITHM_DISPLAY_NAMES.get(algo, algo)


def paper_to_internal(name: str) -> str | None:
    for internal, paper in ALGORITHM_DISPLAY_NAMES.items():
        if paper == name:
            return internal
    return None


def summarize_performance(dataset: MergedDataset) -> pd.DataFrame:
    rows: list[dict] = []
    for func in FUNCTION_NAMES:
        for (f, algo), scores in dataset.best_costs.items():
            if f != func:
                continue
            arr = np.asarray(scores, dtype=float)
            rows.append(
                {
                    "function": func,
                    "algorithm_internal": algo,
                    "algorithm": internal_to_paper(algo),
                    "mean": float(np.nanmean(arr)),
                    "std": float(np.nanstd(arr)),
                    "n_runs": int(np.sum(~np.isnan(arr))),
                }
            )
    df = pd.DataFrame(rows)
    df["rank"] = df.groupby("function")["mean"].rank(method="average", ascending=True)
    return df.sort_values(["function", "rank"]).reset_index(drop=True)


def average_ranks(dataset: MergedDataset) -> pd.DataFrame:
    perf = summarize_performance(dataset)
    avg = perf.groupby("algorithm")["rank"].mean().sort_values()
    return avg.reset_index().rename(columns={"rank": "average_rank"})


def wilcoxon_pairwise_scores(
    dataset: MergedDataset,
    *,
    exclude_functions: set[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank scoring used in the paper (Tables 6 & 7).

    For each algorithm A, compare against every other algorithm B by pooling
    paired run scores across all functions (and runs), then count wins/ties.
    Normalized score = (wins + 0.5 * ties) / total * 100.
    """
    exclude_functions = exclude_functions or set()
    functions = [f for f in FUNCTION_NAMES if f not in exclude_functions]
    algorithms = dataset.algorithms()

    algo_data = {algo: {func: [] for func in functions} for algo in algorithms}
    for (func, algo), values in dataset.best_costs.items():
        if func in exclude_functions:
            continue
        if algo in algo_data:
            algo_data[algo][func] = np.asarray(values, dtype=float)

    results: list[dict] = []
    for algo in algorithms:
        wins = ties = losses = 0
        for other in algorithms:
            if algo == other:
                continue

            a_all: list[float] = []
            b_all: list[float] = []
            for func in functions:
                a = algo_data[algo][func]
                b = algo_data[other][func]
                if len(a) == 0 or len(b) == 0:
                    continue
                n = min(len(a), len(b))
                a = a[:n]
                b = b[:n]
                mask = ~np.isnan(a) & ~np.isnan(b)
                a = a[mask]
                b = b[mask]
                if len(a) < 2:
                    continue
                a_all.extend(a.tolist())
                b_all.extend(b.tolist())

            a_all_arr = np.asarray(a_all, dtype=float)
            b_all_arr = np.asarray(b_all, dtype=float)
            if len(a_all_arr) < 2:
                continue

            try:
                _, p = wilcoxon(a_all_arr, b_all_arr)
            except ValueError:
                ties += 1
                continue

            mean_a = float(np.mean(a_all_arr))
            mean_b = float(np.mean(b_all_arr))
            if p >= alpha:
                ties += 1
            elif mean_a < mean_b:
                wins += 1
            else:
                losses += 1

        total = wins + ties + losses
        norm = (wins + 0.5 * ties) / total * 100 if total else 0.0
        results.append(
            {
                "algorithm": internal_to_paper(algo),
                "algorithm_internal": algo,
                "total_wins": wins,
                "total_ties": ties,
                "normalized_score": norm,
            }
        )

    return pd.DataFrame(results).sort_values("normalized_score", ascending=False).reset_index(drop=True)


def median_convergence_curve(runs: list[list[float]]) -> np.ndarray:
    if not runs:
        return np.array([])
    max_len = max(len(r) for r in runs if r)
    padded = []
    for run in runs:
        if not run:
            padded.append(np.full(max_len, np.nan))
            continue
        arr = np.asarray(run, dtype=float)
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)), mode="edge")
        padded.append(arr)
    return np.nanmedian(np.vstack(padded), axis=0)


def paper_algorithm_order(dimension: int) -> list[str]:
    return PAPER_ALGORITHM_ORDER_30D if dimension == 30 else PAPER_ALGORITHM_ORDER_1000D
