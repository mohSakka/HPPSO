"""Generate paper-style tables from merged datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reproduction.aggregate import (
    average_ranks,
    paper_algorithm_order,
    summarize_performance,
    wilcoxon_pairwise_scores,
)
from reproduction.config import OUTPUT_TABLES_DIR
from reproduction.schema import MergedDataset


def _format_scientific(value: float) -> str:
    if np.isnan(value) or np.isinf(value):
        return "inf" if np.isinf(value) else "nan"
    return f"{value:.2E}"


def build_performance_table(dataset: MergedDataset) -> pd.DataFrame:
    """
    Tables 2 (30D) and 3 (1000D): mean, std, rank per function and algorithm.
    """
    perf = summarize_performance(dataset)
    order = paper_algorithm_order(dataset.dimension)
    rows: list[dict] = []

    for func in perf["function"].unique():
        sub = perf[perf["function"] == func]
        row_mean: dict[str, str | float] = {"function": func, "metric": "mean"}
        row_std: dict[str, str | float] = {"function": func, "metric": "std"}
        row_rank: dict[str, str | float] = {"function": func, "metric": "rank"}

        for paper_algo in order:
            match = sub[sub["algorithm"] == paper_algo]
            if match.empty:
                row_mean[paper_algo] = np.nan
                row_std[paper_algo] = np.nan
                row_rank[paper_algo] = np.nan
            else:
                rec = match.iloc[0]
                row_mean[paper_algo] = rec["mean"]
                row_std[paper_algo] = rec["std"]
                row_rank[paper_algo] = rec["rank"]

        rows.extend([row_mean, row_std, row_rank])

    return pd.DataFrame(rows)


def build_performance_table_formatted(dataset: MergedDataset) -> pd.DataFrame:
    raw = build_performance_table(dataset)
    formatted_rows: list[dict] = []
    for func in raw["function"].unique():
        block = raw[raw["function"] == func]
        mean_row = block[block["metric"] == "mean"].iloc[0]
        std_row = block[block["metric"] == "std"].iloc[0]
        rank_row = block[block["metric"] == "rank"].iloc[0]

        out: dict[str, str] = {"Function": func}
        for col in paper_algorithm_order(dataset.dimension):
            m = mean_row[col]
            s = std_row[col]
            r = rank_row[col]
            if pd.isna(m):
                out[col] = "—"
            else:
                out[f"{col}_mean"] = _format_scientific(float(m))
                out[f"{col}_std"] = _format_scientific(float(s))
                out[f"{col}_rank"] = int(r) if not pd.isna(r) else "—"
        formatted_rows.append(out)

    # Flatten to paper-like triple rows
    paper_rows: list[dict] = []
    for func in raw["function"].unique():
        block = raw[raw["function"] == func]
        for metric in ("mean", "std", "rank"):
            row = block[block["metric"] == metric].iloc[0]
            entry = {"Function": func if metric == "mean" else "", "Row": metric}
            for col in paper_algorithm_order(dataset.dimension):
                val = row[col]
                if metric == "rank" and not pd.isna(val):
                    entry[col] = int(val)
                elif metric in ("mean", "std") and not pd.isna(val):
                    entry[col] = _format_scientific(float(val))
                else:
                    entry[col] = val
            paper_rows.append(entry)
    return pd.DataFrame(paper_rows)


def save_tables(dataset: MergedDataset, *, output_dir: Path | None = None) -> dict[str, Path]:
    output_dir = output_dir or OUTPUT_TABLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    dim = dataset.dimension
    paths: dict[str, Path] = {}

    perf = summarize_performance(dataset)
    perf_path = output_dir / f"table_performance_{dim}d.csv"
    perf.to_csv(perf_path, index=False)
    paths["performance_long"] = perf_path

    perf_fmt = build_performance_table_formatted(dataset)
    table_num = 2 if dim == 30 else 3
    fmt_path = output_dir / f"table_{table_num:02d}_performance_{dim}d_formatted.csv"
    perf_fmt.to_csv(fmt_path, index=False)
    paths["performance_formatted"] = fmt_path

    ranks = average_ranks(dataset)
    table_num_rank = 4 if dim == 30 else 5
    rank_path = output_dir / f"table_{table_num_rank:02d}_average_ranks_{dim}d.csv"
    ranks.to_csv(rank_path, index=False)
    paths["average_ranks"] = rank_path

    exclude = {"f2_schwefel_2_22"} if dim == 1000 else set()
    wilcoxon = wilcoxon_pairwise_scores(dataset, exclude_functions=exclude)
    table_num_w = 6 if dim == 30 else 7
    w_path = output_dir / f"table_{table_num_w:02d}_wilcoxon_{dim}d.csv"
    wilcoxon.to_csv(w_path, index=False)
    paths["wilcoxon"] = w_path

    return paths
