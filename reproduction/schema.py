"""Normalized internal schema for merged benchmark results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MergeReport:
    dimension: int
    replaced_cmaes_from_separate_file: bool = False
    cmaes_keys_replaced: list[tuple[str, str]] = field(default_factory=list)
    cmaes_keys_missing_in_cmaes_file: list[tuple[str, str]] = field(default_factory=list)
    stale_cmaes_keys_dropped: list[tuple[str, str]] = field(default_factory=list)
    shift_vector_mismatch_functions: list[str] = field(default_factory=list)
    missing_algorithms: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class MergedDataset:
    """Unified experiment index for one dimensionality."""

    dimension: int
    # {(function, internal_algo): list[run_scores]}
    best_costs: dict[tuple[str, str], list[float]]
    # {(function, internal_algo): list[list[iteration_cost]]}
    convergence_histories: dict[tuple[str, str], list[list[float]]]
    shift_vectors: dict[str, list[list[float]]]
    merge_report: MergeReport

    def to_long_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for (func, algo), scores in self.best_costs.items():
            for run_idx, score in enumerate(scores):
                rows.append(
                    {
                        "dimension": self.dimension,
                        "function": func,
                        "algorithm_internal": algo,
                        "run": run_idx,
                        "best_cost": float(score) if score is not None else np.nan,
                    }
                )
        return pd.DataFrame(rows)

    def algorithms(self) -> list[str]:
        return sorted({k[1] for k in self.best_costs})

    def functions(self) -> list[str]:
        return sorted({k[0] for k in self.best_costs})
