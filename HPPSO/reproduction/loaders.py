"""Load existing pickle result files without rerunning experiments."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from reproduction.config import FILE_PATTERNS, RESULTS_DIR


def _resolve_path(key: str) -> Path:
    rel = FILE_PATTERNS[key]
    path = RESULTS_DIR / rel
    if not path.exists():
        raise FileNotFoundError(f"Expected result file not found: {path}")
    return path


def load_pickle(key: str) -> Any:
    path = _resolve_path(key)
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_pickle_if_exists(key: str) -> Any | None:
    try:
        return load_pickle(key)
    except FileNotFoundError:
        return None


def load_convergence_histories(dimension: int) -> dict[tuple[str, str], list[list[float]]]:
    if dimension == 30:
        return load_pickle("convergence_30d_main")
    if dimension == 1000:
        data = load_pickle_if_exists("convergence_1000d")
        if data is None:
            data = load_pickle("convergence_1000d_alt")
        return data
    raise ValueError(f"Unsupported dimension: {dimension}")


def load_cmaes_only_histories_30d() -> dict[tuple[str, str], list[list[float]]]:
    return load_pickle("convergence_30d_cmaes")


def load_best_costs_30d() -> dict[tuple[str, str], list[float]] | None:
    return load_pickle_if_exists("best_costs_30d")


def load_shift_vectors(dimension: int) -> dict[str, list[list[float]]]:
    if dimension == 30:
        return load_pickle("shift_vectors_30d")
    data = load_pickle_if_exists("shift_vectors_1000d")
    if data is None:
        data = load_pickle("shift_vectors_1000d_alt")
    return data


def load_cmaes_shift_vectors_30d() -> dict[str, list[list[float]]]:
    return load_pickle("shift_vectors_30d_cmaes")


def list_result_files() -> list[Path]:
    return sorted(RESULTS_DIR.glob("*.pkl"))
