"""Export shift vectors from result pickles to CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reproduction.config import ARTIFACTS_DIR, FILE_PATTERNS, FUNCTION_NAMES, REPO_ROOT, RESULTS_DIR
from reproduction.loaders import load_shift_vectors


def shift_vector_source_path(dimension: int) -> Path:
    """Return the canonical pickle path used for shift vectors at a given dimension."""
    if dimension == 30:
        return RESULTS_DIR / FILE_PATTERNS["shift_vectors_30d"]
    if dimension == 1000:
        primary = RESULTS_DIR / FILE_PATTERNS["shift_vectors_1000d"]
        if primary.exists():
            return primary
        return RESULTS_DIR / FILE_PATTERNS["shift_vectors_1000d_alt"]
    raise ValueError(f"Unsupported dimension: {dimension}")


def _vector_to_csv_field(vec: list[float] | np.ndarray) -> str:
    """Serialize a shift vector as space-separated values in one CSV cell."""
    return " ".join(f"{float(v):.17g}" for v in vec)


def shift_vectors_to_dataframe(shift_vectors: dict[str, list[list[float]]]) -> pd.DataFrame:
    """
    One row per function.

    Columns: function, shifting_vector (all D dimensions in a single cell).

    The pickle stores one vector per seed, but vectors are identical across seeds
    for each function; seed 0 is used.
    """
    rows: list[dict[str, str]] = []
    for func in FUNCTION_NAMES:
        runs = shift_vectors.get(func, [])
        if not runs:
            rows.append({"function": func, "shifting_vector": ""})
            continue
        vec = np.asarray(runs[0], dtype=float)
        rows.append({"function": func, "shifting_vector": _vector_to_csv_field(vec)})
    return pd.DataFrame(rows)


def export_shift_vectors(dimension: int, output_dir: Path | None = None) -> Path:
    """Load shift vectors for one dimension and write CSV (20 rows, one per function)."""
    output_dir = output_dir or ARTIFACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    shift_vectors = load_shift_vectors(dimension)
    df = shift_vectors_to_dataframe(shift_vectors)

    out_path = output_dir / f"shift_vectors_{dimension}D.csv"
    try:
        df.to_csv(out_path, index=False)
    except PermissionError:
        # e.g. file open in Excel — write beside reproduced_tables instead
        fallback_dir = output_dir / "reproduced_tables"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        out_path = fallback_dir / out_path.name
        df.to_csv(out_path, index=False)
    return out_path


def export_all_shift_vectors(output_dir: Path | None = None) -> dict[int, dict[str, Path | str | int]]:
    """Export 30D and 1000D shift vectors; return metadata for reporting."""
    output_dir = output_dir or ARTIFACTS_DIR
    manifest: dict[int, dict[str, Path | str | int]] = {}

    for dim in (30, 1000):
        source = shift_vector_source_path(dim)
        csv_path = export_shift_vectors(dim, output_dir)
        manifest[dim] = {
            "source_pickle": source,
            "exported_csv": csv_path,
            "n_functions": len(FUNCTION_NAMES),
            "n_coordinates": dim,
            "n_rows": len(FUNCTION_NAMES),
        }

    return manifest
