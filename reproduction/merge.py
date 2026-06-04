"""Merge CMA-ES and non-CMA-ES result files by dimension and configuration."""

from __future__ import annotations

import copy

import numpy as np

from reproduction.config import (
    CMAES_ALGORITHM_KEY,
    FUNCTION_NAMES,
    NUM_RUNS,
)
from reproduction.loaders import (
    load_best_costs_30d,
    load_cmaes_only_histories_30d,
    load_convergence_histories,
    load_shift_vectors,
)
from reproduction.schema import MergeReport, MergedDataset


def _final_scores_from_histories(
    histories: dict[tuple[str, str], list[list[float]]],
) -> dict[tuple[str, str], list[float]]:
    best_costs: dict[tuple[str, str], list[float]] = {}
    for key, runs in histories.items():
        finals = []
        for run in runs:
            if not run:
                finals.append(float("nan"))
            else:
                finals.append(float(run[-1]))
        best_costs[key] = finals
    return best_costs


def merge_dimension(dimension: int) -> MergedDataset:
    """
    Build a unified dataset for one dimensionality.

    30D merge policy:
      - Non-CMA algorithms come from `all_convergence_histories 30d.pkl`.
      - CMAES is replaced from `all_convergence_histories 30d cmaes only.pkl`
        (authoritative CMA-ES rerun file).
      - best_costs prefer `best_costs_30d.pkl` for non-CMA; CMAES finals are taken
        from the cmaes-only histories.
      - Shift vectors from `all_svs30d.pkl` (20 seeds per function).

    1000D merge policy:
      - Single file `all_convergence_histories1000.pkl` already contains CMAES.
      - best_costs are derived from convergence history finals.
      - Shift vectors from `all_svs 1000.pkl` (20 seeds per function).
    """
    report = MergeReport(dimension=dimension)
    histories = copy.deepcopy(load_convergence_histories(dimension))
    shift_vectors = copy.deepcopy(load_shift_vectors(dimension))

    if dimension == 30:
        cmaes_histories = load_cmaes_only_histories_30d()

        # Drop stale embedded CMAES entries before replacement.
        stale_keys = [k for k in histories if k[1] == CMAES_ALGORITHM_KEY]
        for key in stale_keys:
            report.stale_cmaes_keys_dropped.append(key)
            histories.pop(key)

        report.replaced_cmaes_from_separate_file = True
        for key, runs in cmaes_histories.items():
            func, algo = key
            if algo != CMAES_ALGORITHM_KEY:
                report.warnings.append(f"Unexpected algorithm in CMA-ES-only file: {key}")
                continue
            histories[key] = runs
            report.cmaes_keys_replaced.append(key)

        # Every function should have CMAES after merge.
        for func in FUNCTION_NAMES:
            key = (func, CMAES_ALGORITHM_KEY)
            if key not in histories:
                report.cmaes_keys_missing_in_cmaes_file.append(key)

        best_costs = load_best_costs_30d()
        if best_costs is None:
            best_costs = _final_scores_from_histories(histories)
            report.warnings.append("best_costs_30d.pkl missing; derived from histories.")
        else:
            best_costs = copy.deepcopy(best_costs)
            # Replace CMAES scores with cmaes-only finals.
            for key in list(best_costs):
                if key[1] == CMAES_ALGORITHM_KEY:
                    del best_costs[key]
            best_costs.update(_final_scores_from_histories(cmaes_histories))

    elif dimension == 1000:
        best_costs = _final_scores_from_histories(histories)
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")

    # Validate run counts and completeness.
    for func in FUNCTION_NAMES:
        for algo in {k[1] for k in histories}:
            key = (func, algo)
            if key not in histories:
                report.warnings.append(f"Missing convergence histories for {key}")
                continue
            n_runs = len(histories[key])
            if n_runs != NUM_RUNS:
                report.warnings.append(f"{key}: expected {NUM_RUNS} runs, found {n_runs}")

    present_algos = sorted({k[1] for k in best_costs})
    if CMAES_ALGORITHM_KEY not in present_algos and dimension == 30:
        report.warnings.append("CMAES missing after merge.")
    if "Sep-CMA-ES" not in present_algos and dimension == 1000:
        report.missing_algorithms.append("Sep-CMA-ES")

    return MergedDataset(
        dimension=dimension,
        best_costs=best_costs,
        convergence_histories=histories,
        shift_vectors=shift_vectors,
        merge_report=report,
    )
