"""Main entry point for reproducing paper figures and tables from existing results."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

from reproduction.aggregate import average_ranks
from reproduction.config import DIMENSIONS, OUTPUT_FIGURES_DIR, OUTPUT_TABLES_DIR, REPO_ROOT
from reproduction.export_shift_vectors import export_all_shift_vectors
from reproduction.loaders import list_result_files
from reproduction.merge import merge_dimension
from reproduction.report import build_coverage_report, save_report
from reproduction.schema import MergedDataset
from reproduction.tables import save_tables
from reproduction.visualize import (
    plot_all_algorithms_convergence,
    plot_average_rank_bar,
    plot_hppso_median_panels,
)


def save_merged_dataset(dataset: MergedDataset) -> dict[str, Path]:
    """Write intermediate merged artifacts."""
    dim = dataset.dimension
    paths: dict[str, Path] = {}

    csv_path = REPO_ROOT / f"merged_results_{dim}D.csv"
    pkl_path = REPO_ROOT / f"merged_results_{dim}D.pkl"

    df = dataset.to_long_dataframe()
    df.to_csv(csv_path, index=False)
    with pkl_path.open("wb") as fh:
        pickle.dump(
            {
                "best_costs": dataset.best_costs,
                "convergence_histories": dataset.convergence_histories,
                "shift_vectors": dataset.shift_vectors,
                "merge_report": dataset.merge_report,
            },
            fh,
        )
    paths["csv"] = csv_path
    paths["pkl"] = pkl_path
    return paths


def print_inventory() -> None:
    print("Result files discovered:")
    for p in list_result_files():
        print(f"  - {p.name} ({p.stat().st_size / 1e6:.2f} MB)")


def main() -> int:
    print("=" * 72)
    print("HPPSO paper reproduction (existing results only)")
    print("=" * 72)
    print_inventory()

    print("\n--- Exporting shift vectors ---")
    shift_manifest = export_all_shift_vectors()
    for dim, meta in shift_manifest.items():
        src = meta["source_pickle"]
        src_name = src.name if isinstance(src, Path) else src
        print(
            f"  {dim}D: results/{src_name} -> {meta['exported_csv'].name} "
            f"({meta['n_rows']} functions)"
        )

    datasets: dict[int, MergedDataset] = {}
    generated: dict[str, Path | None] = {}

    for dim in DIMENSIONS:
        print(f"\n--- Merging {dim}D results ---")
        ds = merge_dimension(dim)
        datasets[dim] = ds
        merged_paths = save_merged_dataset(ds)
        print(f"Saved merged dataset: {merged_paths['csv'].name}, {merged_paths['pkl'].name}")

        mr = ds.merge_report
        print(f"  Algorithms: {', '.join(ds.algorithms())}")
        print(f"  CMA-ES replaced: {mr.replaced_cmaes_from_separate_file}")
        if mr.warnings:
            for w in mr.warnings[:5]:
                print(f"  WARNING: {w}")

        table_paths = save_tables(ds)
        print("  Tables written:")
        for name, path in table_paths.items():
            print(f"    {name}: {path.name}")

        fig_num = 7 if dim == 30 else 8
        fig_path = plot_hppso_median_panels(ds, figure_number=fig_num)
        generated[f"figure_{fig_num}"] = fig_path
        print(f"  Figure: {fig_path}")

        ranks = average_ranks(ds)
        rank_fig = plot_average_rank_bar(ds, ranks)
        print(f"  Supplementary: {rank_fig}")

        for func_key in ("f1_sphere", "f7_ackley", "f6_rastrigin"):
            sup = plot_all_algorithms_convergence(ds, func_key)
            if sup:
                print(f"  Supplementary: {sup.name}")

    report = build_coverage_report(datasets, generated)
    report_path = save_report(report, datasets, shift_manifest=shift_manifest)
    print(f"\nCoverage report: {report_path}")

    counts = report.summary_counts()
    print("\nCoverage summary:")
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n}")

    # Print quick comparison for Table 4/5
    print("\nReproduced average ranks:")
    for dim in DIMENSIONS:
        ranks = pd.read_csv(OUTPUT_TABLES_DIR / f"table_{4 if dim == 30 else 5:02d}_average_ranks_{dim}d.csv")
        print(f"  {dim}D:\n{ranks.to_string(index=False)}")

    print(f"\nOutputs:\n  Figures -> {OUTPUT_FIGURES_DIR}\n  Tables  -> {OUTPUT_TABLES_DIR}")
    print(f"  Shift vectors -> {REPO_ROOT / 'shift_vectors_30D.csv'}, {REPO_ROOT / 'shift_vectors_1000D.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
