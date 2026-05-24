"""Benchmark HPPSO and competitors on 20 shifted classical functions."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from hppso.benchmarks.classical import FUNCTION_NAMES, build_problem_suite
from hppso.experiments.runners import DEFAULT_ALGORITHMS, run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Benchmark on 20 shifted classical functions")
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="results/classical")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(42)
    problems = build_problem_suite(dim=args.dim, random_shift=False)

    results = run_benchmark(
        DEFAULT_ALGORITHMS,
        problems,
        num_runs=args.runs,
        pop_size=args.pop_size,
        max_iters=args.max_iters,
    )

    rows = []
    for algo, prob_results in results.items():
        for func_name, data in prob_results.items():
            rows.append(
                {
                    "Function": func_name,
                    "Algorithm": algo,
                    "Mean Best Score": data["avg_final_score"],
                    "Std Best Score": float(np.nanstd(data["all_final_scores"])),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, "shifted_benchmark_results.csv"), index=False)

    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    shifts = {p["name"]: p["shift_vector"] for p in problems}
    with open(os.path.join(args.output_dir, "shift_vectors.json"), "w", encoding="utf-8") as f:
        json.dump(shifts, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")
    print(f"Functions tested: {len(FUNCTION_NAMES)}")


if __name__ == "__main__":
    main()
