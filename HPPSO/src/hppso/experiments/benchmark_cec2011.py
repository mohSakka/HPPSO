"""Benchmark HPPSO and competitors on CEC2011 real-world problems."""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from hppso.benchmarks.cec2011 import load_cec2011_problems
from hppso.experiments.runners import DEFAULT_ALGORITHMS, run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Benchmark on CEC2011 problems")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=200)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="results/cec2011")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    problems = load_cec2011_problems()
    print(f"Loaded {len(problems)} CEC2011 problems")

    results = run_benchmark(
        DEFAULT_ALGORITHMS,
        problems,
        num_runs=args.runs,
        pop_size=args.pop_size,
        max_iters=args.max_iters,
    )

    rows = []
    for algo, prob_results in results.items():
        for prob_name, data in prob_results.items():
            rows.append(
                {
                    "Problem": prob_name,
                    "Algorithm": algo,
                    "Mean Best Score": data["avg_final_score"],
                }
            )

    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "cec2011_results.csv"), index=False)
    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
