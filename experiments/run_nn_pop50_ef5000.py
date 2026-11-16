"""Launch independent dataset runs for the 10-seed neural-network benchmark."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from hppso.experiments.train_neural_network import ALGORITHMS, DATASETS, OPENML


def completed(output, name, pop_size, budget, sample_cap):
    if not all((output / report).is_file() for report in
               ("held_out_metrics.csv", "table13_median_training_loss.csv")):
        return False
    try:
        settings = json.loads((output / "settings.json").read_text(encoding="utf-8"))
        if (settings["datasets"] != [name] or settings["algorithms"] != list(ALGORITHMS)
                or settings["seeds"] != list(range(10)) or settings["population"] != pop_size
                or settings["evaluations"] != budget or settings.get("sample_cap") != sample_cap):
            return False
        with (output / "trials.jsonl").open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        return (len(rows) == len(ALGORITHMS) * 10
                and all(row["dataset"] == name and row["evaluations"] == budget for row in rows)
                and {(row["algorithm"], row["seed"]) for row in rows}
                == {(algo, seed) for algo in ALGORITHMS for seed in range(10)})
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--group", choices=("local", "external", "all"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--sample-cap", type=int)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--algorithm", choices=(*ALGORITHMS, "all"), default="all")
    parser.add_argument("--hppso-selection", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", nargs="+", choices=DATASETS)
    args = parser.parse_args()
    external = {*OPENML, "california_housing"}
    selected = [name for name in DATASETS
                if args.group == "all" or (name in external) == (args.group == "external")]
    if args.only:
        selected = [name for name in args.only if name in selected]
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1")

    def run(name):
        output = args.output / name
        if (args.resume and args.algorithm == "all" and args.hppso_selection is None
                and completed(output, name, args.pop_size, args.budget, args.sample_cap)):
            print(f"{name}: exit 0 (verified saved trials)", flush=True)
            return 0
        with (args.output / f"{name}.log").open("a" if args.resume else "x", encoding="utf-8") as log:
            command = [sys.executable, "-u", "-m", "hppso.experiments.train_neural_network",
                       "--dataset", name, "--algorithm", args.algorithm, "--pop-size", str(args.pop_size),
                       "--budget", str(args.budget), "--num-seeds", "10", "--seed", "0",
                       "--output", str(output)]
            if args.sample_cap is not None:
                command += ["--sample-cap", str(args.sample_cap)]
            if args.resume:
                command.append("--resume")
            if args.hppso_selection:
                command += ["--hppso-selection", str(args.hppso_selection)]
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"{name}: exit {result.returncode}", flush=True)
        return result.returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        statuses = list(pool.map(run, selected))
    return int(any(statuses))


if __name__ == "__main__":
    raise SystemExit(main())
