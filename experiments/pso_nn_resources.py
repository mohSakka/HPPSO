"""Measure PSO and HPPSO on the same Digits training workload."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter, process_time


def measure(algorithm, seed):
    import numpy as np
    import psutil
    import scipy
    import sklearn
    from threadpoolctl import threadpool_info, threadpool_limits

    from hppso.experiments.train_neural_network import (
        BOUNDS, SPLIT_SEED, layer_parameter_counts, layer_widths, load_partitions, train,
    )
    from hppso.nn.simple_mlp import SimpleNeuralNetwork
    from hppso.nn.training import loss_and_gradient

    partitions, _, _ = load_partitions("digits", sample_cap=2000)
    X, y = partitions["train"]
    widths = layer_widths(X.shape[1], y.shape[1])
    warmup = SimpleNeuralNetwork(*widths)
    process = psutil.Process()
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        loss_and_gradient(warmup, X, y, classification=True)
        del warmup
        before = process.memory_info()
        wall_start = perf_counter()
        cpu_start = process_time()
        _, oracle = train(algorithm, X, y, 30, 3000, seed=seed, classification=True)
        cpu_seconds = process_time() - cpu_start
        wall_seconds = perf_counter() - wall_start
        after = process.memory_info()
    if oracle.evaluations != 3000 or not np.isfinite(oracle.best_score):
        raise RuntimeError("Invalid training trial")
    return {
        "algorithm": algorithm,
        "seed": seed,
        "evaluations": oracle.evaluations,
        "best_training_loss": oracle.best_score,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "baseline_rss_bytes": before.rss,
        "pretraining_peak_rss_bytes": before.peak_wset,
        "peak_rss_bytes": after.peak_wset,
    }, {
        "dataset": "digits",
        "split_seed": SPLIT_SEED,
        "train_rows": len(X),
        "layer_widths": widths,
        "parameters": sum(layer_parameter_counts(widths)),
        "population": 30,
        "bounds": BOUNDS,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "psutil": psutil.__version__,
        "threadpools": pools,
    }


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[key] = "1"
    pso_first = set(random.Random(42).sample(range(30), 15))
    writer = None
    with (output / "nn_trials.csv").open("x", encoding="utf-8", newline="") as stream:
        for seed in range(30):
            order = ("PSO", "HPPSO") if seed in pso_first else ("HPPSO", "PSO")
            for position, algorithm in enumerate(order, start=1):
                result = subprocess.run(
                    [sys.executable, __file__, "--trial", algorithm, str(seed)],
                    env=env, check=True, capture_output=True, text=True,
                )
                row, metadata = json.loads(result.stdout)
                row["order"] = position
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    (output / "nn_environment.json").write_text(
                        json.dumps(metadata, indent=2), encoding="utf-8"
                    )
                writer.writerow(row)
                stream.flush()
                print(f"{seed + 1}/30 seed pairs, {algorithm}, {row['wall_seconds']:.2f} s, "
                      f"{row['peak_rss_bytes'] / 2**20:.1f} MiB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trial", nargs=2, metavar=("ALGORITHM", "SEED"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("This experiment uses Windows peak working set counters")
    if args.trial:
        algorithm, seed = args.trial
        print(json.dumps(measure(algorithm, int(seed))))
    elif args.output:
        run(args.output)
    else:
        parser.error("--output is required")


if __name__ == "__main__":
    main()
