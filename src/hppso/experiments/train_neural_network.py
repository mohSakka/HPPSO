"""Run the neural-network benchmarks from Section 6.2.3."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile
from time import perf_counter, time_ns
import zlib

import numpy as np
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hppso.experiments.runners import DEFAULT_ALGORITHMS
from hppso.nn.magd import SETTINGS as MAGD_SETTINGS, run_magd
from hppso.nn.simple_mlp import SimpleNeuralNetwork
from hppso.nn.training import BudgetReached, TrainingOracle, adam, predictive_metrics


# Label, task, sample count, input width, output width.
DATASETS = {
    "california_housing": ("California Housing", "Regression", 2000, 8, 1),
    "diabetes": ("Diabetes", "Regression", 442, 10, 1),
    "linnerud": ("Linnerud", "Regression", 20, 3, 3),
    "cpu_act": ("CPU Activity", "Regression", 8192, 21, 1),
    "elevators": ("Elevators", "Regression", 16599, 16, 1),
    "pol": ("Pole Telecommunication", "Regression", 15000, 26, 1),
    "ailerons": ("Ailerons", "Regression", 13750, 33, 1),
    "yprop": ("Yprop 4-1", "Regression", 8885, 42, 1),
    "superconduct": ("Superconductivity", "Regression", 21263, 79, 1),
    "mtp": ("MTP", "Regression", 4450, 202, 1),
    "wine": ("Wine", "Classification", 178, 13, 3),
    "breast_cancer": ("Breast Cancer", "Classification", 569, 30, 2),
    "digits": ("Digits", "Classification", 1797, 64, 10),
    "ionosphere": ("Ionosphere", "Classification", 351, 34, 2),
    "satimage": ("Statlog Landsat", "Classification", 6430, 36, 6),
    "texture": ("Texture", "Classification", 5500, 40, 11),
    "qsar_biodeg": ("QSAR Biodegradation", "Classification", 1055, 41, 2),
    "spambase": ("Spambase", "Classification", 4601, 57, 2),
    "mfeat_factors": ("Multiple Features (Factors)", "Classification", 2000, 216, 10),
    "semeion": ("Semeion", "Classification", 1593, 256, 10),
}
OPENML = dict(cpu_act=44132, elevators=44134, pol=44133, ailerons=44137, yprop=45032,
              superconduct=44148, mtp=405, ionosphere=59, satimage=182, texture=40499,
              qsar_biodeg=1494, spambase=44, mfeat_factors=12, semeion=1501)
ALGORITHMS = ("HPPSO", "PSO", "PSO-m", "PSO-RIW", "GWO", "CSA", "Sep-CMA-ES", "SHADE", "GA-MPC", "Adam", "MAGD")
SPLIT_SEED = 42
SUBSAMPLE_SEED = 42
BOUNDS = (-2.0, 2.0)


def layer_widths(inputs: int, outputs: int) -> tuple[int, int, int, int]:
    hidden1 = 2 * inputs // 3
    hidden2 = 2 * hidden1 // 3
    if hidden2 < 1 or outputs < 1:
        raise ValueError("The architecture requires two nonempty hidden layers and at least one output")
    return inputs, hidden1, hidden2, outputs


def layer_parameter_counts(widths):
    return [(left + 1) * right for left, right in zip(widths, widths[1:])]


def load_openml(data_id):
    options = dict(data_id=data_id, as_frame=True, return_X_y=True)
    try:
        return datasets.fetch_openml(**options)
    except (gzip.BadGzipFile, EOFError, zlib.error):
        pass
    except ValueError as exc:
        if "md5 checksum of local file" not in str(exc):
            raise
    cache = Path(datasets.get_data_home())
    recovery = cache.with_name(f"{cache.name}_recovery")
    return datasets.fetch_openml(data_home=recovery, **options)


def load_partitions(name: str, sample_cap: int | None = None):
    _, task, samples, inputs, outputs = DATASETS[name]
    if name in OPENML:
        X, y = load_openml(OPENML[name])
        X = X.to_numpy(dtype=float)
        y = y.to_numpy()
    else:
        loader = datasets.fetch_california_housing if name == "california_housing" else getattr(datasets, f"load_{name}")
        X, y = loader(return_X_y=True)

    original_samples = 20640 if name == "california_housing" else samples
    if X.shape != (original_samples, inputs):
        raise ValueError(f"{name} data dimensions differ from the benchmark manifest: {X.shape}")
    if sample_cap is not None:
        samples = min(samples, sample_cap)
    source_rows = np.arange(original_samples)
    if samples < original_samples:
        source_rows = np.random.default_rng(SUBSAMPLE_SEED).permutation(original_samples)[:samples]
        X, y = X[source_rows], y[source_rows]
    classification = task == "Classification"
    labels = np.asarray(y).reshape(-1) if classification else None
    if classification:
        classes, indices = np.unique(labels, return_inverse=True)
        if len(classes) != outputs:
            raise ValueError(f"{name} has {len(classes)} classes, expected {outputs}")
        y = np.eye(outputs)[indices]
    else:
        y = np.asarray(y, dtype=float).reshape(len(X), -1)
    if X.shape != (samples, inputs) or y.shape != (samples, outputs):
        raise ValueError(f"{name} data dimensions differ from the benchmark manifest")

    train, held_out = train_test_split(
        np.arange(samples), test_size=0.4, random_state=SPLIT_SEED, stratify=labels
    )
    validation, test = train_test_split(
        held_out, test_size=0.5, random_state=SPLIT_SEED,
        stratify=labels[held_out] if classification else None,
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[train])
    scaler = StandardScaler().fit(imputer.transform(X[train]))
    target_scaler = None if classification else StandardScaler().fit(y[train])
    partitions = {}
    split_rows = {}
    for key, indices in (("train", train), ("validation", validation), ("test", test)):
        features = scaler.transform(imputer.transform(X[indices]))
        targets = y[indices] if classification else target_scaler.transform(y[indices])
        partitions[key] = (features, targets)
        split_rows[key] = source_rows[indices].tolist()
    return partitions, target_scaler, split_rows


def load_dataset(name: str):
    partitions, _, _ = load_partitions(name)
    X_train, y_train = partitions["train"]
    X_test, y_test = partitions["test"]
    return X_train, X_test, y_train, y_test


def train_pso_variant(algo: str, X_train, y_train, pop: int, iters: int, *, seed=None):
    """Compatibility entry point for the original MSE-based demo."""
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    nn, oracle = train(algo, X_train, y_train, pop, pop * iters, seed=seed, classification=False)
    return nn, oracle.best_score, [score for _, score in oracle.trace]


def train(algo, X_train, y_train, pop, budget, *, seed, classification, hppso_parameters=None):
    y_train = np.asarray(y_train).reshape(len(X_train), -1)
    widths = layer_widths(X_train.shape[1], y_train.shape[1])
    nn = SimpleNeuralNetwork(*widths)
    dimension = len(nn.get_weights_flat())
    if dimension != sum(layer_parameter_counts(widths)):
        raise RuntimeError("The optimizer vector does not match the architecture")
    lower = np.full(dimension, BOUNDS[0])
    upper = np.full(dimension, BOUNDS[1])
    oracle = TrainingOracle(nn, X_train, y_train, classification, budget)
    try:
        if algo == "Adam":
            adam(oracle, lower, upper, seed)
        elif algo == "MAGD":
            run_magd(oracle, lower, upper, pop, seed)
        elif algo == "HPPSO" and hppso_parameters is not None:
            from hppso.algorithms.hppso import HPPSO

            np.random.seed(seed)
            HPPSO(oracle, n_pop=pop, dimensions=dimension, max_it=budget + 1,
                  bounds=(lower, upper), **hppso_parameters).optimize()
        else:
            runner = DEFAULT_ALGORITHMS["Original PSO" if algo == "PSO" else algo]
            args = (oracle, dimension, lower, upper, pop, budget + 1)
            if algo == "Sep-CMA-ES":
                runner(*args, budget + 1, seed, stopfitness=-np.inf)
            else:
                runner(*args, seed)
    except BudgetReached:
        pass
    if oracle.best_weights is None or oracle.evaluations != budget:
        raise RuntimeError(f"{algo} returned with {oracle.evaluations}/{budget} evaluations or no finite loss")
    nn.set_weights_flat(oracle.best_weights)
    if oracle.trace[-1][0] != budget:
        oracle.trace.append((budget, oracle.best_score))
    return nn, oracle


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def write_tables(output, sample_cap=None):
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "table11_datasets.csv", ["Dataset", "Task", "Samples", "Features", "Outputs", "Source"],
              [(label, task, min(samples, sample_cap) if sample_cap is not None else samples, inputs, outputs,
                f"https://www.openml.org/d/{OPENML[key]}" if key in OPENML else "scikit-learn")
               for key, (label, task, samples, inputs, outputs) in DATASETS.items()])
    rows = []
    for label, _, _, inputs, outputs in DATASETS.values():
        widths = layer_widths(inputs, outputs)
        counts = layer_parameter_counts(widths)
        rows.append((label, *widths, *counts, sum(counts)))
    write_csv(output / "table12_architectures.csv",
              ["Dataset", "Input", "Hidden 1", "Hidden 2", "Output", "Layer 1 parameters",
               "Layer 2 parameters", "Output parameters", "Dimension"], rows)


def write_durable(path, text):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_saved_trials(path, settings):
    raw = path.read_bytes()
    saved = {}
    damaged = False
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            damaged = True
            continue
        key = (row["dataset"], row["algorithm"], row["seed"])
        if (key in saved or key[0] not in settings["datasets"] or key[1] not in settings["algorithms"]
                or key[2] not in settings["seeds"] or row["evaluations"] != settings["evaluations"]):
            raise ValueError(f"Duplicate or mismatched saved trial {key}")
        saved[key] = row
    if damaged or (raw and not raw.endswith(b"\n")):
        backup = path.with_suffix(f".jsonl.interrupted.{time_ns()}")
        with backup.open("xb") as stream:
            stream.write(raw)
        write_durable(path, "".join(json.dumps(row, allow_nan=False) + "\n" for row in saved.values()))
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="diabetes")
    parser.add_argument("--algorithm", choices=[*ALGORITHMS, "all"], default="HPPSO")
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iters", type=int, default=200, help="Used only to derive the default evaluation budget")
    parser.add_argument("--budget", type=int, help="Full-training-set objective evaluations per run")
    parser.add_argument("--seed", type=int, default=0, help="First optimizer seed")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--sample-cap", type=int, help="Maximum rows sampled after a seeded shuffle, before splitting")
    parser.add_argument("--output", type=Path, default=Path("experiments/results/nn_manuscript"))
    parser.add_argument("--resume", action="store_true", help="Keep completed trials with identical settings")
    parser.add_argument("--hppso-selection", type=Path, help="JSON tuning report with HPPSO parameters")
    parser.add_argument("--tables-only", action="store_true", help="Generate Tables 11 and 12 without loading datasets")
    args = parser.parse_args()
    budget = args.budget if args.budget is not None else args.pop_size * args.max_iters
    if min(args.pop_size, args.max_iters, args.num_seeds) < 1 or args.seed < 0 or args.seed + args.num_seeds > 2**32:
        parser.error("Population, iterations and seed count must be positive, with seeds in [0, 2**32)")
    if budget < args.pop_size:
        parser.error("The evaluation budget must include the initial population")
    if args.sample_cap is not None and args.sample_cap < 1:
        parser.error("The sample cap must be positive")
    trial_path = args.output / "trials.jsonl"
    if not args.tables_only and trial_path.exists() and not args.resume:
        parser.error("Choose a new output directory to keep previous runs intact")
    write_tables(args.output, args.sample_cap)
    if args.tables_only:
        return

    selected_datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    selected_algorithms = list(ALGORITHMS) if args.algorithm == "all" else [args.algorithm]
    hppso_parameters = None
    if args.hppso_selection:
        if "HPPSO" not in selected_algorithms:
            parser.error("--hppso-selection requires HPPSO")
        hppso_parameters = json.loads(args.hppso_selection.read_text(encoding="utf-8"))["parameters"]
    seeds = list(range(args.seed, args.seed + args.num_seeds))
    settings = dict(datasets=selected_datasets, algorithms=selected_algorithms, seeds=seeds,
                    population=args.pop_size, evaluations=budget, bounds=BOUNDS,
                    split_seed=SPLIT_SEED, california_subsample_seed=SUBSAMPLE_SEED, split=[0.6, 0.2, 0.2],
                    preprocessing="median imputation and standardization fitted on training data only",
                    classification_targets="one-hot", architecture="floor(2 * previous_width / 3)",
                    hidden_activation="sigmoid", output_activation=dict(classification="softmax", regression="linear"),
                    training_loss=dict(classification="cross-entropy", regression="standardized-target MSE"),
                    adam=dict(learning_rate=.001, beta1=.9, beta2=.999, epsilon=1e-8, batch="full"),
                    magd=dict(**MAGD_SETTINGS, source="notebooks/magd_algorithm_demo.ipynb",
                              batch="full", stopping_rule="objective evaluation budget only"),
                    model_selection="best training loss at the evaluation budget",
                    budget_unit="one full-training-set loss evaluation, including fused loss-and-gradient calls")
    if args.sample_cap is not None:
        settings.update(sample_cap=args.sample_cap, subsample_seed=SUBSAMPLE_SEED)
    if hppso_parameters is not None:
        settings["hppso_parameters"] = hppso_parameters
    settings_path = args.output / "settings.json"
    split_path = args.output / "split_indices.json"
    saved = {}
    saved_splits = {}
    if args.resume and trial_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != json.loads(json.dumps(settings)):
            raise ValueError("Resume settings differ from the saved benchmark settings")
        saved = read_saved_trials(trial_path, settings)
        if saved:
            saved_splits = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        write_durable(settings_path, json.dumps(settings, indent=2))
    table13 = []
    predictive_rows = []
    splits = {}
    with trial_path.open("a" if args.resume else "x", encoding="utf-8") as raw:
        for name in selected_datasets:
            partitions, target_scaler, splits[name] = load_partitions(name, args.sample_cap)
            if name in saved_splits and saved_splits[name] != splits[name]:
                raise ValueError(f"Resume split differs from the saved split for {name}")
            if name not in saved_splits and any(key[0] == name for key in saved):
                raise ValueError(f"Missing split for existing trials on {name}")
            write_durable(split_path, json.dumps(saved_splits | splits))
            X_train, y_train = partitions["train"]
            medians = []
            for algo in selected_algorithms:
                scores = []
                metrics_by_seed = []
                for seed in seeds:
                    row = saved.get((name, algo, seed))
                    if row is None:
                        started = perf_counter()
                        nn, oracle = train(algo, X_train, y_train, args.pop_size, budget,
                                           seed=seed, classification=target_scaler is None,
                                           hppso_parameters=hppso_parameters)
                        elapsed = perf_counter() - started
                        metrics = {partition: predictive_metrics(nn, *partitions[partition], target_scaler)
                                   for partition in ("validation", "test")}
                        row = dict(dataset=name, algorithm=algo, seed=seed, training_loss=oracle.best_score,
                                   **metrics, dimension=len(nn.get_weights_flat()), seconds=elapsed,
                                   evaluations=oracle.evaluations, gradient_evaluations=oracle.gradient_evaluations,
                                   convergence=oracle.trace)
                        raw.write(json.dumps(row, allow_nan=False) + "\n")
                        raw.flush()
                        os.fsync(raw.fileno())
                        print(f"{name} | {algo} | seed {seed} | {budget} evaluations | "
                              f"training loss {oracle.best_score:.6g}", flush=True)
                    else:
                        metrics = {partition: row[partition] for partition in ("validation", "test")}
                    scores.append(row["training_loss"])
                    metrics_by_seed.append(metrics)
                for partition, metrics in metrics_by_seed[0].items():
                    for metric in metrics:
                        values = np.array([entry[partition][metric] for entry in metrics_by_seed])
                        q1, q3 = np.quantile(values, [.25, .75])
                        predictive_rows.append((name, algo, partition, metric, len(values), median(values), q1, q3))
                medians.append(median(scores))
            table13.append((DATASETS[name][0], *medians))
    write_csv(args.output / "table13_median_training_loss.csv", ["Dataset", *selected_algorithms], table13)
    write_csv(args.output / "held_out_metrics.csv",
              ["Dataset", "Algorithm", "Partition", "Metric", "Seeds", "Median", "Q1", "Q3"], predictive_rows)


if __name__ == "__main__":
    main()
