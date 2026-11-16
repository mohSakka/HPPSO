"""Recover R² and replay classification trials to measure training macro-F1."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import lru_cache
import html
import json
import math
import os
from pathlib import Path
from statistics import median

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from sklearn.metrics import r2_score
from hppso.experiments.train_neural_network import DATASETS, load_partitions, train, write_durable
from hppso.nn.training import predictive_metrics


ROOT = Path(__file__).resolve().parent / "results"
TUNED = ROOT / "nn_hppso_tuned_pop30_ef3000_seed10"
BASE = ROOT / "nn_pop30_ef3000_seed10_cap2000"
ALGORITHMS = ("HPPSO tuned", "PSO", "PSO-m", "PSO-RIW", "GWO", "CSA",
              "Sep-CMA-ES", "SHADE", "GA-MPC", "Adam", "MAGD")


@lru_cache(maxsize=20)
def data_for(dataset):
    settings = json.loads((TUNED / dataset / "settings.json").read_text())
    partitions, scaler, indices = load_partitions(dataset, settings["sample_cap"])
    for source in (TUNED, BASE):
        saved = json.loads((source / dataset / "split_indices.json").read_text())[dataset]
        if indices != saved:
            raise ValueError(f"Reconstructed split differs for {dataset}")
    return settings, partitions, scaler


@lru_cache(maxsize=20)
def reference_for(dataset):
    reference = {}
    for folder in (TUNED, BASE):
        with (folder / dataset / "trials.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                run = json.loads(line)
                label = run["algorithm"]
                if label == "HPPSO":
                    if folder == BASE:
                        continue
                    label = "HPPSO tuned"
                key = label, run["seed"]
                if key in reference or run["evaluations"] != 3000:
                    raise ValueError(f"Invalid saved run for {dataset}, {key}")
                reference[key] = run
    return reference


def measure(dataset, algorithm, seed):
    settings, partitions, scaler = data_for(dataset)
    saved = reference_for(dataset)[algorithm, seed]
    xtrain, ytrain = partitions["train"]
    xtest, ytest = partitions["test"]
    row = dict(dataset=dataset, algorithm=algorithm, seed=seed)
    if scaler is not None:
        if not np.allclose(ytrain.var(axis=0), 1, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Training targets lack unit variance for {dataset}")
        row["train_r2"] = 1 - saved["training_loss"]
        ytest = scaler.inverse_transform(ytest)
        if ytest.shape[1] == 1:
            variance = ytest.var()
            if variance <= 0:
                raise ValueError(f"Constant test target for {dataset}")
            row["test_r2"] = 1 - saved["test"]["mse"] / variance
            return dict(row, provenance="recovered")
    method = "HPPSO" if algorithm == "HPPSO tuned" else algorithm
    model, oracle = train(
        method, xtrain, ytrain, settings["population"], settings["evaluations"],
        seed=seed, classification=scaler is None,
        hppso_parameters=settings["hppso_parameters"] if algorithm == "HPPSO tuned" else None,
    )
    replay = predictive_metrics(model, *partitions["test"], scaler)
    matches = math.isclose(oracle.best_score, saved["training_loss"], rel_tol=1e-8, abs_tol=1e-12)
    matches &= all(math.isclose(replay[name], value, rel_tol=1e-8, abs_tol=1e-12)
                   for name, value in saved["test"].items())
    if not matches:
        raise ValueError(f"Replay differs from saved run (training loss: {oracle.best_score:.10g} "
                         f"versus {saved['training_loss']:.10g})")
    if scaler is None:
        row["train_macro_f1"] = predictive_metrics(model, xtrain, ytrain, None)["macro_f1"]
        row["test_macro_f1"] = saved["test"]["macro_f1"]
    else:
        row["test_r2"] = r2_score(ytest, scaler.inverse_transform(model.forward(xtest)))
    return dict(row, provenance="verified replay")


def summarize(output, records):
    fields = ("dataset", "algorithm", "seed", "train_r2", "test_r2",
              "train_macro_f1", "test_macro_f1", "provenance")
    with (output / "per_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summaries = []
    sections = []
    for task, metrics in (("Regression", ("train_r2", "test_r2")),
                          ("Classification", ("train_macro_f1", "test_macro_f1"))):
        headers = ("Train R²", "Test R²") if task == "Regression" else ("Train macro-F1", "Test macro-F1")
        cells = []
        for dataset, (name, kind, *_) in DATASETS.items():
            if kind != task:
                continue
            for algorithm in ALGORITHMS:
                matching = [record for record in records if
                            record["dataset"] == dataset and record["algorithm"] == algorithm]
                if not matching:
                    continue
                values = {metric: median(row[metric] for row in matching) for metric in metrics}
                summaries.append(dict(dataset=dataset, task=task, algorithm=algorithm,
                                      seeds=len(matching), **values))
                cells.append(f"<tr><th scope='row'>{html.escape(name)}</th><td>{html.escape(algorithm)}</td>"
                             + "".join(f"<td>{v:.6g}</td>" for v in values.values())
                             + f"<td>{len(matching)}</td></tr>")
        sections.append(f"<h2>{task}</h2><table><thead><tr><th>Dataset</th><th>Algorithm</th>"
                        f"<th>{headers[0]}</th><th>{headers[1]}</th><th>Seeds</th></tr></thead>"
                        f"<tbody>{''.join(cells)}</tbody></table>")
    with (output / "medians.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "task", "algorithm", "seeds", *fields[3:7]))
        writer.writeheader()
        writer.writerows(summaries)
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Additional neural network metrics</title>'
            '<style>body{font:16px system-ui;margin:2rem;color:#182638}table{border-collapse:collapse}'
            'th,td{padding:.4rem .6rem;border-bottom:1px solid #dce5ec;text-align:right}'
            'thead th{background:#eaf1f6}th[scope=row]{text-align:left}</style>'
            '<h1>Neural network R² and macro-F1</h1>'
            '<p>Values are medians over the seeds listed in each row. Ten seeds indicate a complete result. '
            'Regression R² uses equal weight per output. Training macro-F1 uses the model selected by '
            'minimum observed training loss. Replays are included only when training loss and test metrics '
            'match the saved run within numerical tolerance.</p>'
            '<p><a href="medians.csv">Medians</a> · <a href="per_seed.csv">Per-seed metrics</a></p>'
            + "".join(sections) + "</html>")
    write_durable(output / "index.html", page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    records = []
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for stage in ("recover", "replay"):
            pending = {}
            for dataset, (_, task, _, _, targets) in DATASETS.items():
                recovery = task == "Regression" and targets == 1
                if recovery != (stage == "recover"):
                    continue
                for algorithm in ALGORITHMS:
                    for seed in range(10):
                        job = dataset, algorithm, seed
                        saved = output / "trials" / dataset / algorithm / f"seed_{seed}.json"
                        if saved.is_file():
                            records.append(json.loads(saved.read_text()))
                        else:
                            pending[pool.submit(measure, *job)] = job, saved
            if records:
                summarize(output, records)
            for future in as_completed(pending):
                job, saved = pending[future]
                try:
                    row = future.result()
                except Exception as error:
                    failures += 1
                    print(f"{job}: {error}", flush=True)
                else:
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    write_durable(saved, json.dumps(row, allow_nan=False))
                    records.append(row)
                completed = len(records) + failures
                if completed % 10 == 0:
                    summarize(output, records)
                    print(f"{completed}/2200 examined, {len(records)} verified, {failures} failed", flush=True)
            summarize(output, records)
            print(f"{stage}: {len(records)} verified rows, {failures} failed", flush=True)


if __name__ == "__main__":
    main()
