"""Tune a shared HPPSO configuration on four neural-network validation sets."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from functools import lru_cache
from statistics import median

import numpy as np
from scipy.stats import rankdata

from hppso.experiments.train_neural_network import load_partitions, train, write_durable
from hppso.nn.training import predictive_metrics


OUTPUT = Path(__file__).resolve().parent / "results/nn_hppso_tuning_pop30_ef3000"
DATASETS = ("california_housing", "diabetes", "digits", "breast_cancer")
SEEDS = (0, 1)
PARAMETERS = ("socialism_threshold", "alpha_curio", "eta", "lam", "stagnation_threshold")
VALUES = (
    (.5, .05, .001, .4, 10),
    (.2, .03, .1, .4, 5),
    (.5, .05, .01, .4, 10),
    (.5, .05, .03, .4, 10),
    (.3, .02, .01, .2, 5),
    (.7, .10, .03, .6, 20),
    (.8, .03, .003, .2, 10),
    (.4, .08, .03, .6, 5),
)
CONFIGS = dict(zip(("default", "cec2011", "eta_001", "eta_003", "candidate_4", "candidate_5",
                    "candidate_6", "candidate_7"), (dict(zip(PARAMETERS, row)) for row in VALUES)))


@lru_cache(maxsize=len(DATASETS))
def validation_data(dataset):
    partitions, scaler, _ = load_partitions(dataset, 2000)
    return partitions["train"], partitions["validation"], scaler


def evaluate(dataset, configuration, seed):
    training, validation, scaler = validation_data(dataset)
    model, oracle = train("HPPSO", *training, 30, 3000, seed=seed, classification=scaler is None,
                          hppso_parameters=CONFIGS[configuration])
    metrics = predictive_metrics(model, *validation, scaler)
    return dict(dataset=dataset, configuration=configuration, seed=seed, evaluations=oracle.evaluations,
                training_loss=oracle.best_score, validation=metrics)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = dict(configurations=CONFIGS, datasets=list(DATASETS), seeds=list(SEEDS), population=30,
                budget=3000, sample_cap=2000, selection="mean dataset rank of median validation RMSE or macro-F1")
    path = OUTPUT / "plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("The existing tuning plan differs")
    write_durable(path, json.dumps(plan, indent=2))
    keys = [(dataset, cfg, seed) for dataset in DATASETS for cfg in CONFIGS for seed in SEEDS]
    rows = {}
    journal = OUTPUT / "trials.jsonl"
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["dataset"], row["configuration"], row["seed"])
            if key not in keys or row["evaluations"] != 3000:
                raise ValueError(f"Mismatched trial {key}")
            rows[key] = row
    with journal.open("a", encoding="utf-8") as stream, ProcessPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(evaluate, *key): key for key in keys if key not in rows}
        for future in as_completed(pending):
            row = future.result()
            rows[pending[future]] = row
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(f"{len(rows)}/{len(keys)} {row['dataset']} {row['configuration']} seed {row['seed']}", flush=True)
    ranks = {}
    metrics = {}
    for dataset in DATASETS:
        classification = "macro_f1" in rows[dataset, "default", 0]["validation"]
        metric = "macro_f1" if classification else "rmse"
        scores = [median(rows[dataset, cfg, seed]["validation"][metric] for seed in SEEDS) for cfg in CONFIGS]
        metrics[dataset] = dict(metric=metric, medians=dict(zip(CONFIGS, scores)))
        ranks[dataset] = rankdata(-np.array(scores) if classification else scores)
    mean_ranks = dict(zip(CONFIGS, np.mean(list(ranks.values()), axis=0).tolist()))
    winner = min(mean_ranks, key=mean_ranks.get)
    report = dict(winner=winner, parameters=CONFIGS[winner], mean_validation_ranks=mean_ranks, datasets=metrics)
    write_durable(OUTPUT / "selection.json", json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
