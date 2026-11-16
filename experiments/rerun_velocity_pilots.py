"""Rerun archived pilot trials on their saved instances with zero-velocity PSO."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

from experiments.final_benchmark import (
    ROOT, BudgetComplete, StrictObjective, read_result, result_path, shard_lease,
    source_hash, suite, write_result,
)

import numpy as np
from scipy.fft import dct
from hppso.benchmarks.additional import AdditionalProblem
from hppso.benchmarks.classical import problem_definitions
from hppso.experiments.runners import DEFAULT_ALGORITHMS


RESULTS = ROOT / "experiments/results"
VARIANTS = {"Original PSO", "PSO-m", "PSO-RIW"}
PILOTS = (
    "all_functions_1000d_budget_v2", "all_functions_30d_1000d",
    "all_functions_30d_1500iters", "all_functions_30d_3000iters", "scaling_pilot",
)


def legacy_problem(definition):
    shift = np.asarray(definition["shift"])
    signs = np.asarray(definition["signs"])
    permutation = np.asarray(definition["permutation"])
    name = definition["name"]
    dimension = definition["dimension"]

    def rosenbrock(z):
        return float(np.sum(100 * (z[1:] - 2 * z[:-1] - z[:-1] ** 2) ** 2 + z[:-1] ** 2))

    def objective(x):
        z = x - shift
        if name == "sphere":
            return float(np.dot(z, z))
        if name == "rosenbrock":
            return rosenbrock(z)
        y = dct(signs * z[permutation], type=2, norm="ortho")
        if name == "rotated_ellipsoid":
            return AdditionalProblem.ellipsoid(y)
        if name == "rotated_rastrigin":
            return AdditionalProblem.rastrigin(y)
        if name == "composition":
            a, b, c = np.array_split(y, 3)
            return dimension * (AdditionalProblem.ellipsoid(a) / (1e6 * len(a))
                                + AdditionalProblem.rastrigin(b) / (10 * len(b))
                                + rosenbrock(c) / (100 * len(c)))
        raise ValueError(f"Unknown archived objective {name}")

    return dict(function=objective, bounds=(np.asarray(definition["lower"]), np.asarray(definition["upper"])),
                objective_scale=definition["objective_scale"])


def catalog():
    jobs, definitions = {}, {}
    for name in PILOTS:
        folder = RESULTS / name
        defs = json.loads((folder / "problem_definitions.json").read_text())
        scaling = name == "scaling_pilot"
        with (folder / ("scaling_runs.csv" if scaling else "runs.csv")).open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["algorithm"] not in VARIANTS or row["status"] != "budget_exhausted":
                    continue
                job = {k: int(row[k]) for k in ("dimension", "seed", "budget")}
                job.update(function=row["function"], algorithm=row["algorithm"],
                           population=30 if scaling else int(row["population"]))
                records = defs[str(job["dimension"])]
                definition = (next(d for d in records if d["name"] == job["function"])
                              if scaling else records[job["function"]])
                if scaling:
                    job["function"] = "scaling_" + job["function"]
                key = (job["dimension"], job["function"])
                if key in definitions and definitions[key] != definition:
                    raise ValueError(f"Conflicting saved definitions for {key}")
                definitions[key] = definition
                trial_key = (*key, job["algorithm"], job["seed"], job["population"], job["budget"])
                if trial_key not in jobs:
                    jobs[trial_key] = dict(job, sources=[])
                jobs[trial_key]["sources"].append(name)
    return list(jobs.values()), definitions


def execute(item):
    job, definition, output, fingerprint = item
    dimension, name = job["dimension"], job["function"]
    if name.startswith("scaling_"):
        problem = legacy_problem(definition)
    else:
        problem = suite(dimension, 42)[name]
        if problem_definitions([problem])[name] != definition:
            raise ValueError(f"Current definition differs for {dimension}D {name}")
    objective = StrictObjective(problem["function"], job["budget"], points=300)
    try:
        DEFAULT_ALGORITHMS[job["algorithm"]](objective, dimension, *problem["bounds"],
                                             job["population"], job["budget"] + 1, job["seed"])
    except BudgetComplete:
        if objective.count != job["budget"]:
            raise RuntimeError(f"Unexpected budget stop for {job}")
    else:
        raise RuntimeError(f"Run returned before its budget for {job}")
    result = dict(job, fingerprint=fingerprint, status="budget_reached", evaluations=objective.count,
                  best_score=objective.best, best_position=objective.best_position.tolist(),
                  objective_scale=problem["objective_scale"], trace=objective.trace)
    write_result(result_path(output, job), result)
    return result


def run(output, workers):
    jobs, definitions = catalog()
    manifest = dict(jobs=jobs, definitions={f"{d}D/{n}": v for (d, n), v in definitions.items()},
                    velocity_initialization="zeros")
    fingerprint = hashlib.sha256(
        source_hash(manifest).encode() + Path(__file__).read_bytes()
    ).hexdigest()
    manifest["fingerprint"] = fingerprint
    output.mkdir(parents=True, exist_ok=True)
    with shard_lease(output, "pilots"):
        pending = []
        rows = []
        plan = output / "plan.json"
        if plan.exists() and json.loads(plan.read_text()) != manifest:
            raise ValueError("Pilot plan changed. Choose a fresh output directory.")
        if not plan.exists():
            plan.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for job in jobs:
            block = output / f"pop{job['population']}_fe{job['budget']}"
            path = result_path(block, job)
            if path.exists():
                saved = read_result(path)
                if saved["fingerprint"] != fingerprint or saved["status"] != "budget_reached" or saved["evaluations"] != job["budget"]:
                    raise ValueError(f"Invalid saved trial at {path}")
                rows.append(saved)
                continue
            pending.append((job, definitions[job["dimension"], job["function"]], block, fingerprint))
        print(f"{len(rows)}/{len(jobs)} saved, {len(pending)} pending", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(execute, pending):
                rows.append(result)
                print(f"[{len(rows)}/{len(jobs)}] {result['dimension']}D {result['function']} "
                      f"{result['algorithm']} seed {result['seed']}", flush=True)
        omit = {"trace", "best_position"}
        fields = [key for key in rows[0] if key not in omit]
        with (output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.workers)
