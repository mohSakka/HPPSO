"""Evaluation-budgeted representative dimensional scaling experiments."""
import json
from time import perf_counter

import numpy as np
import pandas as pd

from hppso.benchmarks.additional import NAMES, VERSION, build_additional_suite
from hppso.experiments.runners import DEFAULT_ALGORITHMS


class BudgetExhausted(Exception):
    pass


class EvaluationBudget:
    """Count every call, including initialization and inner-loop comparisons."""
    def __init__(self, function, limit):
        self.function, self.limit = function, int(limit)
        self.count, self.best = 0, float("inf")
        self.trace = []

    def __call__(self, x):
        if self.count >= self.limit:
            raise BudgetExhausted
        value = float(self.function(x))
        self.count += 1
        if not np.isfinite(value):
            raise ValueError("Non-finite objective evaluation")
        self.best = min(self.best, value)
        if self.count == 1 or self.count % max(1, self.limit//100) == 0 or self.count == self.limit:
            self.trace.append([self.count, self.best])
        return value


def summarize(rows):
    df = pd.DataFrame(rows)
    records = []
    keys = ["budget_mode", "dimension", "function", "algorithm"]
    for key, group in df.groupby(keys, sort=False):
        ok = group[group.status != "failed"]
        record = dict(zip(keys, (v.item() if isinstance(v, np.generic) else v for v in key)))
        record.update(n_runs=len(group), n_valid=len(ok), n_failed=len(group)-len(ok))
        for field in ("error_per_dimension", "relative_error", "elapsed_seconds", "evaluations"):
            values = ok[field].to_numpy(float)
            for suffix, quantile in (("q25", .25), ("median", .5), ("q75", .75)):
                record[field+"_"+suffix] = float(np.quantile(values, quantile)) if len(values) else None
        records.append(record)
    return records


def run_scaling(cfg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    dimensions = cfg.get("dimensions", list(range(100, 1001, 100)))
    modes = cfg.get("budget_modes", ["proportional", "fixed"])
    names = cfg.get("functions", list(NAMES))
    algorithms = cfg["algorithms"]
    if set(algorithms) - DEFAULT_ALGORITHMS.keys():
        raise ValueError("Unknown scaling algorithm")
    if not modes or set(modes) - {"proportional", "fixed"}:
        raise ValueError("budget_modes must contain proportional and/or fixed")
    pop, runs, seed = cfg["pop_size"], cfg["runs"], cfg.get("seed", 42)
    if runs < 1 or pop < 5 or not dimensions:
        raise ValueError("Need runs >= 1, pop_size >= 5 and dimensions")
    rows, definitions = [], {}
    for dim in dimensions:
        problems = build_additional_suite(dim, seed, names)
        definitions[str(dim)] = [p.definition() for p in problems]
        (out_dir / "problem_definitions.json").write_text(json.dumps(definitions, indent=2), encoding="utf-8")
        for mode in modes:
            limit = int(cfg.get("evaluations_per_dimension", 10)*dim if mode == "proportional"
                        else cfg.get("fixed_evaluations", 1000))
            if limit < pop:
                raise ValueError("Evaluation budget must cover the initial population")
            for problem in problems:
                # Common reference points are independent of algorithms and run seeds.
                rng = np.random.default_rng(np.random.SeedSequence([seed, dim, NAMES.index(problem.name), 999]))
                lo, hi = problem.bounds
                reference = float(np.median([problem(x) for x in rng.uniform(lo, hi, (100, dim))]))
                for algo in algorithms:
                    for run in range(runs):
                        objective = EvaluationBudget(problem, limit)
                        started = perf_counter()
                        status, error = "early_stop", None
                        try:
                            runner = DEFAULT_ALGORITHMS[algo]
                            args = (objective, dim, lo, hi, pop, int(np.ceil(limit/pop))+1)
                            if algo in ("CMA-ES", "Sep-CMA-ES"):
                                runner(*args, limit, seed+run)
                            else:
                                runner(*args, seed+run)
                        except BudgetExhausted:
                            status = "budget_exhausted"
                        except Exception as exc:
                            status, error = "failed", str(exc)
                        if objective.count == limit and status != "failed":
                            status = "budget_exhausted"
                        best = objective.best if status != "failed" and np.isfinite(objective.best) else None
                        row = dict(budget_mode=mode, dimension=dim, function=problem.name,
                                   algorithm=algo, run=run, seed=seed+run, budget=limit,
                                   evaluations=objective.count, best_score=best,
                                   error_per_dimension=best/dim if best is not None else None,
                                   relative_error=best/reference if best is not None else None,
                                   reference_score=reference, elapsed_seconds=perf_counter()-started,
                                   status=status, error=error, trace=objective.trace)
                        rows.append(row)
                print(f"Scaling {dim}D {mode}: {problem.name} finished", flush=True)
                # Persist completed work after every function.
                pd.DataFrame([{k:v for k,v in r.items() if k != "trace"} for r in rows]).to_csv(out_dir/"scaling_runs.csv", index=False)
                (out_dir/"scaling_checkpoint.json").write_text(json.dumps(rows, allow_nan=False), encoding="utf-8")
    summary = summarize(rows)
    pd.DataFrame(summary).to_csv(out_dir/"scaling_summary.csv", index=False)
    return dict(runs=rows, summary=summary, suite_version=VERSION)
