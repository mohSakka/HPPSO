"""Run and report the isolated openness experiment; portable and resumable."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np

from experiments.final_benchmark import (
    StrictObjective, BudgetComplete, suite, read_result, write_result, shard_lease,
)
from hppso.algorithms.openness_ablation import OpennessPSO, VARIANTS
from hppso.benchmarks.classical import problem_definitions
from hppso.benchmarks.additional import NAMES as HARD_FUNCTIONS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/openness_ablation.json"
DEFAULT_OUTPUT = ROOT / "experiments/results/openness_ablation_100d"


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    for key in ("dimension", "population", "budget", "trace_points", "bootstrap_resamples"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"Invalid {key}")
    if cfg["population"] < 2 or cfg["budget"] <= cfg["population"]:
        raise ValueError("Budget must exceed a population of at least two")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Seeds must be unique and nonempty")
    if any(not isinstance(s, int) or s < 0 or s >= 2**32 for s in cfg["seeds"]):
        raise ValueError("Seeds must be unsigned 32-bit integers")
    allowed = {"f1_sphere", "f4_rosenbrock", "f10_levy", "f7_ackley", *HARD_FUNCTIONS}
    if not cfg["functions"] or len(set(cfg["functions"])) != len(cfg["functions"]) or not set(cfg["functions"]) <= allowed:
        raise ValueError("Choose unique functions from the four classical and ten additional ablation functions")
    if not 0 <= cfg["threshold"] <= 1 or cfg["lam"] < 0:
        raise ValueError("Invalid openness settings")
    if not all(np.isfinite(cfg[k]) for k in ("w", "c1", "c2", "lam", "threshold")):
        raise ValueError("Parameters must be finite")
    return cfg


def fingerprint(cfg):
    files = [Path(__file__), ROOT / "experiments/final_benchmark.py",
             ROOT / "src/hppso/algorithms/openness_ablation.py",
             ROOT / "src/hppso/algorithms/_pso_bounds.py"]
    files += sorted((ROOT / "src/hppso/benchmarks").glob("*.py"))
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    for file in files:
        digest.update(file.name.encode())
        digest.update(file.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def jobs(cfg):
    for function in cfg["functions"]:
        for seed in cfg["seeds"]:
            for variant in VARIANTS:
                yield dict(function=function, algorithm=variant, seed=seed)


def result_path(output, job):
    return Path(output) / "trials" / job["function"] / job["algorithm"] / f"seed_{job['seed']:03d}.json.gz"


def trial(job, cfg, output, digest):
    problem = suite(cfg["dimension"], cfg["instance_seed"])[job["function"]]
    objective = StrictObjective(problem["function"], cfg["budget"], cfg["trace_points"])
    start = time.perf_counter()
    error, status = None, "failed"
    try:
        optimizer = OpennessPSO(objective, problem["bounds"], cfg["population"],
            job["algorithm"], job["seed"], **{k: cfg[k] for k in ("w", "c1", "c2", "lam", "threshold")})
        optimizer.optimize()
    except BudgetComplete:
        status = "budget_reached"
    except Exception:
        error = traceback.format_exc()
    elapsed = time.perf_counter() - start
    result = dict(job, dimension=cfg["dimension"], population=cfg["population"],
        budget=cfg["budget"], evaluations=objective.count, status=status, error=error,
        best_score=objective.best if np.isfinite(objective.best) else None,
        best_position=None if objective.best_position is None else objective.best_position.tolist(),
        trace=objective.trace, elapsed_seconds=elapsed, fingerprint=digest,
        host=platform.node(), finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_result(result_path(output, job), result)
    return {k: v for k, v in result.items() if k not in ("trace", "best_position")}


def run(cfg, output, workers):
    output.mkdir(parents=True, exist_ok=True)
    digest = fingerprint(cfg)
    plan = dict(config=cfg, variants=VARIANTS, fingerprint=digest,
                numpy_version=np.__version__, python_version=platform.python_version())
    with shard_lease(output, "openness"):
        planfile = output / "plan.json"
        if planfile.exists():
            if json.loads(planfile.read_text())["fingerprint"] != digest:
                raise ValueError("Code/configuration changed; choose a new output directory")
        else:
            planfile.write_text(json.dumps(plan, indent=2))
            problems = suite(cfg["dimension"], cfg["instance_seed"])
            definitions = problem_definitions([problems[n] for n in cfg["functions"]])
            # All supported additions have their global minimum at the shift.
            # Rosenbrock and Levy instead have base minimizers equal to one.
            for name, definition in definitions.items():
                offset = 1 if name in ("f4_rosenbrock", "f10_levy") else 0
                definition["known_minimizer"] = (np.asarray(definition["shift_vector"]) + offset).tolist()
                definition["known_minimum"] = 0.0
            (output / "problem_definitions.json").write_text(json.dumps(definitions, indent=2))
        alljobs = list(jobs(cfg))
        pending = []
        for job in alljobs:
            path = result_path(output, job)
            if path.exists():
                if read_result(path)["fingerprint"] != digest:
                    raise ValueError(f"Mixed results: {path}")
            else:
                pending.append(job)
        done = len(alljobs) - len(pending)
        print(f"{len(alljobs)} trials; {done} saved; {len(pending)} pending; {workers} workers", flush=True)
        def progress():
            temporary = output / "progress.tmp"
            temporary.write_text(json.dumps(dict(completed=done, total=len(alljobs), workers=workers)))
            os.replace(temporary, output / "progress.json")
        progress()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(trial, j, cfg, str(output), digest) for j in pending]
            for future in as_completed(futures):
                result = future.result()
                done += 1
                print(f"[{done}/{len(alljobs)}] {result['function']} {result['algorithm']} "
                      f"seed={result['seed']} {result['status']} FE={result['evaluations']} "
                      f"best={result['best_score']} time={result['elapsed_seconds']:.2f}s", flush=True)
                progress()


def bootstrap_median(values, draws):
    """Seed bootstrap; values shape (seeds, checkpoints), shared draws across arms."""
    # Batching bounds temporary memory independently of trace length.
    estimates = np.concatenate([np.median(values[index], axis=1)
                               for index in np.array_split(draws, max(1, len(draws) // 100))])
    return np.quantile(estimates, [0.025, 0.975], axis=0)


def report(output):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from experiments.openness_plot_pages import make_pages, save_pages
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    cfg = plan["config"]
    records = []
    for job in jobs(cfg):
        path = result_path(output, job)
        if path.exists():
            record = read_result(path)
            if record["fingerprint"] != plan["fingerprint"]:
                raise ValueError("Mixed trial fingerprints")
            records.append(record)
    if not records:
        raise ValueError("No saved results")
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("trace", "best_position")}
                  for r in records]).to_csv(output / "runs.csv", index=False)
    lookup = {(r["function"], r["algorithm"], r["seed"]): r for r in records}
    plots = output / "convergence"
    plots.mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    draws = rng.integers(0, len(cfg["seeds"]), (cfg["bootstrap_resamples"], len(cfg["seeds"])))
    summaries, comparisons, curves, ranks = [], [], [], []
    overviews, axes = make_pages(len(cfg["functions"]))
    labels = {"PSO": "PSO", "PSO-open-worse-pbest": "Open: worse → pbest (original)",
              "PSO-open-better-pbest": "Open: better → pbest",
              "PSO-open-any-pbest": "Open: any → pbest",
              "PSO-open-worse-current": "Open: worse → current"}
    complete_functions = []
    with PdfPages(plots / "all_functions.pdf") as pdf:
        for function, overview_ax in zip(cfg["functions"], axes):
            required = [(function, a, s) for a in VARIANTS for s in cfg["seeds"]]
            if any(key not in lookup or lookup[key]["status"] != "budget_reached" or
                   lookup[key]["evaluations"] != cfg["budget"] for key in required):
                overview_ax.set_title(f"{function}: incomplete/invalid")
                continue
            complete_functions.append(function)
            fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
            final_values = {}
            reference = lookup[(function, "PSO", cfg["seeds"][0])]
            grid = np.asarray(reference["trace"])[:, 0].astype(int)
            for index, algorithm in enumerate(VARIANTS):
                rows = [lookup[(function, algorithm, seed)] for seed in cfg["seeds"]]
                for row in rows:
                    if not np.array_equal(np.asarray(row["trace"])[:, 0], grid):
                        raise ValueError("Inconsistent evaluation checkpoints")
                values = np.array([np.asarray(row["trace"])[:, 1] for row in rows])
                median = np.median(values, axis=0)
                lo, hi = bootstrap_median(values, draws)
                q1, q3 = np.quantile(values, [0.25, 0.75], axis=0)
                final = values[:, -1]
                final_values[algorithm] = final
                summaries.append(dict(function=function, algorithm=algorithm, n=len(rows),
                    mean=final.mean(), std=final.std(ddof=1) if len(final)>1 else 0,
                    median=median[-1], q25=q1[-1], q75=q3[-1], median_ci95_low=lo[-1],
                    median_ci95_high=hi[-1], mean_seconds=np.mean([r["elapsed_seconds"] for r in rows])))
                curves.extend(dict(function=function, algorithm=algorithm, evaluations=int(x), median=m,
                    median_ci95_low=l, median_ci95_high=h, q25=q, q75=t)
                    for x, m, l, h, q, t in zip(grid, median, lo, hi, q1, q3))
                for axis in (ax, overview_ax):
                    axis.plot(grid, median, color=f"C{index}", label=labels[algorithm], linewidth=1.6)
                    axis.fill_between(grid, lo, hi, color=f"C{index}", alpha=0.15)
            for reference_name in ("PSO", "PSO-open-worse-pbest"):
                for algorithm, final in final_values.items():
                    if algorithm == reference_name:
                        continue
                    differences = final - final_values[reference_name]
                    low, high = bootstrap_median(differences[:, None], draws)[:, 0]
                    comparisons.append(dict(function=function, algorithm=algorithm, reference=reference_name,
                        median_paired_difference=np.median(differences), ci95_low=low, ci95_high=high,
                        wins=int(np.sum(differences < 0)), ties=int(np.sum(differences == 0)),
                        losses=int(np.sum(differences > 0)), n=len(differences)))
            function_ranks = pd.Series({a: np.median(v) for a, v in final_values.items()}).rank()
            ranks.extend(dict(function=function, algorithm=a, rank=r) for a, r in function_ranks.items())
            for axis in (ax, overview_ax):
                axis.set_title(function.split("_", 1)[1].replace("_", " ").title() + f" — {cfg['dimension']}D")
                axis.set_xlabel("Objective evaluations (including initialization)")
                axis.set_ylabel("Best-so-far objective error, f(x) − f* (f*=0)")
                # Symlog retains exact zeros and tiny negative Ackley roundoff.
                axis.set_yscale("symlog", linthresh=1e-12)
                axis.set_xlim(0, cfg["budget"])
                axis.grid(alpha=0.2)
                axis.legend(fontsize=8)
            fig.suptitle("Median of matched seeds; shading: pointwise 95% bootstrap CI for median", fontsize=10)
            fig.savefig(plots / f"{function}.png", dpi=170)
            pdf.savefig(fig)
            plt.close(fig)
    title = f"PSO openness ablation — {cfg['dimension']}D, {len(cfg['seeds'])} seeds, {cfg['budget']:,} FE\nMedian; pointwise 95% bootstrap confidence intervals"
    with PdfPages(plots / "overview_pages.pdf") as overview_pdf:
        overview_files = save_pages(overviews, plots, "overview", title, overview_pdf)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "paired_comparisons.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "convergence_summary.csv", index=False)
    pd.DataFrame(ranks).to_csv(output / "function_ranks.csv", index=False)
    if ranks:
        pd.DataFrame(ranks).groupby("algorithm")["rank"].mean().sort_values().to_csv(output / "average_ranks.csv")
    (output / "report_status.json").write_text(json.dumps(dict(saved_trials=len(records),
        expected_trials=len(list(jobs(cfg))), complete_functions=complete_functions,
        failed_trials=sum(r["status"] != "budget_reached" for r in records)), indent=2))
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Openness ablation</title>'
        '<h1>PSO openness ablation</h1><p>Median and pointwise 95% bootstrap confidence intervals.</p>'
        + ''.join(f'<img src="convergence/{name}" style="max-width:100%">' for name in overview_files) + '<p>'
        '<a href="convergence/all_functions.pdf">All curves PDF</a> · <a href="summary.csv">Summary</a> · '
        '<a href="runs.csv">Per-seed results</a> · <a href="paired_comparisons.csv">Paired comparisons</a></p>')
    print(f"Report: {len(complete_functions)}/{len(cfg['functions'])} complete function blocks; {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", action="store_true", help="Report existing results without running")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if not args.report:
        run(load_config(args.config), args.output, args.workers)
    report(args.output)


if __name__ == "__main__":
    main()
