"""Two-stage, resumable all-function experiment with a localhost progress page."""
import os
# Keep each worker's BLAS from spawning a competing CPU pool.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from time import perf_counter, time
import traceback

import numpy as np
import pandas as pd

from experiments.scaling import EvaluationBudget, BudgetExhausted
from hppso.benchmarks.classical import problem_definitions
from hppso.benchmarks.suite import BENCHMARK_NAMES, SUITE_VERSION, build_benchmark_suite
from hppso.experiments.runners import DEFAULT_ALGORITHMS

DIMENSIONS = (30,)
POPULATIONS = {30: 30, 250: 90, 500: 160, 750: 230, 1000: 300}


def settings(dimension, phase):
    if phase == "smoke":
        return dict(population=30, iterations=2, budget=90)
    pop = POPULATIONS[dimension]
    iterations = 3000 if dimension == 30 else int(np.ceil(1000*dimension/pop))-1
    return dict(population=pop, iterations=iterations, budget=pop*(iterations+1))


def atomic_json(path, data):
    # Readers retry transient partial JSON. Direct writes avoid Windows rename
    # failures while the local progress server has the prior file open.
    Path(path).write_text(json.dumps(data, allow_nan=False), encoding="utf-8")


def algorithms_for(dimension):
    return [name for name in DEFAULT_ALGORITHMS
            if name != ("Sep-CMA-ES" if dimension == 30 else "CMA-ES")]


class LiveBudget(EvaluationBudget):
    def __init__(self, problem, limit, active_path, identity, check_bounds):
        super().__init__(problem["function"], limit)
        self.active_path, self.identity = active_path, identity
        self.lo, self.hi = (np.asarray(b) for b in problem["bounds"])
        self.check_bounds = check_bounds
        self.last_write = 0
        self.started = perf_counter()
        self.publish()

    def publish(self):
        try:
            atomic_json(self.active_path, dict(self.identity, evaluations=self.count,
                        best_score=self.best if np.isfinite(self.best) else None,
                        elapsed_seconds=perf_counter()-self.started, updated=time(), done=False))
        except OSError:
            pass  # A temporarily busy progress file must never fail a trial.
        self.last_write = perf_counter()

    def __call__(self, x):
        if self.check_bounds and (np.any(x < self.lo-1e-9) or np.any(x > self.hi+1e-9)):
            raise ValueError("Objective called outside feasible bounds")
        value = super().__call__(x)
        if perf_counter()-self.last_write >= 3:
            self.publish()
        return value


def execute_trial(job, folder):
    problem, identity = job
    active = Path(folder)/f"worker_{os.getpid()}.json"
    objective = LiveBudget(problem, identity["budget"], active, identity,
                           check_bounds=identity["phase"] == "smoke")
    status, error = "iteration_limit", None
    started = perf_counter()
    try:
        args = (objective, identity["dimension"], *problem["bounds"],
                identity["population"], identity["iterations"])
        if identity["algorithm"] in ("CMA-ES", "Sep-CMA-ES"):
            # Some benchmarks have negative optima: the native 1e-10 target
            # would stop at the first generation even far from their optimum.
            DEFAULT_ALGORITHMS[identity["algorithm"]](
                *args, identity["budget"], identity["seed"], stopfitness=-np.inf)
        else:
            DEFAULT_ALGORITHMS[identity["algorithm"]](*args, identity["seed"])
    except BudgetExhausted:
        status = "budget_exhausted"
    except Exception:
        status, error = "failed", traceback.format_exc()
    if objective.count == identity["budget"] and status != "failed":
        status = "budget_exhausted"
    row = dict(identity, status=status, error=error, evaluations=objective.count,
               best_score=objective.best if np.isfinite(objective.best) else None,
               elapsed_seconds=perf_counter()-started, trace=objective.trace)
    try:
        atomic_json(active, dict(identity, done=True, updated=time()))
    except OSError:
        pass
    return row


def combined_suite(dim):
    np.random.seed(42+dim)
    problems = build_benchmark_suite(dim, michalewicz_dimension=None)
    return problems, problem_definitions(problems)


def job_key(row):
    return (row["phase"], row["dimension"], row["function"], row["algorithm"], row["seed"])


def write_results(folder, completed):
    rows = [{k:v for k,v in r.items() if k != "trace"} for r in completed.values()]
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(folder/"runs.csv", index=False)
    valid = frame[frame.status != "failed"]
    if len(valid):
        valid.groupby(["phase", "dimension", "function", "algorithm"]).agg(
            completed_seeds=("seed", "count"), mean_best=("best_score", "mean"),
            std_best=("best_score", "std"), min_best=("best_score", "min"),
            evaluations=("evaluations", "sum"), seconds=("elapsed_seconds", "sum")
        ).reset_index().to_csv(folder/"summary.csv", index=False)


PAGE = """<!doctype html><meta charset="utf-8"><title>HPPSO live experiments</title>
<style>body{font:16px system-ui;margin:32px;background:#111827;color:#e5e7eb}a{color:#93c5fd}
progress{width:100%;height:24px}td,th{text-align:left;padding:8px;border-bottom:1px solid #374151}
table{width:100%;font-size:14px}.card{padding:16px;background:#1f2937;margin:12px 0;white-space:pre-wrap}</style>
<h1>HPPSO — live experiment progress</h1><p>{function_count} functions · 9 algorithms (CMA-ES at 30D; Sep-CMA-ES at higher dimensions) · 2 seeds · dimensions listed in the budget plan</p>
<p><a href="runs.csv">Completed run results CSV</a> · <a href="summary.csv">Summary CSV</a> ·
<a href="plan.json">Exact budget plan</a> · <a href="problem_definitions.json">Problem definitions</a></p>
<h2 id="phase">Connecting…</h2><progress id="bar" max="2500"></progress><p id="counts"></p>
<p>Smoke scores are diagnostics at 90 evaluations, not final optimization results. Means with only one completed seed are provisional.</p>
<div id="active"></div><h2>Most recently completed trials</h2><div id="recent"></div>
<script>
const esc=x=>String(x??'').replaceAll('&','&amp;').replaceAll('<','&lt;');
const fmt=x=>x==null?'—':Number(x).toExponential(4);
async function refresh(){try{const s=await(await fetch('progress.json?t='+Date.now())).json();
document.getElementById('phase').textContent=s.phase;
document.getElementById('bar').max=s.total;document.getElementById('bar').value=s.completed;
document.getElementById('counts').textContent=`${s.completed} / ${s.total} trials completed in this phase · ${s.failures} failures · updated ${new Date(s.updated*1000).toLocaleTimeString()}`;
document.getElementById('active').innerHTML=s.active.map(a=>`<div class="card">${esc(a.phase)} · ${a.dimension}D · ${esc(a.function)} · ${esc(a.algorithm)} · seed ${a.seed}
${a.evaluations.toLocaleString()} / ${a.budget.toLocaleString()} evaluations · best ${fmt(a.best_score)} · ${Math.round(a.elapsed_seconds)} s</div>`).join('');
document.getElementById('recent').innerHTML='<table><tr><th>Phase</th><th>D</th><th>Function</th><th>Algorithm</th><th>Seed</th><th>Status</th><th>Best score</th></tr>'+s.recent.map(r=>`<tr><td>${esc(r.phase)}</td><td>${r.dimension}</td><td>${esc(r.function)}</td><td>${esc(r.algorithm)}</td><td>${r.seed}</td><td>${esc(r.status)}</td><td>${fmt(r.best_score)}</td></tr>`).join('')+'</table>';
}catch(e){document.getElementById('counts').textContent='Waiting for progress…';}}
refresh();setInterval(refresh,3000);
</script>"""


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dimensions", type=int, nargs="+", default=list(DIMENSIONS))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--phase", choices=("smoke", "full", "both"), default="both")
    args = parser.parse_args(argv)
    dimensions = tuple(dict.fromkeys(args.dimensions))
    if not dimensions or set(dimensions)-POPULATIONS.keys():
        parser.error("Supported dimensions: 30, 250, 500, 750, 1000")
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    plan = dict(dimensions=list(dimensions), seeds=[0, 1], algorithms_by_dimension={d: algorithms_for(d) for d in dimensions},
                full_settings={d:settings(d, "full") for d in dimensions},
                smoke_settings=settings(30, "smoke"), functions_per_dimension=len(BENCHMARK_NAMES),
                functions=list(BENCHMARK_NAMES), suite_version=SUITE_VERSION,
                note="30D uses 3000 iterations and excludes Sep-CMA-ES; higher dimensions require explicit selection; initialization counted")
    plan = json.loads(json.dumps(plan))
    plan_path = folder/"plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing output uses a different plan; select a new output directory")
    atomic_json(plan_path, plan)
    (folder/"process_id.txt").write_text(str(os.getpid()))
    (folder/"index.html").write_text(PAGE.replace("{function_count}", str(len(BENCHMARK_NAMES))), encoding="utf-8")
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(folder), **kw)
        def log_message(self, *a):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    completed = {}
    journal = folder/"trials.jsonl"
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            if row["dimension"] in dimensions:
                completed[job_key(row)] = row
    write_results(folder, completed)
    suites, definitions = {}, {}
    for dim in dimensions:
        suites[dim], definitions[str(dim)] = combined_suite(dim)
    atomic_json(folder/"problem_definitions.json", definitions)
    def publish(phase):
        phase_rows = [r for r in completed.values() if r["phase"] == phase]
        active = []
        for path in folder.glob("worker_*.json"):
            try:
                value = json.loads(path.read_text())
                if value["phase"] == phase and not value.get("done") and time()-value["updated"] < 60:
                    active.append(value)
            except (OSError, ValueError):
                pass
        atomic_json(folder/"progress.json", dict(phase=phase, completed=len(phase_rows),
                    total=sum(len(suites[d])*len(algorithms_for(d))*2 for d in dimensions), failures=sum(r["status"] == "failed" for r in phase_rows),
                    active=active, recent=[{k:v for k,v in r.items() if k not in ("trace", "error")}
                                          for r in list(completed.values())[-20:][::-1]], updated=time()))
    print(f"Live progress: http://127.0.0.1:{args.port}", flush=True)
    phases = ("smoke", "full") if args.phase == "both" else (args.phase,)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for phase in phases:
            futures = set()
            identities = {}
            for dim in dimensions:
                for p in suites[dim]:
                    for algo in algorithms_for(dim):
                        for seed in (0, 1):
                            identity = dict(phase=phase, dimension=dim, function=p["name"], algorithm=algo,
                                            seed=seed, **settings(dim, phase))
                            previous = completed.get(job_key(identity))
                            if previous is None or previous["status"] == "failed":
                                future = pool.submit(execute_trial, (p, identity), str(folder))
                                futures.add(future)
                                identities[future] = identity
            publish(phase)
            last_save = perf_counter()
            while futures:
                ready, futures = wait(futures, timeout=3, return_when=FIRST_COMPLETED)
                for future in ready:
                    try:
                        row = future.result()
                    except Exception:
                        row = dict(identities[future], status="failed", error="Worker infrastructure failure: "+traceback.format_exc(),
                                   evaluations=0, best_score=None, elapsed_seconds=0, trace=[])
                    completed[job_key(row)] = row
                    with journal.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, allow_nan=False)+"\n")
                    print(f"{phase} {row['dimension']}D {row['function']} {row['algorithm']} seed={row['seed']} {row['status']} best={row['best_score']}", flush=True)
                if perf_counter()-last_save >= 10 or not futures:
                    write_results(folder, completed)
                    last_save = perf_counter()
                publish(phase)
            if any(r["status"] == "failed" for r in completed.values() if r["phase"] == phase):
                print(f"{phase} has failures; subsequent phase not launched. See runs.csv.", flush=True)
                break
    print("Requested phases finished. Progress page remains available; Ctrl+C stops server.", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
