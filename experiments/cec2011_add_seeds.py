"""Extend the refreshed-HPPSO comparison to seeds 0–9."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
from pathlib import Path
import sys
import threading

from experiments import cec2011_checkpointed as base
from experiments import cec2011_hppso_coordinate_refresh as refresh


RESULTS = base.ROOT / 'experiments/results'
BASELINE = RESULTS / 'cec2011_pop200_fe200k'
REFRESHED = RESULTS / 'cec2011_hppso_coordinate_refresh_200k'
OUTPUT = RESULTS / 'cec2011_coordinate_refresh_10_seeds'
SEEDS = list(range(10))


PAGE = r'''<!doctype html><meta charset="utf-8"><title>CEC2011 seed extension</title>
<h1>CEC2011 seed extension</h1>
<p>Nine algorithms and 22 problems. Population 200 and 200,000 evaluations per run.
Seeds 0–2 come from the original comparison. Seeds 3–9 are new runs.</p>
<p><a href="plan.json">Experiment settings</a> · <a href="tables/final_scores.csv">Final scores</a> ·
<a href="tables/per_problem_wilcoxon/index.html">Per-problem Wilcoxon report</a></p>
<pre id="status">Starting…</pre>
<script>async function update() {
  try {
    const p = await (await fetch('progress.json?t='+Date.now())).json();
    document.getElementById('status').textContent =
      p.stage + ' · ' + p.completed + '/' + p.total + ' new trials complete · ' +
      p.failed + ' failed\n\n' + p.active.map(r =>
        'T' + r.problem + ' · ' + r.algorithm + ' · seed ' + r.seed + ' · ' +
        r.evaluations + '/' + r.budget + ' evaluations').join('\n');
  } catch (_) {}
} update(); setInterval(update,3000);</script>'''


def load_trial(root, job, fingerprint):
    row = base.read_result(base.path_for(root, job))
    fields = ('problem', 'dimension', 'algorithm', 'seed', 'population', 'budget')
    if any(row[key] != job[key] for key in fields) or row['fingerprint'] != fingerprint:
        raise ValueError(f'Mismatched saved trial in {root}: {job}')
    return row


def complete(row):
    return (row['status'] == 'budget_reached' and row['evaluations'] == row['budget']
            and row['best_score'] is not None and base.np.isfinite(row['best_score']))


def run_trial(job, output, fingerprint):
    runner = refresh.trial if job['algorithm'] == 'HPPSO' else base.trial
    return runner(job, output, fingerprint)


def progress(stage, rows, total):
    active = []
    for file in (OUTPUT / 'active').glob('worker_*.json'):
        try:
            row = json.loads(file.read_text())
        except (OSError, ValueError):
            continue
        if not row['done'] and base.time.time() - row['updated'] < 120:
            active.append(row)
    count = sum(complete(row) for row in rows)
    base.atomic(OUTPUT / 'progress.json', dict(stage=stage, completed=count, total=total,
                                             failed=len(rows)-count, active=active))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--port', type=int, default=8796)
    args = parser.parse_args()
    baseline = json.loads((BASELINE / 'plan.json').read_text())
    refreshed = json.loads((REFRESHED / 'plan.json').read_text())
    config = baseline['config']
    if (refreshed['baseline_fingerprint'] != baseline['fingerprint']
            or config['algorithms'] != base.ALGORITHMS or config['seeds'] != [0, 1, 2]
            or config['population'] != 200 or config['evaluation_budget'] != 200000
            or config['minionpy'] != importlib.metadata.version('minionpy')
            or any(refreshed['sampling'][key] != value for key, value in refresh.PARAMETERS.items())):
        raise ValueError('The saved comparison does not match the seed-extension protocol')
    config = dict(config, seeds=SEEDS, schedule_horizon=200000, hppso=refreshed['sampling'])
    provenance = dict(baseline=baseline['fingerprint'], refreshed_hppso=refreshed['fingerprint'])
    digest = hashlib.sha256(json.dumps([config, provenance, sys.version, base.np.__version__],
                                       sort_keys=True).encode())
    sources = [Path(__file__), Path(base.__file__), Path(refresh.__file__), Path(refresh.study.__file__),
               base.ROOT / 'experiments/optimizer_checkpoint.py', base.ROOT / 'experiments/final_benchmark.py',
               base.ROOT / 'src/hppso/experiments/runners.py']
    sources.extend(sorted((base.ROOT / 'src/hppso/algorithms').glob('*.py')))
    for source in sources:
        digest.update(source.read_bytes())
    fingerprint = digest.hexdigest()
    manifest = dict(config=config, provenance=provenance, fingerprint=fingerprint)
    old, jobs = [], []
    for problem in config['problems']:
        number = problem['problem']
        _, lower, upper = base.problem(number)
        if lower.tolist() != problem['lower'] or upper.tolist() != problem['upper']:
            raise ValueError(f'T{number} search bounds differ from the prior experiment')
        for algorithm in config['algorithms']:
            for seed in SEEDS:
                job = dict(problem=number, dimension=problem['dimension'], algorithm=algorithm, seed=seed,
                           population=200, budget=200000, schedule_horizon=200000)
                if seed >= 3:
                    jobs.append(job)
                else:
                    root, key = (REFRESHED, 'refreshed_hppso') if algorithm == 'HPPSO' else (BASELINE, 'baseline')
                    row = load_trial(root, job, provenance[key])
                    if not complete(row):
                        raise ValueError(f'Incomplete existing trial: {job}')
                    old.append(row)
    OUTPUT.mkdir(exist_ok=True)
    with base.shard_lease(OUTPUT, 'extension'):
        path = OUTPUT / 'plan.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('Saved manifest differs from current protocol or source files')
        if not path.exists():
            base.atomic(path, manifest)
        (OUTPUT / 'index.html').write_text(PAGE, encoding='utf-8')
        server = ThreadingHTTPServer(('127.0.0.1', args.port),
                                     partial(SimpleHTTPRequestHandler, directory=str(OUTPUT)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        rows, pending = [], []
        for job in jobs:
            if base.path_for(OUTPUT, job).exists():
                row = load_trial(OUTPUT, job, fingerprint)
                if complete(row):
                    rows.append(row)
                    continue
            pending.append(job)
        progress('running', rows, len(jobs))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            active = {pool.submit(run_trial, job, str(OUTPUT), fingerprint) for job in pending}
            while active:
                finished, active = wait(active, timeout=2, return_when=FIRST_COMPLETED)
                for future in finished:
                    row = future.result()
                    rows.append(row)
                    print(f'[{len(rows)}/{len(jobs)}] T{row["problem"]} {row["algorithm"]} '
                          f'seed {row["seed"]}: {row["status"]}, {row["evaluations"]} evaluations', flush=True)
                    if row['error']:
                        print(row['error'], flush=True)
                progress('running', rows, len(jobs))
        folder = OUTPUT / 'tables'
        folder.mkdir(exist_ok=True)
        fields = ['problem', 'seed', 'algorithm', 'evaluations', 'best_score', 'status', 'fingerprint']
        with (folder / 'final_scores.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(sorted(old + rows, key=lambda r: (r['problem'], r['algorithm'], r['seed'])))
        valid = all(complete(row) for row in rows)
        if valid:
            from experiments.cec2011_problem_wilcoxon import main as wilcoxon
            wilcoxon(['--output', str(OUTPUT), '--significance-only', '--scores', str(folder / 'final_scores.csv')])
        progress('complete' if valid else 'incomplete', rows, len(jobs))
    threading.Event().wait()


if __name__ == '__main__':
    main()
