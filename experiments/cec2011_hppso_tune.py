"""One-parameter-at-a-time HPPSO screening on T21 and T22."""
from experiments import cec2011_checkpointed as base
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import threading
import numpy as np
import pandas as pd
from hppso.algorithms.hppso import HPPSO


REFERENCE = dict(theta=0.9, alpha_mut=0.05, eta=0.001, lam=0.4)
SWEEP = {'theta': (0.7, 0.95), 'alpha_mut': (0.01, 0.1),
         'eta': (0.0001, 0.01), 'lam': (0.2, 0.7)}


def trial(job, output, fingerprint):
    def runner(objective, dimension, lower, upper, population, iterations, seed):
        random.seed(seed)
        np.random.seed(seed)
        return HPPSO(objective, n_pop=population, dimensions=dimension, max_it=iterations,
                     bounds=(lower, upper), socialism_threshold=job['theta'],
                     alpha_curio=job['alpha_mut'], eta=job['eta'], lam=job['lam']).optimize()

    base.DEFAULT_ALGORITHMS['HPPSO'] = runner
    return base.trial(job, output, fingerprint)


def report(output, rows):
    data = pd.DataFrame(rows)
    successful = data[data.status.eq('budget_reached')]
    summary = successful.groupby(['problem', 'setting']).best_score.agg(['count', 'mean', 'median', 'std', 'min'])
    complete = summary[summary['count'].eq(3)].copy()
    complete['provisional_rank'] = complete.groupby(level='problem')['mean'].rank(method='average')
    complete = complete.sort_values(['problem', 'provisional_rank'])
    data.to_csv(output / 'runs.csv', index=False)
    complete.to_csv(output / 'summary.csv')
    (output / 'summary.html').write_text('<!doctype html><meta charset="utf-8"><h1>Tuning results</h1>'
        '<p>Lower mean is better. Ranks include settings with all three seeds finished. '
        'Each setting changes one parameter relative to theta 0.9, alpha 0.05, eta 0.001, lambda 0.4. '
        'These are screening results on the tuning seeds.</p>' + complete.to_html(float_format=lambda x: f'{x:.6g}'),
        encoding='utf-8')


def main():
    results = base.ROOT / 'experiments/results'
    source = results / 'cec2011_pop200_fe400k_checkpointed'
    baseline = json.loads((source / 'plan.json').read_text())
    config = baseline['config']
    assert config['population'] == 200 and config['evaluation_budget'] == 400000
    output = results / 'cec2011_hppso_tuning_t21_t22'
    output.mkdir(exist_ok=True)
    settings = {f'{name}_{value:g}': dict(REFERENCE, **{name: value})
                for name, values in SWEEP.items() for value in values}
    digest = hashlib.sha256(baseline['fingerprint'].encode())
    for path in (Path(__file__), Path(base.__file__), base.ROOT / 'src/hppso/algorithms/hppso.py',
                 base.ROOT / 'src/hppso/algorithms/_pso_bounds.py', base.ROOT / 'experiments/optimizer_checkpoint.py'):
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    jobs = [dict(problem=p['problem'], dimension=p['dimension'], algorithm='HPPSO', seed=seed,
                 population=200, budget=400000, schedule_horizon=config['schedule_horizon'],
                 setting=label, **params)
            for label, params in settings.items() for p in config['problems']
            if p['problem'] in (21, 22) for seed in config['seeds']]
    references = {'reference': (results / 'cec2011_hppso_theta09', REFERENCE),
                  'theta_0.5': (source, dict(REFERENCE, theta=0.5)),
                  'alpha_mut_0.001': (results / 'cec2011_hppso_theta09_alpha0001', dict(REFERENCE, alpha_mut=0.001))}
    rows = []
    for label, (folder, params) in references.items():
        for problem in (21, 22):
            for seed in config['seeds']:
                row = base.read_result(base.path_for(folder, dict(problem=problem, algorithm='HPPSO', seed=seed)))
                assert row['status'] == 'budget_reached' and row['evaluations'] == 400000 and row['population'] == 200
                rows.append(dict({k: v for k, v in row.items() if k not in ('trace', 'best_position')},
                                 setting=label, source=str(folder), **params))
    plan = dict(fingerprint=fingerprint, reference=REFERENCE, jobs=jobs,
                reused={label: str(folder) for label, (folder, _) in references.items()})
    with base.shard_lease(output, 'hppso-tuning'):
        if (output / 'plan.json').exists():
            assert json.loads((output / 'plan.json').read_text()) == plan
        else:
            base.atomic(output / 'plan.json', plan)
        (output / 'index.html').write_text('''<!doctype html><meta charset="utf-8"><h1>HPPSO tuning on T21 and T22</h1>
<p>Population 200, 400,000 evaluations per trial, seeds 0–2.</p>
<p><a href="summary.html">Current results and ranks</a> · <a href="runs.csv">Trial CSV</a></p><pre id="status"></pre>
<script>async function update(){try{document.getElementById('status').textContent=JSON.stringify(await(await fetch('progress.json?t='+Date.now())).json(),null,2)}catch(e){}}update();setInterval(update,3000)</script>''', encoding='utf-8')
        server = ThreadingHTTPServer(('127.0.0.1', 8777), partial(SimpleHTTPRequestHandler, directory=str(output)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        pending = []
        for job in jobs:
            folder = output / job['setting']
            path = base.path_for(folder, job)
            if path.exists():
                row = base.read_result(path)
                assert row['fingerprint'] == fingerprint
                if row['status'] == 'budget_reached':
                    rows.append({k: v for k, v in row.items() if k not in ('trace', 'best_position')})
                    continue
            pending.append(job)
        report(output, rows)
        completed = len(jobs) - len(pending)
        with ProcessPoolExecutor(max_workers=2) as pool:
            active = {pool.submit(trial, job, str(output / job['setting']), fingerprint): job for job in pending}
            while active:
                ready, _ = wait(active, timeout=2, return_when=FIRST_COMPLETED)
                for future in ready:
                    active.pop(future)
                    row = future.result()
                    rows.append(row)
                    completed += 1
                    print(row['setting'], row['problem'], row['seed'], row['status'], row['best_score'], flush=True)
                if ready:
                    report(output, rows)
                workers = []
                for path in output.glob('*/active/worker_*.json'):
                    try:
                        worker = json.loads(path.read_text())
                        if not worker['done'] and base.time.time() - worker['updated'] < 120:
                            workers.append(worker)
                    except (OSError, ValueError):
                        pass
                base.atomic(output / 'progress.json', dict(completed=completed, total=len(jobs), reused=18,
                    failed=sum(r['status'] != 'budget_reached' for r in rows), active=workers, stage='running'))
        base.atomic(output / 'progress.json', dict(completed=completed, total=len(jobs), reused=18,
                    failed=sum(r['status'] != 'budget_reached' for r in rows), stage='complete'))
    threading.Event().wait()


if __name__ == '__main__':
    main()
