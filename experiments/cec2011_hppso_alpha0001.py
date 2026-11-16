"""Compare HPPSO curiosity mutation factors on CEC2011 T21 and T22."""
from experiments import cec2011_checkpointed as base
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import threading
import numpy as np
from hppso.algorithms.hppso import HPPSO


def run_hppso(objective, dimension, lower, upper, population, iterations, seed):
    random.seed(seed)
    np.random.seed(seed)
    return HPPSO(objective, n_pop=population, dimensions=dimension,
                 max_it=iterations, bounds=(lower, upper), socialism_threshold=0.9, alpha_curio=0.001).optimize()


def trial(job, output, fingerprint):
    base.DEFAULT_ALGORITHMS['HPPSO'] = run_hppso
    return base.trial(job, output, fingerprint)


def main():
    source = base.ROOT / 'experiments/results/cec2011_pop200_fe400k_checkpointed'
    reference = source.parent / 'cec2011_hppso_theta09'
    output = source.parent / 'cec2011_hppso_theta09_alpha0001'
    output.mkdir(exist_ok=True)
    baseline = json.loads((source / 'plan.json').read_text())
    config = baseline['config']
    assert config['population'] == 200 and config['evaluation_budget'] == 400000
    digest = hashlib.sha256(baseline['fingerprint'].encode())
    for path in (Path(__file__), Path(base.__file__), base.ROOT / 'src/hppso/algorithms/hppso.py',
                 base.ROOT / 'src/hppso/algorithms/_pso_bounds.py', base.ROOT / 'experiments/optimizer_checkpoint.py'):
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    jobs = [dict(problem=p['problem'], dimension=p['dimension'], algorithm='HPPSO', seed=seed,
                 population=200, budget=400000, schedule_horizon=config['schedule_horizon'], theta=0.9, alpha_mut=0.001)
            for p in config['problems'] if p['problem'] in (21, 22) for seed in config['seeds']]
    plan = dict(fingerprint=fingerprint, baseline_fingerprint=baseline['fingerprint'], jobs=jobs)
    with base.shard_lease(output, 'alpha0001'):
        if (output / 'plan.json').exists():
            assert json.loads((output / 'plan.json').read_text()) == plan
        else:
            base.atomic(output / 'plan.json', plan)
        (output / 'index.html').write_text('''<!doctype html><meta charset="utf-8"><h1>HPPSO T21 and T22, theta 0.9 and alpha_mut 0.001</h1>
<pre id="status"></pre><pre id="workers"></pre><a href="comparison.csv">Paired comparison CSV</a>
<script>async function refresh(){try{for(const [file,id] of [['progress.json','status'],['active.json','workers']]){
document.getElementById(id).textContent=JSON.stringify(await(await fetch(file+'?t='+Date.now())).json(),null,2)
}}catch(e){}}refresh();setInterval(refresh,3000)</script>''', encoding='utf-8')
        server = ThreadingHTTPServer(('127.0.0.1', 8776), partial(SimpleHTTPRequestHandler, directory=str(output)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        rows = []
        pending = []
        for job in jobs:
            path = base.path_for(output, job)
            if path.exists():
                row = base.read_result(path)
                assert row['fingerprint'] == fingerprint
                if row['status'] == 'budget_reached':
                    rows.append(row)
                    continue
            pending.append(job)
        with ProcessPoolExecutor(max_workers=2) as pool:
            active = {pool.submit(trial, job, str(output), fingerprint) for job in pending}
            while active:
                ready, active = base.wait(active, timeout=2, return_when=base.FIRST_COMPLETED)
                for future in ready:
                    row = future.result()
                    rows.append(row)
                    print(row['problem'], row['seed'], row['status'], row['best_score'], flush=True)
                base.atomic(output / 'progress.json', dict(completed=len(rows), total=6,
                    failed=sum(r['status'] != 'budget_reached' for r in rows), stage='running'))
                workers = []
                for path in (output / 'active').glob('worker_*.json'):
                    try:
                        worker = json.loads(path.read_text())
                        if not worker['done']:
                            workers.append(worker)
                    except (OSError, ValueError):
                        pass
                base.atomic(output / 'active.json', workers)
        import csv
        with (output / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['problem', 'seed', 'alpha_0.05', 'alpha_0.001', 'difference_0.001_minus_0.05', 'status'])
            for row in sorted(rows, key=lambda r: (r['problem'], r['seed'])):
                old = base.read_result(base.path_for(reference, row))
                assert old['status'] == 'budget_reached' and old['evaluations'] == 400000
                new = row['best_score'] if row['status'] == 'budget_reached' else None
                writer.writerow([row['problem'], row['seed'], old['best_score'], new,
                                 None if new is None else new - old['best_score'], row['status']])
        base.atomic(output / 'active.json', [])
        base.atomic(output / 'progress.json', dict(completed=len(rows), total=6,
            failed=sum(r['status'] != 'budget_reached' for r in rows), stage='complete'))
    threading.Event().wait()


if __name__ == '__main__':
    main()
