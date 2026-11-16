"""Rerun HPPSO with independently sampled shared parameters each epoch."""
from experiments import cec2011_checkpointed as base
from experiments.cec2011_hppso_random_search import OPTIONS
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import sys
import threading
import numpy as np
import pandas as pd
from hppso.algorithms.hppso import HPPSO


class EpochRandomHPPSO(HPPSO):
    def _start_epoch(self):
        self.socialism_threshold = random.random()
        self.alpha_curio = random.choice(OPTIONS['alpha_mut'])
        self.eta = random.choice(OPTIONS['eta'])
        self.lam = random.choice(OPTIONS['lam'])
        self.stagnation_threshold = random.choice(OPTIONS['stagnation_threshold'])


def run_hppso(objective, dimension, lower, upper, population, iterations, seed):
    random.seed(seed)
    np.random.seed(seed)
    return EpochRandomHPPSO(objective, n_pop=population, dimensions=dimension,
                            max_it=iterations, bounds=(lower, upper)).optimize()


def trial(job, output, fingerprint):
    base.DEFAULT_ALGORITHMS['HPPSO'] = run_hppso
    return base.trial(job, output, fingerprint)


def report(output, baseline, rows, title, description):
    valid = pd.DataFrame(rows)
    valid = valid[valid.status.eq('budget_reached') & valid.evaluations.eq(200000)]
    stats = valid.groupby('problem').best_score.agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    stats = stats[stats['count'].eq(3)].reset_index()
    stats['algorithm'] = 'HPPSO'
    others = baseline[baseline.problem.isin(stats.problem) & baseline.algorithm.ne('HPPSO')]
    comparison = pd.concat([others, stats], ignore_index=True)
    comparison['rank'] = comparison.groupby('problem')['mean'].rank()
    comparison = comparison.sort_values(['problem', 'rank'])
    folder = output / 'tables'
    folder.mkdir(exist_ok=True)
    comparison.to_csv(folder / 'performance_and_ranks.csv', index=False)
    comparison.groupby('algorithm')['rank'].mean().sort_values().to_csv(folder / 'overall_ranks.csv')
    (folder / 'index.html').write_text(f'<!doctype html><meta charset="utf-8"><h1>{title} comparison</h1>'
        f'<p>Population 200, 200,000 evaluations and seeds 0–2. {description} '
        'The eight comparator algorithms use the existing equal-budget results. Only complete three-seed blocks are ranked.</p>'
        + comparison.to_html(index=False, float_format=lambda x: f'{x:.6g}'), encoding='utf-8')
    return len(stats)


def main(policy=None):
    if policy is None:
        policy = dict(title='Epoch-random HPPSO', folder='cec2011_hppso_epoch_random_200k', port=8780,
                      sampling={**OPTIONS, 'theta': 'continuous uniform [0, 1)'}, trial=trial,
                      description='HPPSO resamples its five shared parameters every epoch.',
                      source=Path(__file__))
    root = base.ROOT / 'experiments/results'
    source = root / 'cec2011_pop200_fe200k'
    plan = json.loads((source / 'plan.json').read_text())
    config = plan['config']
    assert config['population'] == 200 and config['evaluation_budget'] == 200000 and config['seeds'] == [0, 1, 2]
    baseline = pd.read_csv(source / 'tables/performance_and_ranks.csv')
    assert len(baseline) == 198 and baseline['count'].eq(3).all()
    output = root / policy['folder']
    output.mkdir(exist_ok=True)
    sampling = policy['sampling']
    digest = hashlib.sha256(json.dumps(sampling, sort_keys=True).encode() + plan['fingerprint'].encode())
    for path in (Path(__file__), policy['source'], Path(base.__file__), base.ROOT / 'src/hppso/algorithms/hppso.py',
                 base.ROOT / 'src/hppso/algorithms/_pso_bounds.py', base.ROOT / 'experiments/optimizer_checkpoint.py',
                 *policy.get('dependencies', ())):
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    jobs = [dict(problem=p['problem'], dimension=p['dimension'], algorithm='HPPSO', seed=seed,
                 population=200, budget=200000, schedule_horizon=200000, sampling=policy['title'])
            for p in config['problems']
            if policy.get('problems') is None or p['problem'] in policy['problems']
            for seed in config['seeds']]
    total = len(jobs)
    manifest = dict(fingerprint=fingerprint, baseline_fingerprint=plan['fingerprint'], sampling=sampling, jobs=jobs)
    with base.shard_lease(output, 'epoch-random'):
        if (output / 'plan.json').exists():
            assert json.loads((output / 'plan.json').read_text()) == manifest
        else:
            base.atomic(output / 'plan.json', manifest)
        page = '''<!doctype html><meta charset="utf-8"><h1>HPPSO_VARIANT</h1>
<p><a href="plan.json">Sampling policy</a> · <a href="tables/index.html">Comparisons</a></p><pre id="status"></pre>
<script>async function update(){try{document.getElementById('status').textContent=JSON.stringify(await(await fetch('progress.json?t='+Date.now())).json(),null,2)}catch(e){}}update();setInterval(update,3000)</script>'''
        (output / 'index.html').write_text(page.replace('HPPSO_VARIANT', policy['title']), encoding='utf-8')
        server = ThreadingHTTPServer(('127.0.0.1', policy['port']), partial(SimpleHTTPRequestHandler, directory=str(output)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        rows = []
        pending = []
        for job in jobs:
            path = base.path_for(output, job)
            if path.exists():
                row = base.read_result(path)
                assert row['fingerprint'] == fingerprint
                if row['status'] == 'budget_reached' and row['evaluations'] == 200000:
                    rows.append({k: v for k, v in row.items() if k not in ('trace', 'best_position')})
                    continue
            pending.append(job)
        report_results = policy.get('report', report)
        completed_problems = report_results(output, baseline, rows, policy['title'], policy['description']) if rows else 0
        with ProcessPoolExecutor(max_workers=2) as pool:
            active = {pool.submit(policy['trial'], job, str(output), fingerprint) for job in pending}
            while active:
                ready, active = wait(active, timeout=2, return_when=FIRST_COMPLETED)
                for future in ready:
                    row = future.result()
                    rows.append(row)
                    print(row['problem'], row['seed'], row['status'], row['best_score'], flush=True)
                if ready:
                    completed_problems = report_results(output, baseline, rows, policy['title'], policy['description'])
                workers = []
                for path in (output / 'active').glob('worker_*.json'):
                    try:
                        worker = json.loads(path.read_text())
                        if not worker['done'] and base.time.time() - worker['updated'] < 120:
                            workers.append(worker)
                    except (OSError, ValueError):
                        pass
                base.atomic(output / 'progress.json', dict(stage='running', completed=len(rows), total=total,
                    completed_problems=completed_problems, failed=sum(r['status'] != 'budget_reached' for r in rows), active=workers))
        if completed_problems == 22:
            from experiments.cec2011_wilcoxon import main as wilcoxon
            sys.argv = [sys.argv[0], '--output', str(output)]
            wilcoxon()
        base.atomic(output / 'progress.json', dict(stage='complete', completed=len(rows), total=total,
            completed_problems=completed_problems, failed=sum(r['status'] != 'budget_reached' for r in rows)))
    threading.Event().wait()


if __name__ == '__main__':
    main()
