"""Full factorial HPPSO tuning with separate comparison seeds."""
from experiments import cec2011_checkpointed as base
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from itertools import product
import json
from pathlib import Path
import random
import sys
import threading
import numpy as np
import pandas as pd
from hppso.algorithms.hppso import HPPSO


GRID = dict(theta=[0.7, 0.9], alpha_mut=[0.05, 0.1], eta=[0.001, 0.01],
            lam=[0.4, 0.7], stagnation_threshold=[10, 30])
TUNING_SEEDS = [3, 4, 5]
COMPARISON_SEEDS = [0, 1, 2]
BUDGET = 200000


def trial(job, output, fingerprint):
    def runner(objective, dimension, lower, upper, population, iterations, seed):
        random.seed(seed)
        np.random.seed(seed)
        return HPPSO(objective, n_pop=population, dimensions=dimension, max_it=iterations,
                     bounds=(lower, upper), socialism_threshold=job['theta'],
                     alpha_curio=job['alpha_mut'], eta=job['eta'], lam=job['lam'],
                     stagnation_threshold=job['stagnation_threshold']).optimize()

    base.DEFAULT_ALGORITHMS['HPPSO'] = runner
    return base.trial(job, output, fingerprint)


def batch(pool, output, jobs, fingerprint, status):
    rows = []
    pending = []
    for job in jobs:
        folder = output / job['stage'] / job['setting']
        path = base.path_for(folder, job)
        if path.exists():
            row = base.read_result(path)
            assert row['fingerprint'] == fingerprint and all(row[k] == v for k, v in job.items())
            if row['status'] == 'budget_reached' and row['evaluations'] == BUDGET:
                rows.append({k: v for k, v in row.items() if k not in ('trace', 'best_position')})
                continue
        pending.append((job, folder))
    iterator = iter(pending)
    active = set()

    def fill():
        while len(active) < 2:
            item = next(iterator, None)
            if item is None:
                break
            job, folder = item
            active.add(pool.submit(trial, job, str(folder), fingerprint))

    fill()
    while active:
        ready, active = wait(active, timeout=3, return_when=FIRST_COMPLETED)
        for future in ready:
            row = future.result()
            rows.append(row)
            print(row['stage'], row['problem'], row['setting'], row['seed'], row['status'], row['best_score'], flush=True)
        base.atomic(output / 'progress.json', dict(status, batch_completed=len(rows), batch_total=len(jobs),
            batch_failures=sum(row['status'] != 'budget_reached' for row in rows)))
        fill()
    return pd.DataFrame(rows)


def summarize(rows):
    valid = rows[rows.status.eq('budget_reached') & rows.evaluations.eq(BUDGET)]
    scores = valid.groupby('setting').best_score.agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    return scores[scores['count'].eq(3)].sort_values(['mean', 'median'], kind='stable')


def report(output, stats, selected):
    stats = stats[stats.problem.isin(selected)].copy()
    stats['rank'] = stats.groupby('problem')['mean'].rank(method='average')
    stats = stats.sort_values(['problem', 'rank'])
    folder = output / 'tables'
    folder.mkdir(exist_ok=True)
    stats.to_csv(folder / 'performance_and_ranks.csv', index=False)
    stats.pivot(index='problem', columns='algorithm', values='rank').to_csv(folder / 'ranks_by_problem.csv')
    overall = stats.groupby('algorithm')['rank'].mean().sort_values().to_frame('average_rank')
    overall.to_csv(folder / 'overall_ranks.csv')
    choices = pd.DataFrame(list(selected.values()))
    choices.to_csv(folder / 'selected_settings.csv', index=False)
    note = '<p>Population 200 and 200,000 evaluations. Settings selected using seeds 3–5. Comparison uses seeds 0–2. Other algorithms come from the completed equal-budget experiment.</p>'
    (folder / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Grid-tuned HPPSO comparison</h1>' + note
        + '<h2>Selected settings</h2>' + choices.to_html(index=False)
        + '<h2>Ranks across finished problems</h2>' + overall.to_html()
        + '<h2>Per-problem results</h2>' + stats.to_html(index=False), encoding='utf-8')


def main():
    results = base.ROOT / 'experiments/results'
    source = results / 'cec2011_pop200_fe200k'
    baseline = json.loads((source / 'plan.json').read_text())
    config = baseline['config']
    audit = json.loads((source / 'tables/audit.json').read_text())
    assert audit['successful'] == 594 and config['population'] == 200 and config['evaluation_budget'] == BUDGET
    assert config['seeds'] == COMPARISON_SEEDS
    stats = pd.read_csv(source / 'tables/performance_and_ranks.csv')
    assert len(stats) == 198 and stats['count'].eq(3).all()
    settings = {f'grid_{i:02d}': dict(zip(GRID, values))
                for i, values in enumerate(product(*GRID.values()), 1)}
    digest = hashlib.sha256(baseline['fingerprint'].encode())
    for path in (Path(__file__), Path(base.__file__), base.ROOT / 'src/hppso/algorithms/hppso.py',
                 base.ROOT / 'src/hppso/algorithms/_pso_bounds.py', base.ROOT / 'experiments/optimizer_checkpoint.py'):
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    output = results / 'cec2011_hppso_grid_200k'
    output.mkdir(exist_ok=True)
    plan = dict(fingerprint=fingerprint, baseline_fingerprint=baseline['fingerprint'], grid=GRID,
                settings=settings, tuning_seeds=TUNING_SEEDS, comparison_seeds=COMPARISON_SEEDS,
                population=200, budget=BUDGET, tuning_trials=2112, comparison_trials=66)
    with base.shard_lease(output, 'hppso-grid'):
        if (output / 'plan.json').exists():
            assert json.loads((output / 'plan.json').read_text()) == plan
        else:
            base.atomic(output / 'plan.json', plan)
        (output / 'index.html').write_text('''<!doctype html><meta charset="utf-8"><h1>CEC2011 HPPSO grid search</h1>
<p>32 settings per problem. Population 200 and 200,000 evaluations. Tuning seeds 3–5, comparison seeds 0–2.</p>
<p><a href="plan.json">Grid settings</a> · <a href="tables/index.html">Completed comparisons</a></p><pre id="status"></pre>
<script>async function update(){try{document.getElementById('status').textContent=JSON.stringify(await(await fetch('progress.json?t='+Date.now())).json(),null,2)}catch(e){}}update();setInterval(update,3000)</script>''', encoding='utf-8')
        server = ThreadingHTTPServer(('127.0.0.1', 8778), partial(SimpleHTTPRequestHandler, directory=str(output)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        selected = {}
        with ProcessPoolExecutor(max_workers=2) as pool:
            for problem in config['problems']:
                number = problem['problem']
                common = dict(problem=number, dimension=problem['dimension'], algorithm='HPPSO',
                              population=200, budget=BUDGET, schedule_horizon=BUDGET)
                status = dict(problem=number, completed_problems=len(selected), total_problems=22)
                tuning = [dict(common, stage='tuning', setting=label, seed=seed, **params)
                          for label, params in settings.items() for seed in TUNING_SEEDS]
                rows = batch(pool, output, tuning, fingerprint, dict(status, stage='tuning'))
                scores = summarize(rows)
                scores.to_csv(output / f'T{number}_grid_scores.csv')
                if scores.empty:
                    raise RuntimeError(f'T{number} has no complete tuning setting')
                winner = scores.index[0]
                comparison = [dict(common, stage='comparison', setting=winner, seed=seed, **settings[winner])
                              for seed in COMPARISON_SEEDS]
                rows = batch(pool, output, comparison, fingerprint, dict(status, stage='comparison', selected=winner))
                scores = summarize(rows)
                if scores.empty:
                    raise RuntimeError(f'T{number} has an incomplete comparison seed set')
                mask = stats.problem.eq(number) & stats.algorithm.eq('HPPSO')
                for column in ('count', 'mean', 'median', 'std', 'min', 'max'):
                    stats.loc[mask, column] = scores.loc[winner, column]
                selected[number] = dict(problem=number, setting=winner, **settings[winner])
                report(output, stats, selected)
        from experiments.cec2011_wilcoxon import main as wilcoxon
        sys.argv = [sys.argv[0], '--output', str(output)]
        wilcoxon()
        base.atomic(output / 'progress.json', dict(stage='complete', completed_problems=22, total_problems=22))
    threading.Event().wait()


if __name__ == '__main__':
    main()
