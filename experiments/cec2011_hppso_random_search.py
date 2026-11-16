"""Stratified random search of HPPSO settings at 200,000 evaluations."""
from experiments.cec2011_hppso_grid import (
    base, batch, summarize, report, BUDGET, TUNING_SEEDS, COMPARISON_SEEDS,
)
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import sys
import threading
import pandas as pd


OPTIONS = dict(theta=[i / 10 for i in range(10)],
               alpha_mut=[0.001, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3],
               eta=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
               lam=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
               stagnation_threshold=[5, 10, 20, 30, 50])
SEARCH_SEED = 20261109
DEFAULT = dict(theta=0.5, alpha_mut=0.05, eta=0.001, lam=0.4, stagnation_threshold=10)


def sample_settings():
    rng = random.Random(SEARCH_SEED)
    settings = {'default': DEFAULT}
    for theta in OPTIONS['theta']:
        while True:
            candidate = {'theta': theta}
            candidate.update({key: rng.choice(values) for key, values in OPTIONS.items() if key != 'theta'})
            if candidate != DEFAULT:
                break
        settings[f'random_theta_{theta:.1f}'] = candidate
    return settings


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
    settings = sample_settings()
    digest = hashlib.sha256(baseline['fingerprint'].encode())
    for path in (Path(__file__), Path(base.__file__), base.ROOT / 'experiments/cec2011_hppso_grid.py',
                 base.ROOT / 'src/hppso/algorithms/hppso.py', base.ROOT / 'src/hppso/algorithms/_pso_bounds.py',
                 base.ROOT / 'experiments/optimizer_checkpoint.py'):
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    output = results / 'cec2011_hppso_random_200k'
    output.mkdir(exist_ok=True)
    plan = dict(fingerprint=fingerprint, baseline_fingerprint=baseline['fingerprint'], options=OPTIONS,
                settings=settings, search_seed=SEARCH_SEED, tuning_seeds=TUNING_SEEDS,
                comparison_seeds=COMPARISON_SEEDS, population=200, budget=BUDGET,
                tuning_trials=726, comparison_trials=66)
    with base.shard_lease(output, 'hppso-random'):
        if (output / 'plan.json').exists():
            assert json.loads((output / 'plan.json').read_text()) == plan
        else:
            base.atomic(output / 'plan.json', plan)
        (output / 'index.html').write_text('''<!doctype html><meta charset="utf-8"><h1>CEC2011 HPPSO random search</h1>
<p>Ten stratified random settings plus default HPPSO per problem. Population 200 and 200,000 evaluations. Tuning seeds 3–5, comparison seeds 0–2.</p>
<p><a href="plan.json">Options and sampled settings</a> · <a href="tables/index.html">Completed comparisons</a></p><pre id="status"></pre>
<script>async function update(){try{document.getElementById('status').textContent=JSON.stringify(await(await fetch('progress.json?t='+Date.now())).json(),null,2)}catch(e){}}update();setInterval(update,3000)</script>''', encoding='utf-8')
        server = ThreadingHTTPServer(('127.0.0.1', 8779), partial(SimpleHTTPRequestHandler, directory=str(output)))
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
                scores = summarize(batch(pool, output, tuning, fingerprint, dict(status, stage='tuning')))
                scores.to_csv(output / f'T{number}_random_scores.csv')
                if scores.empty:
                    raise RuntimeError(f'T{number} has no complete tuning setting')
                winner = scores.index[0]
                comparison = [dict(common, stage='comparison', setting=winner, seed=seed, **settings[winner])
                              for seed in COMPARISON_SEEDS]
                scores = summarize(batch(pool, output, comparison, fingerprint, dict(status, stage='comparison', selected=winner)))
                if scores.empty:
                    raise RuntimeError(f'T{number} has an incomplete comparison seed set')
                mask = stats.problem.eq(number) & stats.algorithm.eq('HPPSO')
                for column in ('count', 'mean', 'median', 'std', 'min', 'max'):
                    stats.loc[mask, column] = scores.loc[winner, column]
                selected[number] = dict(problem=number, setting=winner, **settings[winner])
                report(output, stats, selected)
                page = output / 'tables/index.html'
                page.write_text(page.read_text(encoding='utf-8').replace('Grid-tuned HPPSO', 'Random-search HPPSO'), encoding='utf-8')
        from experiments.cec2011_wilcoxon import main as wilcoxon
        sys.argv = [sys.argv[0], '--output', str(output)]
        wilcoxon()
        base.atomic(output / 'progress.json', dict(stage='complete', completed_problems=22, total_problems=22))
    threading.Event().wait()


if __name__ == '__main__':
    main()
