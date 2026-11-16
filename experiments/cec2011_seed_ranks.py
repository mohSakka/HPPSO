"""Rank CEC2011 algorithms within each problem and seed at a matched budget."""

import argparse
import gzip
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ranks(scores):
    order = np.argsort(scores)
    result = np.empty(len(order))
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and np.isclose(
            scores[order[end]], scores[order[start]], rtol=1e-12, atol=1e-12
        ):
            end += 1
        result[order[start:end]] = (start + 1 + end) / 2
        start = end
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads((args.output / 'plan.json').read_text())
    baseline_plan = json.loads((args.baseline / 'plan.json').read_text())
    if plan['baseline_fingerprint'] != baseline_plan['fingerprint']:
        raise ValueError('The baseline does not match the experiment manifest')
    algorithms = baseline_plan['config']['algorithms']
    rows = []
    for job in plan['jobs']:
        problem, seed = job['problem'], job['seed']
        for algorithm in algorithms:
            root = args.output if algorithm == 'HPPSO' else args.baseline
            path = root / 'trials' / f'T{problem}' / algorithm.replace(' ', '_') / f'seed_{seed:03d}.json.gz'
            with gzip.open(path, 'rt') as file:
                trial = json.load(file)
            fingerprint = plan['fingerprint'] if algorithm == 'HPPSO' else baseline_plan['fingerprint']
            expected = dict(problem=problem, seed=seed, algorithm=algorithm, fingerprint=fingerprint,
                            evaluations=job['budget'], population=job['population'], status='budget_reached')
            if any(trial[key] != value for key, value in expected.items()) or not np.isfinite(trial['best_score']):
                raise ValueError(f'Invalid or mismatched result in {path}')
            rows.append((problem, seed, algorithm, trial['best_score']))

    scores = pd.DataFrame(rows, columns=['problem', 'seed', 'algorithm', 'best_objective'])
    wide = scores.pivot(index=['problem', 'seed'], columns='algorithm', values='best_objective')
    if wide.isna().any().any():
        raise ValueError('Missing algorithm results in a problem/seed block')
    ranked = pd.DataFrame(np.stack([ranks(row) for row in wide.to_numpy()]),
                          index=wide.index, columns=wide.columns)
    per_problem = ranked.groupby(level='problem').mean()
    overall = per_problem.mean().sort_values().rename('mean_rank').to_frame()
    overall.insert(0, 'position', overall.mean_rank.rank(method='min').astype(int))
    profiles = {p: block.to_numpy() for p, block in wide.groupby(level='problem')}
    duplicates = {b for a, b in combinations(profiles, 2) if np.array_equal(profiles[a], profiles[b])}
    folder = args.output / 'tables'
    ranked.stack().rename('rank').to_frame().join(scores.set_index(['problem', 'seed', 'algorithm'])).to_csv(
        folder / 'seed_ranks.csv')
    per_problem.to_csv(folder / 'problem_mean_seed_ranks.csv')
    overall.to_csv(folder / 'overall_mean_seed_ranks.csv')
    page = ('<!doctype html><meta charset="utf-8"><title>CEC2011 mean seed ranks</title>'
            '<h1>CEC2011 mean seed ranks</h1><p>Algorithms are ranked by final best objective within '
            'each problem and seed. Numerical ties share the average occupied rank '
            '(relative and absolute tolerance 1e-12). Seed ranks are averaged within each problem, '
            'then problem ranks are averaged with equal weight. Lower ranks are better.</p>'
            '<p><a href="seed_ranks.csv">Raw scores and seed ranks</a> · '
            '<a href="problem_mean_seed_ranks.csv">Problem ranks CSV</a> · '
            '<a href="overall_mean_seed_ranks.csv">Overall ranks CSV</a></p>'
            + overall.round(4).to_html() + '<h2>Mean seed ranks by problem</h2>' + per_problem.round(4).to_html())
    if duplicates:
        sensitivity = per_problem.drop(index=sorted(duplicates)).mean().sort_values().rename('mean_rank')
        sensitivity.to_csv(folder / 'overall_mean_seed_ranks_distinct_profiles.csv')
        page += (f'<h2>Identical score profiles</h2><p>Sensitivity after dropping problems '
                 f'{sorted(duplicates)}, whose raw score profiles duplicate earlier problems.</p>'
                 + sensitivity.round(4).to_frame().to_html())
    (folder / 'seed_ranks.html').write_text(page, encoding='utf-8')
    print(overall.to_string())
    print('Duplicate raw profiles omitted in sensitivity:', sorted(duplicates))


if __name__ == '__main__':
    main()
