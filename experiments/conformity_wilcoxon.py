"""Rank the conformity variants using paired tests within each benchmark."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.cec2011_problem_wilcoxon import exact_signed_rank


FUNCTIONS = {
    'f1_sphere': ('Shifted Sphere', 'Unimodal, strictly convex, separable'),
    'f10_levy': ('Shifted Lévy', 'Multimodal, nonconvex, separable'),
    'f13_michalewicz': ('Shifted Michalewicz', 'Multimodal, nonconvex, separable'),
    'f26_rotated_griewank': ('Rotated Griewank', 'Multimodal, nonconvex, nonseparable'),
    'f7_ackley': ('Shifted Ackley', 'Multimodal, nonconvex, nonseparable'),
    'f22_rotated_ellipsoid': ('Rotated Ellipsoid', 'Unimodal, strictly convex, nonseparable, ill-conditioned'),
    'f4_rosenbrock': ('Shifted Rosenbrock', 'Multimodal at 100D, nonconvex, nonseparable, narrow valley'),
    'f25_composition': ('Composition', 'Multimodal, nonconvex, nonseparable, three rotated components'),
    'f28_schwefel_1_2': ('Schwefel 1.2', 'Unimodal, strictly convex, nonseparable'),
    'f29_rotated_schaffer_f7': ('Rotated Schaffer F7', 'Multimodal, nonconvex, nonseparable'),
}

ALGORITHMS = ('PSO', 'PSO-conformity-individual', 'PSO-conformity-uniform')


def report(output, algorithms=ALGORITHMS, study='Conformity', *, variant_labels=None):
    runs = pd.read_csv(output / 'runs.csv')
    if variant_labels is not None:
        runs = runs.loc[runs.variant.isin(variant_labels)].assign(label=lambda frame: frame.variant.map(variant_labels))
    plan = output / 'plan.json'
    budget = json.loads(plan.read_text(encoding='utf-8'))['config']['budget'] if plan.exists() else 60_000
    if not (runs.status.eq('budget_reached').all() and runs.evaluations.eq(budget).all()):
        raise ValueError(f'Expected completed runs at {budget:,} evaluations')
    if set(runs.function) != set(FUNCTIONS) or set(runs.label) != set(algorithms):
        raise ValueError(f'Unexpected function or algorithm in the {study} study')

    rank_rows, test_rows = [], []
    for function, (name, description) in FUNCTIONS.items():
        scores = runs.loc[runs.function.eq(function)].pivot(index='seed', columns='label', values='best_score')
        if set(scores.index) != set(range(10)) or not np.isfinite(scores.to_numpy()).all():
            raise ValueError(f'Expected 10 complete matched seeds for {function}')
        tests = []
        for a, b in combinations(algorithms, 2):
            differences = (scores[a] - scores[b]).to_numpy()
            positive, negative, p = exact_signed_rank(differences)
            tests.append(dict(function=function, algorithm_a=a, algorithm_b=b,
                              nonzero_pairs=int(np.count_nonzero(differences)),
                              positive_rank_sum=positive, negative_rank_sum=negative,
                              median_difference_a_minus_b=float(np.median(differences)), p_raw=p))

        adjusted = 0.0
        net_wins = dict.fromkeys(algorithms, 0)
        for i, test in enumerate(sorted(tests, key=lambda row: row['p_raw'])):
            adjusted = max(adjusted, min(1.0, (len(tests) - i) * test['p_raw']))
            test['p_holm'] = adjusted
            test['significant_winner'] = ''
            if adjusted <= 0.05:
                a, b = test['algorithm_a'], test['algorithm_b']
                winner, loser = (a, b) if test['negative_rank_sum'] > test['positive_rank_sum'] else (b, a)
                test['significant_winner'] = winner
                net_wins[winner] += 1
                net_wins[loser] -= 1

        ranks = pd.Series(net_wins).rank(ascending=False, method='average')
        rank_rows.append({'Function': name, 'Type': description, **ranks.to_dict()})
        test_rows.extend(tests)

    folder = output / 'wilcoxon'
    folder.mkdir(exist_ok=True)
    ranks = pd.DataFrame(rank_rows)
    tests = pd.DataFrame(test_rows)
    ranks.to_csv(folder / 'ranks_by_function.csv', index=False)
    tests.to_csv(folder / 'pairwise_tests.csv', index=False)
    means = ranks[list(algorithms)].mean().rename('Mean rank').to_frame()
    pair_count = len(algorithms) * (len(algorithms) - 1) // 2
    (folder / 'index.html').write_text(
        f'<!doctype html><meta charset="utf-8"><title>{study} Wilcoxon ranks</title>'
        '<style>body{font:16px system-ui;margin:2rem}table{border-collapse:collapse}'
        'th,td{padding:.5rem;border:1px solid #bbb;text-align:left}</style>'
        f'<h1>{study} ablation: Wilcoxon ranks by function</h1>'
        '<p>Exact two-sided signed-rank tests compare final objectives on 10 matched seeds '
        f'per function. Holm correction covers all {pair_count} comparisons within each function at α = 0.05. '
        'Ranks order algorithms by significant wins minus significant losses. Equal scores share '
        'the average of the occupied ranks. Rank 1 is best.</p>'
        '<p><a href="ranks_by_function.csv">Ranks CSV</a> · '
        '<a href="pairwise_tests.csv">Pairwise tests and adjusted p-values</a></p>'
        + ranks.to_html(index=False, float_format=lambda value: f'{value:g}')
        + '<h2>Mean ranks across functions</h2>' + means.to_html(float_format=lambda value: f'{value:g}')
        + '<p>Function types follow the implemented objectives. '
        'The 100D Rosenbrock classification was checked numerically by locating a second '
        'interior minimum with a positive definite Hessian.</p>',
        encoding='utf-8',
    )
    print(ranks.to_string(index=False))
    print(means.to_string())
    print(f'Significant comparisons: {tests.p_holm.le(0.05).sum()}/{len(tests)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    report(parser.parse_args().output)
