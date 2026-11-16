"""Friedman test over CEC2011 problems with Holm-adjusted paired sign tests."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, friedmanchisquare

from experiments.cec2011_seed_ranks import ranks


def analyze(means):
    within_problem = np.stack([ranks(scores) for scores in means.to_numpy()])
    if np.all(within_problem == within_problem[:, :1]):
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = friedmanchisquare(*within_problem.T)
    average = pd.Series(within_problem.mean(axis=0), index=means.columns, name='mean_rank')
    average = average.sort_values().to_frame()
    pairs = []
    if p_value < .05:
        for i, j in combinations(range(len(means.columns)), 2):
            difference = within_problem[:, i] - within_problem[:, j]
            wins = int((difference < 0).sum())
            losses = int((difference > 0).sum())
            p = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
            pairs.append(dict(algorithm_a=means.columns[i], algorithm_b=means.columns[j],
                              a_wins=wins, b_wins=losses, ties=len(means)-wins-losses, p_raw=p))
    tests = pd.DataFrame(pairs, columns=['algorithm_a', 'algorithm_b', 'a_wins', 'b_wins', 'ties', 'p_raw'])
    tests = tests.sort_values('p_raw', kind='stable')
    tests['p_holm'] = np.minimum(1, tests.p_raw.mul(np.arange(len(tests), 0, -1)).cummax())
    tests['significant'] = tests.p_holm.lt(.05)
    return statistic, p_value, average, tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    args = parser.parse_args()
    scores = pd.read_csv(args.scores).rename(columns={'best_objective': 'best_score'})
    if 'status' in scores and not (scores.status.eq('budget_reached') & scores.evaluations.eq(200000)).all():
        raise ValueError('The score export includes unfinished or mismatched runs')
    wide = scores.pivot(index=['problem', 'seed'], columns='algorithm', values='best_score')
    seeds = set(scores.seed)
    if (wide.shape != (22 * len(seeds), 9) or set(scores.problem) != set(range(1, 23))
            or seeds not in (set(range(3)), set(range(10))) or not np.isfinite(wide.to_numpy()).all()):
        raise ValueError('Expected a complete matched CEC2011 score export')
    means = wide.groupby(level='problem').mean()
    profiles = {p: block.to_numpy() for p, block in wide.groupby(level='problem')}
    duplicates = sorted({b for a, b in combinations(profiles, 2) if np.array_equal(profiles[a], profiles[b])})
    analyses = [('All problems', '', means)]
    if duplicates:
        analyses.append((f'Sensitivity omitting duplicate score profiles {duplicates}',
                         '_distinct_profiles', means.drop(index=duplicates)))

    folder = args.scores.parent / 'friedman'
    folder.mkdir(exist_ok=True)
    page = ('<!doctype html><meta charset="utf-8"><title>CEC2011 Friedman analysis</title>'
            '<h1>CEC2011 Friedman analysis</h1>'
            f'<p>One block per problem. Scores are means of final best objectives over {len(seeds)} matched seeds. '
            'Friedman ranks the nine algorithms within each block, with average ranks for numerical ties '
            '(relative and absolute tolerance 1e-12). When the omnibus test rejects at 0.05, exact two-sided '
            'paired sign tests count wins and losses across problems and omit ties. Holm adjusts all 36 '
            'pairwise p-values. A significant omnibus result does not establish a full ordering.</p>')
    for label, suffix, subset in analyses:
        statistic, p_value, average, tests = analyze(subset)
        rank_csv = f'average_ranks{suffix}.csv'
        pair_csv = f'pairwise_sign_tests{suffix}.csv'
        average.to_csv(folder / rank_csv)
        tests.to_csv(folder / pair_csv, index=False)
        significant = tests[tests.significant]
        page += (f'<h2>{label}</h2><p>{len(subset)} problems. Friedman chi-square with 8 degrees of freedom '
                 f'= {statistic:.6g}, p = {p_value:.6g}. Significant post hoc pairs = {len(significant)}.</p>'
                 f'<p><a href="{rank_csv}">Average ranks</a> · <a href="{pair_csv}">Pairwise tests</a></p>'
                 + average.to_html(float_format=lambda x: f'{x:.4g}')
                 + significant.to_html(index=False, float_format=lambda x: f'{x:.4g}'))
        print(f'{label}: Friedman chi-square={statistic:.8g}, p={p_value:.8g}, significant pairs={len(significant)}')
        print(average.to_string())
        print(significant[significant.algorithm_a.eq('HPPSO') | significant.algorithm_b.eq('HPPSO')].to_string(index=False))
    (folder / 'index.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    main()
