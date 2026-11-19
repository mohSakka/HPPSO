"""Rank final benchmark results by significant paired wins minus losses."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def exact_signed_rank(differences):
    nonzero = differences[differences != 0]
    # Double average ranks to make tied ranks integral. Count subset sums
    # instead of enumerating all 2**n possible sign assignments.
    ranks = np.rint(2 * rankdata(np.abs(nonzero))).astype(int)
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    total = positive + negative
    counts = np.zeros(total + 1, dtype=np.int64)
    counts[0] = 1
    for rank in ranks:
        counts[rank:] += counts[:-rank].copy()
    extreme = np.abs(2 * np.arange(total + 1) - total) >= abs(positive - negative)
    return positive / 2, negative / 2, counts[extreme].sum() / 2**len(ranks)


def report(source):
    plan = json.loads((source / 'plan.json').read_text(encoding='utf-8'))
    cfg = plan['config']
    runs = pd.read_csv(source / 'runs.csv', float_precision='round_trip')
    ranks, pairs = [], []
    for dim in cfg['dimensions']:
        names = plan['algorithms'][str(dim)]
        budget = dim * cfg['evaluations_per_dimension']
        for function in plan['functions']:
            block = runs.loc[runs.dimension.eq(dim) & runs.function.eq(function)]
            if not (block.status.eq('budget_reached').all() and block.evaluations.eq(budget).all()):
                raise ValueError(f'Invalid or incomplete runs for {dim}D {function}')
            scores = block.pivot(index='seed', columns='algorithm', values='best_score')
            if (set(scores.index) != set(cfg['seeds']) or set(scores.columns) != set(names)
                    or not np.isfinite(scores.to_numpy()).all()):
                raise ValueError(f'Missing or non-finite matched objectives for {dim}D {function}')
            tests = []
            for a, b in combinations(names, 2):
                d = (scores[a] - scores[b]).to_numpy()
                positive, negative, p = exact_signed_rank(d)
                tests.append(dict(dimension=dim, function=function, algorithm_a=a, algorithm_b=b,
                                  n_seeds=len(d), nonzero_pairs=np.count_nonzero(d),
                                  median_difference=float(np.median(d)), p_raw=p,
                                  positive_rank_sum=positive, negative_rank_sum=negative))
            wins, losses = dict.fromkeys(names, 0), dict.fromkeys(names, 0)
            adjusted = 0.0
            for i, test in enumerate(sorted(tests, key=lambda test: test['p_raw'])):
                adjusted = min(1.0, max(adjusted, (len(tests) - i) * test['p_raw']))
                test.update(p_holm=adjusted, significant_winner='')
                if adjusted <= .05:
                    a, b = test['algorithm_a'], test['algorithm_b']
                    winner, loser = (a, b) if test['negative_rank_sum'] > test['positive_rank_sum'] else (b, a)
                    wins[winner] += 1
                    losses[loser] += 1
                    test['significant_winner'] = winner
            pairs.extend(tests)
            net = pd.Series({name: wins[name] - losses[name] for name in names})
            for name, rank in net.rank(ascending=False, method='average').items():
                ranks.append(dict(dimension=dim, function=function, algorithm=name, rank=rank,
                                  significant_wins=wins[name], significant_losses=losses[name]))
    ranks, pairs = pd.DataFrame(ranks), pd.DataFrame(pairs)
    by_function = ranks.pivot(index=['dimension', 'function'], columns='algorithm', values='rank')
    order = [(d, f) for d in cfg['dimensions'] for f in plan['functions']]
    by_function = by_function.reindex(order)
    aggregates = dict(mean_rank=('rank', 'mean'), median_rank=('rank', 'median'),
                      n_functions=('rank', 'count'))
    by_dimension = ranks.groupby(['dimension', 'algorithm'], as_index=False).agg(**aggregates)
    by_dimension = by_dimension.sort_values(['dimension', 'mean_rank', 'algorithm'])
    overall = ranks.groupby('algorithm', as_index=False).agg(**aggregates).sort_values('mean_rank')
    output = source / 'wilcoxon'
    output.mkdir(exist_ok=True)
    for name, frame in [('ranks_by_function', by_function.reset_index()), ('rank_details', ranks),
                        ('rank_statistics_by_dimension', by_dimension), ('rank_statistics_overall', overall),
                        ('pairwise_tests', pairs)]:
        frame.to_csv(output / f'{name}.csv', index=False)
    fmt = lambda x: f'{x:.3f}'.rstrip('0').rstrip('.')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Final benchmark Wilcoxon ranks</title>'
            '<style>body{font:16px system-ui;margin:2rem}table{border-collapse:collapse;margin-bottom:1rem}'
            'td,th{padding:.35rem .55rem;border:1px solid #ccd6df;text-align:right}thead th{background:#edf3f8}</style>'
            '<h1>Final benchmark Wilcoxon ranks</h1>'
            '<p>For each function and dimension, exact two-sided paired Wilcoxon signed-rank tests compare '
            'final costs on matched seeds. Zero differences are omitted and tied magnitudes receive average ranks. '
            'Holm correction is applied within each function and dimension at 0.05. Variants are ranked by '
            'significant wins minus significant losses. Equal totals share the average of the occupied ranks.</p>'
            f'<p>{pairs.p_holm.le(.05).sum():,} of {len(pairs):,} comparisons were significant across '
            f'{len(by_function)} function–dimension blocks.</p>'
            '<p><a href="pairwise_tests.csv">Pairwise tests</a> · <a href="rank_details.csv">Wins and losses</a> · '
            '<a href="ranks_by_function.csv">Per-function ranks</a> · '
            '<a href="rank_statistics_by_dimension.csv">Rank statistics by dimension</a> · '
            '<a href="rank_statistics_overall.csv">Pooled rank statistics</a></p>'
            '<h2>Pooled rank statistics</h2><p>Each algorithm is summarized over its available blocks. '
            'CMA-ES appears only at 30D and Sep-CMA-ES only at 300D, 600D, and 1000D.</p>'
            + overall.to_html(index=False, float_format=fmt))
    for dim in cfg['dimensions']:
        page += f'<h2>{dim}D</h2>' + by_dimension.loc[by_dimension.dimension.eq(dim)].to_html(
            index=False, float_format=fmt)
        page += by_function.loc[dim, plan['algorithms'][str(dim)]].to_html(float_format=fmt)
    (output / 'index.html').write_text(page + '</html>', encoding='utf-8')
    print(by_dimension.to_string(index=False))
    print(overall.to_string(index=False))
    print(f'Significant: {pairs.p_holm.le(.05).sum()}/{len(pairs)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    report(parser.parse_args().source)
