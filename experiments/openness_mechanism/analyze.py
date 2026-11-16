"""Aggregate matched observations, test the planned contrasts, and draw three figures."""

import argparse
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np
import pandas as pd

from experiments.openness_mechanism.run import RULES
from experiments.final_benchmark import read_result
from experiments.trait_study import result_path


PEER_RULES = RULES[:3]
COLORS = dict(zip(RULES, ('#007783', '#b05b1b', '#635bb4', '#596270')))
LABELS = {'Worse': 'Worse peer', 'Better': 'Better peer', 'Random': 'Random peer', 'None': 'No attraction'}


def interval(values, cfg):
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    draws = rng.integers(0, len(values), (cfg['bootstrap_resamples'], len(values)))
    return np.quantile(values[draws].mean(axis=1), [.025, .975])


def contrasts(data, metrics, peers, cfg):
    rows = []
    for function, frame in data.groupby('function', sort=False):
        for metric in metrics:
            pivot = frame.pivot(index='seed', columns='rule', values=metric).reindex(columns=RULES)
            for peer in peers:
                values = (pivot.Worse - pivot[peer]).dropna().to_numpy()
                n = len(values)
                if n:
                    low, high = interval(values, cfg)
                    flips = np.asarray(list(product((-1, 1), repeat=n)))
                    observed = abs(values.sum())
                    threshold = observed - 1e-12 * np.abs(values).sum()
                    p = np.mean(np.abs(flips @ values) >= threshold)
                    mean, median = values.mean(), np.median(values)
                else:
                    low = high = p = mean = median = np.nan
                rows.append(dict(function=function, outcome=metric, comparator=peer, n_seeds=n,
                                 mean_worse_minus_comparator=mean, median_worse_minus_comparator=median,
                                 ci95_low=low, ci95_high=high, p_exact=p))
    result = pd.DataFrame(rows)
    for _, group in result.groupby('outcome'):
        p = group.p_exact.dropna().sort_values()
        # Each outcome uses its planned family across functions and comparisons.
        adjusted = np.maximum.accumulate(np.minimum(1, p.to_numpy() * np.arange(len(group), len(group)-len(p), -1)))
        result.loc[p.index, 'p_holm'] = adjusted
    return result


def figures(output, cfg, names, historical, recovery, summary):
    plt.rcParams.update({'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    for figure in (1, 2, 3):
        fig, axes = plt.subplots(2, 2, figsize=(9, 6.6), layout='constrained')
        for ax, function in zip(axes.flat, cfg['functions'], strict=True):
            hs = historical.loc[historical.function.eq(function)]
            rs = recovery.loc[recovery.function.eq(function)]
            n = rs.seed.nunique()
            if figure == 1:
                for i, state in enumerate((False, True)):
                    frame = hs.loc[hs.stagnant.eq(state)]
                    ax.text(i, .97, f'{frame.seed.nunique()} seeds', ha='center', va='top',
                            transform=ax.get_xaxis_transform(), fontsize=8)
                    for j, rule in enumerate(PEER_RULES):
                        values = frame.loc[frame.rule.eq(rule), 'distance'].to_numpy()
                        if len(values):
                            x = i + (j - 1) * .19
                            lo, hi = interval(values, cfg)
                            ax.vlines(x, lo, hi, color=COLORS[rule], lw=1.3)
                            ax.scatter(x, values.mean(), color=COLORS[rule], s=25, zorder=3)
                ax.set_xticks([0, 1], ['Non-stagnated', 'Stagnated'])
                ax.set_xlim(-.45, 1.45)
                ax.margins(y=.19)
            else:
                ax.set_xticks(range(4), [LABELS[r] for r in RULES], fontsize=8)
                ax.set_xlim(-.35, 3.35)
                if figure == 2:
                    values = rs.pivot(index='seed', columns='rule', values='improvement').reindex(columns=RULES)
                    offsets = np.linspace(-.13, .13, n) if n > 1 else np.zeros(n)
                    for offset, row in zip(offsets, values.to_numpy(), strict=True):
                        x = np.arange(4) + offset
                        ax.plot(x, row, color='#a7a9ad', alpha=.55, lw=.65)
                        ax.scatter(x, row, color=[COLORS[r] for r in RULES], s=14, zorder=3)
                    if n:
                        ax.scatter(range(4), values.median().to_numpy(), color='#1e2530', marker='_', s=130, zorder=4)
                    ax.yaxis.set_major_locator(MaxNLocator(4))
                    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3), useMathText=True)
                else:
                    frame = summary.loc[summary.function.eq(function)].set_index('rule')
                    for j, rule in enumerate(RULES):
                        if rule in frame.index:
                            row = frame.loc[rule]
                            ax.vlines(j, row.recovery_ci95_low, row.recovery_ci95_high, color=COLORS[rule], lw=1.5)
                            ax.scatter(j, row.recovery_probability, color=COLORS[rule], s=30, zorder=3)
                    ax.set_ylim(-.055, 1.055)
                    ax.set_yticks(np.linspace(0, 1, 5))
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
            suffix = f' · {n} seeds' if figure > 1 else ''
            ax.set_title(names[function] + suffix, loc='left', fontsize=10, fontweight='bold', pad=8)
            ax.grid(axis='y', color='#e3e6eb', linewidth=.75)
            ax.set_axisbelow(True)
        if figure == 1:
            handles = [plt.Line2D([], [], marker='o', ls='', color=COLORS[r], label=LABELS[r]) for r in PEER_RULES]
            fig.legend(handles=handles, loc='outside lower center', ncols=3, frameon=False)
        title, ylabel, filename = {
            1: ('Historical memory distance by swarm state', 'Normalized distance to global best', '01_memory_distance'),
            2: ('Global-best improvement after ten updates', r'Normalized improvement $I_{10}$', '02_stagnation_recovery'),
            3: ('Recovery within ten updates', r'Seed-averaged recovery probability ($I_{10}\geq 10^{-3}$)', '03_recovery_probability'),
        }[figure]
        fig.suptitle(title, fontweight='bold')
        fig.supylabel(ylabel)
        if figure == 2:
            fig.supxlabel('Connected dots show matched seed means. Black bars mark medians.', fontsize=9)
        elif figure == 3:
            fig.supxlabel('Dots show seed means. Bars show 95% seed-bootstrap intervals.', fontsize=9)
        fig.savefig(output / f'{filename}.pdf')
        plt.close(fig)


def analyze(output):
    plan = json.loads((output/'plan.json').read_text())
    cfg = plan['config']
    raw_a, raw_b, coverage = [], [], []
    for function in cfg['functions']:
        for seed in cfg['seeds']:
            job = dict(function=function, seed=seed)
            record = read_result(result_path(output, job))
            assert record['fingerprint'] == plan['fingerprint']
            assert record['replay_evaluations'] == cfg['budget']
            assert 0 <= record['snapshots'] <= cfg['max_snapshots']
            assert len(record['outcomes']) == len(RULES) * record['snapshots']
            assert record['branch_evaluations'] == len(record['outcomes']) * cfg['population'] * cfg['horizon']
            raw_a.extend(dict(job, **row) for row in record['memories'])
            raw_b.extend(dict(job, **row) for row in record['outcomes'])
            coverage.append(dict(job, **{key: record[key] for key in ('snapshots', 'stagnant_states',
                'eligible_stagnant_states', 'replay_evaluations', 'branch_evaluations')}))
    a, b, counts = map(pd.DataFrame, (raw_a, raw_b, coverage))
    a.to_csv(output/'historical_states.csv', index=False)
    b.to_csv(output/'branch_snapshots.csv', index=False)
    counts.to_csv(output/'coverage.csv', index=False)
    keys = ['function', 'seed', 'rule']
    a_seed = a.groupby(keys + ['stagnant'], sort=False).agg(distance=('distance','mean'),
        retained_improvement=('retained_improvement','mean'), states=('update','size')).reset_index()
    b_seed = b.groupby(keys, sort=False).agg(improvement=('improvement','mean'), recovery=('recovered','mean'),
        distance=('distance','median'), retained_improvement=('retained_improvement','median'),
        alignment=('alignment','mean'), observed_events=('event','sum'), snapshots=('update','size'),
        restricted_time=('time','mean')).reset_index()
    a_seed.to_csv(output/'historical_by_seed.csv', index=False)
    b_seed.to_csv(output/'recovery_by_seed.csv', index=False)
    primary = contrasts(b_seed, ['improvement'], RULES[1:], cfg)
    primary.to_csv(output/'primary_comparisons.csv', index=False)
    h1 = []
    for state in (False, True):
        table = contrasts(a_seed.loc[a_seed.stagnant.eq(state)], ['distance', 'retained_improvement'], PEER_RULES[1:], cfg)
        h1.append(table.assign(stagnant=state))
    pd.concat(h1, ignore_index=True).to_csv(output/'historical_comparisons.csv', index=False)
    secondary = [contrasts(b_seed, ['recovery', 'restricted_time'], RULES[1:], cfg),
                 contrasts(b_seed, ['alignment', 'distance', 'retained_improvement'], PEER_RULES[1:], cfg)]
    pd.concat(secondary, ignore_index=True).to_csv(output/'supporting_comparisons.csv', index=False)

    rows = []
    for (function, rule), frame in b_seed.groupby(['function','rule'], sort=False):
        lo, hi = interval(frame.recovery.to_numpy(), cfg)
        raw = b.loc[b.function.eq(function) & b.rule.eq(rule)]
        rows.append(dict(function=function, rule=rule, seeds=len(frame), snapshots=frame.snapshots.sum(),
            mean_improvement=frame.improvement.mean(), median_improvement=frame.improvement.median(),
            recovery_probability=frame.recovery.mean(), recovery_ci95_low=lo, recovery_ci95_high=hi,
            snapshot_recovery_fraction=raw.recovered.mean(), median_selected_distance=frame.distance.median(),
            mean_alignment=frame.alignment.mean(), undefined_alignment=int(raw.alignment.isna().sum()) if rule != 'None' else None,
            mean_restricted_time=frame.restricted_time.mean(), strict_improvements=frame.observed_events.sum()))
    summary = pd.DataFrame(rows)
    summary.to_csv(output/'branch_summary.csv', index=False)
    a_seed.groupby(['function','stagnant','rule'], sort=False).agg(
        seeds=('seed','size'), mean_distance=('distance','mean'),
        mean_retained_improvement=('retained_improvement','mean')).to_csv(output/'historical_summary.csv')
    figures(output, cfg, plan['settings']['function_display_names'], a_seed, b_seed, summary)
    print(counts.groupby('function', sort=False).snapshots.agg(['sum', lambda x: (x>0).sum()]).to_string())
    print(counts[['replay_evaluations','branch_evaluations']].sum().to_string())
    print(primary.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    analyze(parser.parse_args().output)
