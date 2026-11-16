"""Summarize projected coverage across functions and matched seeds."""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.openness_mechanism.analyze import COLORS, LABELS, contrasts, interval
from experiments.openness_mechanism.run import RULES
from experiments.final_benchmark import read_result
from experiments.trait_study import result_path


def report(output):
    plan = json.loads((output/'plan.json').read_text())
    source, cfg = plan['source'], plan['config']
    detail, raw, outcomes = [], [], []
    evaluations = 0
    for function in source['functions']:
        for seed in source['seeds']:
            job = dict(function=function, seed=seed)
            run = read_result(result_path(output, job))
            assert run['fingerprint'] == plan['fingerprint'] and run['original_worse_verified']
            evaluations += run['evaluations']
            for epoch, counts, best in zip(run['epochs'], run['counts'], run['best_objectives'], strict=True):
                indexed = dict(job, epoch=epoch, evaluations_per_rule=(epoch+1)*source['population'])
                for rule, objective in zip(RULES, best, strict=True):
                    outcomes.append(dict(indexed, rule=rule, best_objective=objective))
                for width, arms in zip(cfg['cell_widths'], counts, strict=True):
                    for rule, cells in zip(RULES, arms, strict=True):
                        detail.append(dict(indexed, cell_width=width, rule=rule, cells=np.mean(cells),
                                           projected_volume=np.mean(cells) * width**cfg['projection_dimension']))
                        if epoch == run['epochs'][-1]:
                            raw.extend(dict(indexed, cell_width=width, rule=rule, projection=j,
                                            cells=n, projected_volume=n*width**cfg['projection_dimension'])
                                       for j, n in enumerate(cells))

    frame = pd.DataFrame(detail)
    frame.to_csv(output/'cumulative_coverage_by_seed.csv', index=False)
    pd.DataFrame(raw).to_csv(output/'final_by_projection.csv', index=False)
    pd.DataFrame(outcomes).to_csv(output/'best_objective_by_seed.csv', index=False)
    final = frame.loc[frame.epoch.eq(frame.epoch.max())]
    keys = ['function', 'cell_width', 'rule']
    summary = final.groupby(keys, sort=False).agg(mean_cells=('cells', 'mean'), max_cells=('cells', 'max'),
        mean_projected_volume=('projected_volume', 'mean'), max_seed_projected_volume=('projected_volume', 'max')).reset_index()
    summary.to_csv(output/'final_coverage_summary.csv', index=False)
    suite = summary.groupby(['cell_width','rule'], sort=False).mean(numeric_only=True).reset_index()
    suite.to_csv(output/'fixed_suite_means.csv', index=False)
    width = cfg['reference_cell_width']
    tests = contrasts(final.loc[final.cell_width.eq(width)], ['projected_volume'], RULES[1:], source)
    tests.to_csv(output/'paired_reference_width.csv', index=False)

    matplotlib.rcParams.update({'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.1), layout='constrained')
    for ax, function in zip(axes.flat, source['functions'], strict=True):
        f = frame.loc[frame.function.eq(function) & frame.cell_width.eq(width)]
        for rule in RULES:
            trajectory = f.loc[f.rule.eq(rule)].pivot(index='epoch', columns='seed', values='projected_volume')
            lo, hi = np.asarray([interval(row, source) for row in trajectory.to_numpy()]).T
            ax.plot(trajectory.index, trajectory.mean(axis=1), label=LABELS[rule], color=COLORS[rule])
            ax.fill_between(trajectory.index, lo, hi, alpha=.10, color=COLORS[rule], linewidth=0)
        ax.set_xscale('log')
        ax.set_title(plan['display_names'][function], loc='left', fontweight='bold')
        ax.grid(axis='y', alpha=.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=4, frameon=False)
    fig.suptitle(f'Cumulative search coverage at {source["dimension"]}D', fontweight='bold')
    fig.supylabel('Mean occupied volume in 3D projections')
    for ax in axes[-1]:
        ax.set_xlabel('Swarm update (log scale)')
    fig.savefig(output/'cumulative_coverage.pdf')
    fig.savefig(output/'cumulative_coverage.png', dpi=180)
    plt.close(fig)

    base = suite.loc[suite.cell_width.eq(width)].set_index('rule')
    winner = base.mean_projected_volume.idxmax()
    maximum = final.loc[final.cell_width.eq(width)].sort_values('projected_volume', ascending=False).iloc[0]
    print(suite[['cell_width','rule','mean_cells','mean_projected_volume']].to_string(index=False))
    print(tests[['function','comparator','mean_worse_minus_comparator','p_holm']].to_string(index=False))
    print('Maximum single run:', maximum.to_dict())
    print(f'{evaluations:,} objective evaluations')
    (output/'README.md').write_text(
        '# Cumulative search coverage at 1000D\n\n'
        'All four rules operated throughout each full HPPSO run. At each update every eligible particle '
        'draws one other peer uniformly. Worse and Better gate on the sampled peer’s strict current-objective '
        'comparison. Random always uses the sampled peer. No attraction omits only the openness term. '
        'The three attraction rules use the peer’s personal best. Curiosity, conformity, aggressiveness, '
        'and personality refreshing remain active. The arms use common initial states and random streams. '
        'The worse-peer arm reproduced the saved final objectives for all four functions and ten seeds.\n\n'
        f'Coverage counts every evaluated swarm position from update 1 through update {int(frame.epoch.max())}. '
        'Initialization counts toward the evaluation budget. Coordinates are centered and normalized by their search ranges. '
        f'We use {cfg["projections"]} fixed orthonormal {cfg["projection_dimension"]}D random projections and fixed grid origins '
        'shared by every run. At width w, one projection’s volume estimate is the number of unique occupied '
        'cells times w³. Each run averages those estimates over the projections. The sets are updated '
        'in batches and store only occupied cells. The projections are repeated measurements within a run. '
        'The plotted intervals bootstrap matched seeds, and the volume represents projected coverage.\n\n'
        f'At the reference width of {width:g}, {LABELS[winner]} has the largest mean projected coverage '
        f'({base.loc[winner,"mean_projected_volume"]:.6g}), giving each function equal weight. '
        f'The maximum single-run projection-averaged volume is {maximum.projected_volume:.6g} '
        f'({LABELS[maximum.rule]}, {plan["display_names"][maximum.function]}, seed {maximum.seed}). '
        'Compare rules within each resolution.\n\n'
        f'The experiment used {evaluations:,} objective evaluations. The coverage calculation itself '
        'uses evaluated positions and adds zero objective evaluations.\n\n'
        'The per-function tests compare Worse with each of the other three arms at the reference width using paired '
        'sign flips and Holm correction across the 12 contrasts. The better and random arms can activate '
        'at different rates under these one-shot rules, so the results compare the complete rules.\n\n'
        '## Reproduce\n\n```powershell\n$env:PYTHONPATH = "$PWD\\src"\n'
        'python -m experiments.openness_mechanism.full_run_coverage\n```\n', encoding='utf-8')
