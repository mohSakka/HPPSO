"""Plot one figure per parameter using the saved one-at-a-time sweep results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


SWEEPS = (
    ('curiosity', 'eta', 'etas', 'α', 'alpha_mut', 'η', True),
    ('curiosity', 'alpha_mut', 'alphas', 'η', 'eta', 'α', True),
    ('openness', 'lam', 'lambdas', 'θ', 'theta', 'λ', False),
    ('openness', 'theta', 'thetas', 'λ', 'lam', 'θ', False),
)


def report(root, only=None):
    chosen = []
    gallery = []
    plt.rcParams.update({'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    for trait, param, levels_key, fixed_symbol, fixed_param, symbol, log_scale in SWEEPS:
        if only and trait != only:
            continue
        folder = root / trait
        plan = json.loads((folder / 'plan.json').read_text())
        cfg = plan['config']
        raw = pd.read_csv(folder / 'runs.csv')
        if (len(raw) != len(cfg['functions']) * len(cfg['seeds']) * len(plan['protocol']['variants'])
                or not raw.status.eq('budget_reached').all() or not raw.evaluations.eq(cfg['budget']).all()):
            raise ValueError(f'Incomplete {trait} sweep')
        summary = pd.read_csv(folder / 'summary.csv')
        params = pd.read_csv(folder / 'parameter_results.csv')
        levels = cfg[levels_key]
        fixed = cfg[fixed_param]
        rows, columns = (1, 3) if trait == 'curiosity' else (5, 2)
        fig, axes = plt.subplots(rows, columns, figsize=(12.5, 4.4 if rows == 1 else 13), layout='constrained')
        for ax, function in zip(axes.flat, cfg['functions'], strict=True):
            data = params.loc[params.function.eq(function) & np.isclose(params[fixed_param], fixed)]
            series = (
                ('individual', f'PSO-{trait} (individual)', '#007A87', 'o'),
                ('shared', f'PSO-{trait} (shared)', '#B35726', 's'),
            )
            if cfg.get('full_hppso_sensitivity'):
                series = (('individual', 'Full HPPSO', '#007A87', 'o'),)
            for mode, label, color, marker in series:
                arm = data.loc[data.personalization.eq(mode) & data[param].isin(levels)].set_index(param).reindex(levels)
                if arm.n.isna().any() or not arm.n.eq(len(cfg['seeds'])).all():
                    raise ValueError(f'Missing {function} results for {mode} {param}')
                ax.plot(levels, arm['median'], marker=marker, color=color, label=label, linewidth=1.6, markersize=4.5)
                ax.fill_between(levels, arm.median_ci95_low, arm.median_ci95_high, color=color, alpha=.15)
                best = arm['median'].idxmin()
                chosen.append(dict(trait=trait, parameter=param, fixed_parameter=fixed_param, fixed_value=fixed,
                                   function=function, variant=label, best_tested_value=best,
                                   median_final_objective=arm.loc[best, 'median']))
            base = summary.loc[summary.function.eq(function) & summary.variant.eq(plan['protocol']['baseline']), 'median'].item()
            baseline_label = 'Default HPPSO median' if cfg.get('full_hppso_sensitivity') else 'PSO median'
            ax.axhline(base, label=baseline_label, color='#515969', ls='--', lw=1)
            if log_scale:
                ax.set_xscale('log')
            ax.set_xticks(levels, [f'{v:g}' for v in levels], fontsize=8)
            ax.set_title(cfg['function_display_names'][function], loc='left', fontweight='bold')
            ax.grid(axis='y', alpha=.23)
            fmt = ScalarFormatter(useMathText=True, useOffset=False)
            fmt.set_powerlimits((-2, 3))
            ax.yaxis.set_major_formatter(fmt)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .91 if rows == 1 else .966),
                   ncols=3, frameon=False, fontsize=9)
        fig.get_layout_engine().set(rect=(0, 0, 1, .84 if rows == 1 else .932))
        prefix = 'Full HPPSO · ' if cfg.get('full_hppso_sensitivity') else ''
        fig.suptitle(f'{prefix}{trait.capitalize()} sensitivity · Vary {symbol} with {fixed_symbol} = {fixed:g}', y=.99,
                     fontsize=14, fontweight='bold')
        fig.supylabel('Final best objective (lower is better)')
        fig.supxlabel(f'{symbol} · Median and pointwise 95% bootstrap intervals across {len(cfg["seeds"])} matched seeds', fontsize=10)
        filename = f'{trait}_{param}'
        fig.savefig(root / f'{filename}.pdf')
        fig.savefig(root / f'{filename}.png', dpi=220, facecolor='white')
        plt.close(fig)
        gallery.append((f'{trait.capitalize()} · {symbol}', filename))

    if only:
        return
    pd.DataFrame(chosen).to_csv(root / 'best_tested_settings.csv', index=False)
    if cfg.get('full_hppso_sensitivity'):
        description = ('Each sweep uses full HPPSO with individualized traits and personality refreshing. '
                       'At θ = 1, only the openness update is disabled. The dashed line shows the saved default HPPSO median.')
    else:
        description = ('Shared curiosity draws one level per iteration for the swarm. Shared openness uses the same '
                       'activation probability for every particle, with independent draws. At θ = 1, openness is disabled.')
    (root / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><h1>Curiosity and openness sensitivity</h1>'
        '<p>100D · 50 particles · 60,000 evaluations · 10 matched seeds. Each panel uses its own objective scale. '
        'The lines show medians and shaded bands show pointwise 95% seed-bootstrap intervals. '
        'The three curiosity functions were selected for their strongest Holm-adjusted evidence '
        'that individualized curiosity improved on PSO in the previous ablation. Openness uses all 10 functions.</p>'
        f'<p>{description}</p>'
        '<p><a href="best_tested_settings.csv">Best tested levels by median</a> · '
        '<a href="curiosity/parameter_results.csv">Curiosity data</a> · <a href="openness/parameter_results.csv">Openness data</a></p>'
        + ''.join(f'<h2>{title}</h2><a href="{name}.pdf">Figure PDF</a><br>'
                  f'<img src="{name}.png" style="width:100%;max-width:1200px" loading="lazy">' for title, name in gallery),
        encoding='utf-8',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--only', choices=('curiosity', 'openness'))
    args = parser.parse_args()
    report(args.root, args.only)
