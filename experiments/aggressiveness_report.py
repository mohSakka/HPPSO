"""Create combined boxplots and Wilcoxon tables for isolated trait studies."""

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from experiments.conformity_wilcoxon import report as wilcoxon_report


def report(output):
    plan = json.loads((output / 'plan.json').read_text())
    config = plan['config']
    factorial = config.get('factorial_pso_control', False)
    group = 'hp_ablation' if factorial else config['groups'][0]
    refresh = group == 'personality_refresh'
    peer_rules = group == 'openness_peer_rules'
    hp_peer_rules = group == 'hppso_openness_peer_rules'
    study = ('HPPSO personality reinitialization' if refresh else 'HPPSO' if factorial
             else 'HPPSO openness peer rules' if hp_peer_rules
             else 'Openness peer rules' if peer_rules else group.capitalize())
    variant_labels = {entry['variant']: entry['label'] for entry in plan['protocol']['groups'][group]}
    algorithms = list(variant_labels.values())
    wilcoxon_report(output, algorithms, study, variant_labels=variant_labels)
    runs = pd.read_csv(output / 'runs.csv')
    runs = runs.loc[runs.variant.isin(variant_labels)].assign(label=lambda frame: frame.variant.map(variant_labels))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(algorithms)]
    detailed_labels = group in ('curiosity', 'openness', 'openness_peer_rules', 'hppso_openness_peer_rules')
    figsize = (15, 13) if factorial else (10.4, 12.3) if detailed_labels else (8.2, 10.7)
    fig, axes = plt.subplots(5, 2, figsize=figsize, layout='constrained')
    title = f'{study} ablation at {config["dimension"]} dimensions'
    if detailed_labels or factorial:
        title += (f'\n{config["population"]} particles · {config["budget"]:,} evaluations per run '
                  f'· {len(config["seeds"])} matched seeds')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    tick_labels = [label.replace('-individual', '\n(individual)').replace('-shared', '\n(shared)')
                   for label in algorithms] if detailed_labels else algorithms
    if factorial:
        tick_labels = [label.replace('HPPSO without ', 'Without\n') for label in algorithms]
    if refresh:
        tick_labels = ['HPPSO', 'HPPSO without personality\nreinitialization']
    if peer_rules:
        tick_labels[-1] = 'PSO-openness-\ncurrent'
    if hp_peer_rules:
        tick_labels = ['HPPSO', 'HPPSO-openness\nbetter', 'HPPSO-openness\nrandom',
                       'HPPSO-worse\ncurrent', 'HPPSO without\nopenness']
    for ax, function in zip(axes.flat, config['functions'], strict=True):
        scores = runs.loc[runs.function.eq(function)].pivot(index='seed', columns='label', values='best_score')
        values = scores[algorithms].to_numpy()
        boxes = ax.boxplot(values, patch_artist=True, widths=.50, showfliers=False,
                           medianprops={'color': '#161e25', 'linewidth': 1.5})
        for i, (box, color) in enumerate(zip(boxes['boxes'], colors, strict=True)):
            box.set(facecolor=color, edgecolor=color, alpha=.25)
            offsets = np.random.default_rng(100 + i).permutation(np.linspace(-.14, .14, len(values)))
            ax.scatter(i + 1 + offsets, values[:, i], color=color, s=11, alpha=.82, zorder=3)
        ax.set_title(config['function_display_names'][function], loc='left', fontweight='bold', fontsize=10, pad=8)
        ax.set_xticks(range(1, len(algorithms) + 1), tick_labels, fontsize=8)
        if detailed_labels or factorial:
            ax.tick_params(axis='x', length=0, pad=6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e1e6ec', linewidth=.7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(formatter)
    fig.supylabel('Final best objective (lower is better)', fontsize=11)
    caption = f'Each dot is one matched seed. {config["population"]} particles and {config["budget"]:,} evaluations per run.'
    if group == 'curiosity':
        shared = 'one draw per iteration for the swarm' if config.get('curiosity_shared_per_iteration') else 'one fixed value for the swarm'
        caption = (f'Each dot is one seed. Both curiosity variants use η = {config["eta"]:g} '
                   f'and αₘᵤₜ = {config["alpha_mut"]:g}.\n'
                   f'Individual: one fixed draw per particle. Shared: {shared}.')
    elif group == 'openness':
        caption = (f'Each dot is one seed. Both openness variants use θ = {config["theta"]:g} '
                   f'and λ = {config["lam"]:g}.\n'
                   'Individual: fixed per-particle openness. Shared: common activation probability per particle.')
    elif peer_rules:
        caption = (f'Each dot is one matched seed. Openness uses θ = {config["theta"]:g} and λ = {config["lam"]:g}.\n'
                   'Original: worse peer’s personal best. Better and any: peer’s personal best. Current: worse peer’s current position.')
    elif hp_peer_rules:
        caption += '\nHPPSO: worse → personal best. Better and random → personal best. Worse-current → current position.'
    elif factorial:
        caption += '\n“Without” denotes HPPSO with the indicated trait disabled.'
    elif refresh:
        caption += f'\nFull HPPSO refreshes traits after {config["stagnation_threshold"]} stagnant iterations.'
    fig.supxlabel(caption, fontsize=9)
    for extension in ('pdf', 'png'):
        fig.savefig(output / 'figures' / f'final_objective_boxplots.{extension}', dpi=260, facecolor='white')
    plt.close(fig)

    index = output / 'index.html'
    page = index.read_text(encoding='utf-8')
    links = ('<p><a href="figures/final_objective_boxplots.pdf">Boxplots (PDF)</a> · '
             '<a href="wilcoxon/index.html">Per-function Wilcoxon ranks</a></p>')
    if group == 'curiosity':
        shared_description = ('The shared variant draws one curiosity value from U(0, 1) per iteration for the whole swarm. '
                              if config.get('curiosity_shared_per_iteration') else
                              f'The shared variant uses curiosity {config["shared_level"]:g} for every particle. ')
        links += (f'<p>Both curiosity variants use η = {config["eta"]:g} and α = {config["alpha_mut"]:g}. '
                  + shared_description +
                  'The individual variant assigns each particle a fixed draw from U(0, 1). '
                  'The PSO control comes from the matched aggressiveness study.</p>')
    elif group == 'openness':
        links += (f'<p>Both openness variants use θ = {config["theta"]:g} and λ = {config["lam"]:g}. '
                  'The individual variant draws one fixed openness level per particle. The shared variant '
                  'uses a common probability 1 − θ, with independent activation draws per particle each iteration. '
                  'An activated particle samples one random peer and moves toward its personal best only '
                  'if the peer’s current objective is worse. The PSO control comes from the matched aggressiveness study.</p>')
    elif peer_rules:
        links += ('<p>Every openness arm uses fixed individual openness traits and samples one peer uniformly '
                  'from the other particles. Better and worse refer to the current objective. '
                  'PSO-openness-individual uses a worse peer’s personal best. PSO-open-better uses a better peer’s '
                  'personal best. PSO-open-any uses the sampled peer’s personal best regardless of its objective. '
                  'PSO-openness-current uses a worse peer’s current position. The original openness and PSO '
                  'results come from saved matched runs.</p>')
    elif hp_peer_rules:
        links += ('<p>Full HPPSO and HPPSO without openness use saved matched results. Each of the three new variants '
                  'changes only the openness rule. Peer quality is assessed at the current position. Random uses '
                  'the sampled peer regardless of quality. All other traits and personality refreshing stay enabled.</p>')
    elif factorial:
        links += factorial_summary(output, config)
    elif refresh:
        links += (f'<p>The saved HPPSO control refreshes a particle’s traits after '
                  f'{config["stagnation_threshold"]} iterations without personal-best improvement. '
                  'The comparison keeps each particle’s initial traits fixed throughout the run. '
                  'Both variants use all four traits and matched seeds.</p>')
    else:
        links += ('<p>PSO uses fixed inertia 0.65. PSO-aggressiveness assigns fixed per-particle '
                  'inertia in [0.4, 0.9]. PSO-RIW draws shared inertia in [0.4, 0.9] each iteration.</p>')
    anchor = '<p><a href="figures/convergence.pdf">'
    end = page.index(anchor)
    start = page.find('<p><a href="figures/final_objective_boxplots.pdf">', 0, end)
    if start == -1:
        start = end
    index.write_text(page[:start] + links + page[end:], encoding='utf-8')


def factorial_summary(output, config):
    effects = pd.read_csv(output / 'factorial_effects.csv')
    tests = pd.read_csv(output / 'wilcoxon' / 'pairwise_tests.csv')
    # The seven comparisons against full HPPSO form a focused family.
    focused = tests.loc[tests.algorithm_a.eq('HPPSO')].sort_values(['function', 'p_raw']).copy()
    for _, group in focused.groupby('function', sort=False):
        factors = np.arange(len(group), 0, -1)
        focused.loc[group.index, 'p_holm'] = np.maximum.accumulate(np.minimum(1, factors * group.p_raw))
    focused['significant_winner'] = np.where(
        focused.p_holm.le(.05),
        np.where(focused.negative_rank_sum.gt(focused.positive_rank_sum), focused.algorithm_a, focused.algorithm_b),
        '',
    )
    focused.to_csv(output / 'wilcoxon' / 'hppso_reference_tests.csv', index=False)
    higher = effects.loc[effects.order.ge(3)]
    aggregate = higher.loc[higher.function.eq('aggregate_fixed_functions')]
    fig, ax = plt.subplots(figsize=(8.5, 4.4), layout='constrained')
    y = np.arange(len(aggregate))
    ax.hlines(y, aggregate.ci95_low, aggregate.ci95_high, color='#1f77b4', lw=2)
    ax.plot(aggregate.mean_contrast, y, 'o', color='#1f77b4')
    ax.axvline(0, color='#777777', lw=1)
    ax.set_yticks(y, [textwrap.fill(label.title(), 42) for label in aggregate.traits])
    ax.invert_yaxis()
    ax.set(title='Three- and four-trait interactions across the 10 functions',
           xlabel='Mean contrast in relative objective change (95% seed bootstrap interval)')
    file = output / 'figures' / 'interactions' / 'higher_order'
    fig.savefig(file.with_suffix('.pdf'))
    fig.savefig(file.with_suffix('.png'), dpi=200)
    plt.close(fig)
    return ('<p><a href="factorial_effects.csv">All factorial effects</a> · '
            '<a href="pair_interactions_by_context.csv">Pair interactions in each background</a> · '
            '<a href="figures/interactions/aggregate.pdf">Mean pair interaction matrix (PDF)</a> · '
            '<a href="figures/interactions/higher_order.pdf">Three- and four-trait interactions (PDF)</a></p>'
            '<p><a href="wilcoxon/hppso_reference_tests.csv">Seven comparisons against full HPPSO per function</a> '
            'use a separate Holm correction. For the 28 all-pairs comparisons, the smallest possible '
            'exact two-sided p-value with 10 seeds is 0.001953, above the first Holm threshold of 0.001786.</p>'
            '<p>The 16 trait combinations use matched seeds. The zero-trait condition is PSO. '
            'The response is (final − initial)/|initial|. Pair, triple, and four-way estimates are '
            'factorial contrasts averaged over the states of the remaining traits. Intervals resample '
            'matched seed blocks and describe uncertainty for this benchmark set.</p>'
            f'<p>Active traits refresh after {config["stagnation_threshold"]} stagnant iterations. '
            'PSO, PSO-m, and PSO-RIW use saved matched runs.</p>')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    report(parser.parse_args().output)
