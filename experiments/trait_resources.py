"""Measure each isolated algorithm in a fresh process under the factorial budget."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.diversity_study import LABELS, protocol
from experiments.final_benchmark import shard_lease, read_result, write_result
from experiments.trait_batch import BatchBudget, BatchFunction, BatchedTraitPSO
from experiments.trait_functions import trait_suite
from experiments.trait_study import jobs, result_path
from experiments.trait_interactions import bootstrap_ci
import numpy as np


def measure(reference, function, seed, label):
    import psutil

    cfg = json.loads((reference / 'plan.json').read_text())['config']
    ids, specs = protocol(cfg)
    i = LABELS.index(label)
    problem = trait_suite(cfg['dimension'], cfg['instance_seed'])[function]
    saved = read_result(result_path(reference, dict(function=function, seed=seed)))
    column = saved['variant_ids'].index(ids[i])
    process = psutil.Process()
    before = process.memory_info()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    tracker = BatchBudget(BatchFunction(problem), cfg['budget'], 1, cfg['dimension'], cfg['trace_points'])
    engine = BatchedTraitPSO(tracker, problem['bounds'], cfg['population'], seed, [specs[i]], cfg)
    engine.optimize()
    cpu_seconds, wall_seconds = time.process_time() - cpu_start, time.perf_counter() - wall_start
    after = process.memory_info()
    assert tracker.count == cfg['budget']
    np.testing.assert_allclose(tracker.best[0], saved['best_scores'][column], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(tracker.best_positions[0], saved['best_positions'][column], rtol=1e-12, atol=1e-10)
    result = dict(function=function, seed=seed, algorithm=label, evaluations=tracker.count,
                  best_score=float(tracker.best[0]), wall_seconds=wall_seconds, cpu_seconds=cpu_seconds,
                  baseline_rss_bytes=before.rss, peak_rss_bytes=after.peak_wset,
                  pretrial_peak_rss_bytes=before.peak_wset,
                  peak_above_baseline_bytes=after.peak_wset - before.rss,
                  reference_fingerprint=saved['fingerprint'])
    print(json.dumps(result))


def report(output, reference, records):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plan = json.loads((reference / 'plan.json').read_text())
    cfg = plan['config']
    runs = pd.DataFrame(records)
    for metric in ('peak_rss', 'peak_above_baseline'):
        runs[metric + '_mib'] = runs[metric + '_bytes'] / 2**20
    metrics = ('wall_seconds', 'cpu_seconds', 'peak_rss_mib', 'peak_above_baseline_mib')
    runs.to_csv(output / 'runs.csv', index=False)
    medians = runs.groupby(['function', 'algorithm'])[list(metrics)].median().unstack('algorithm')
    medians.to_csv(output / 'function_medians.csv')
    paired = []
    draw = np.random.default_rng(2026).integers(0, len(cfg['seeds']), (2000, len(cfg['seeds'])))
    for metric in metrics:
        grid = runs.pivot(index=['function', 'seed'], columns='algorithm', values=metric)
        for label in LABELS[1:]:
            relative = grid.HPPSO - grid[label]
            blocks = relative.unstack('seed').to_numpy()
            by_seed = np.mean(blocks, axis=0)
            lo, hi = bootstrap_ci(by_seed[:, None], draw, 'mean')[:, 0]
            paired.append(dict(metric=metric, comparator=label, mean_hp_minus_comparator=by_seed.mean(),
                               ci95_low=lo, ci95_high=hi))
    pd.DataFrame(paired).to_csv(output / 'paired_differences.csv', index=False)
    overall = runs.groupby('algorithm')[list(metrics)].median().reindex(LABELS)
    overall.to_csv(output / 'overall_medians.csv')

    functions = cfg['functions']
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout='constrained')
    x = np.arange(len(functions))
    for ax, metric, ylabel in zip(axes, ('wall_seconds', 'peak_rss_mib'), ('Wall time (s)', 'Peak working set (MiB)')):
        for i, label in enumerate(LABELS):
            ax.bar(x + (i - 1.5) * .19, medians[metric][label].reindex(functions), .18, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(ncols=4)
    axes[-1].set_xticks(x, [cfg['function_display_names'][f] for f in functions], rotation=30, ha='right')
    fig.suptitle('Runtime and memory by function · median of 10 matched seeds')
    fig.savefig(output / 'resources.pdf')
    fig.savefig(output / 'resources.png', dpi=180)
    plt.close(fig)
    (output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><h1>Computational metrics</h1>'
        '<p>Each run used a fresh process. Trials ran sequentially, with one numerical-library thread. '
        'Wall and CPU time cover optimizer initialization through the final evaluation. '
        'The Windows peak working set is a process high-water mark and includes the Python runtime, '
        'imports, objective workspaces, and convergence recording. The baseline column records the '
        'working set immediately before the timed run.</p>'
        '<p><a href="runs.csv">Per-seed measurements</a> · <a href="function_medians.csv">Function medians</a> · '
        '<a href="paired_differences.csv">Paired differences and bootstrap intervals</a> · '
        '<a href="resources.pdf">Figure PDF</a></p>'
        + overall.to_html(float_format=lambda x: f'{x:.3f}')
        + '<img src="resources.png" style="max-width:100%">', encoding='utf-8')
    print(overall.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', nargs=3, metavar=('FUNCTION', 'SEED', 'ALGORITHM'))
    args = parser.parse_args()
    if args.worker:
        function, seed, label = args.worker
        measure(args.reference, function, int(seed), label)
        return
    plan = json.loads((args.reference / 'plan.json').read_text())
    cfg = plan['config']
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    total = len(jobs(cfg)) * len(LABELS)
    records = []
    with shard_lease(output, 'resources'):
        for i, job in enumerate(jobs(cfg)):
            # Rotate run order between paired trials to distribute thermal and load drift.
            for label in LABELS[i % 4:] + LABELS[:i % 4]:
                function, seed = job['function'], job['seed']
                path = output / 'trials' / function / label / f'seed_{seed:03d}.json.gz'
                if path.exists():
                    row = read_result(path)
                    if row['reference_fingerprint'] != plan['fingerprint']:
                        raise ValueError('Saved resource trial uses a different reference')
                else:
                    process = subprocess.run(
                        [sys.executable, '-m', 'experiments.trait_resources', '--reference', str(args.reference),
                         '--worker', function, str(seed), label], stdout=subprocess.PIPE, text=True, check=True)
                    row = json.loads(process.stdout)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    write_result(path, row)
                records.append(row)
                (output / 'progress.json').write_text(json.dumps(dict(completed=len(records), total=total)))
                print(f'[{len(records)}/{total}] {function} seed={seed} {label}: {row["wall_seconds"]:.2f}s', flush=True)
        report(output, args.reference, records)


if __name__ == '__main__':
    main()
