"""Replay saved diversity trials and measure instantaneous behavioral diversity."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import html
import json
from pathlib import Path

from experiments.diversity_study import (
    BatchBudget, BatchFunction, BatchedTraitPSO, interval, jobs, np, protocol,
    read_result, result_path, trait_suite, write_result,
)


def replay(source, output, cfg, job):
    old = read_result(result_path(source, job))
    problem = trait_suite(cfg['dimension'], cfg['instance_seed'])[job['function']]
    keys, specs = protocol(cfg)
    assert keys == old['variant_ids']
    tracker = BatchBudget(BatchFunction(problem), cfg['budget'], len(keys),
                          cfg['dimension'], cfg['trace_points'])
    engine = BatchedTraitPSO(tracker, problem['bounds'], cfg['population'],
                             job['seed'], specs, cfg)
    observed = []

    def observe(inertia, cognitive, social, peer, mutation):
        magnitudes = np.stack([np.linalg.norm(term / engine.width, axis=-1)
                               for term in (inertia, cognitive, social, peer)])
        # Exact ties between nonzero terms use the fixed order above.
        dominant = np.where(magnitudes.max(axis=0) == 0, 4, magnitudes.argmax(axis=0))
        categories = 2 * dominant + mutation.any(axis=-1)
        observed.append([np.bincount(row, minlength=10).tolist() for row in categories])

    targets = set(old['evaluations'][1:])
    checkpoints = [tracker.count]
    # There is no behavior observation at initialization.
    counts = [None]
    while tracker.count < tracker.budget:
        next_count = min(tracker.count + cfg['population'], tracker.budget)
        engine.step(observe if next_count in targets else None)
        if next_count in targets:
            counts.append(observed.pop())
            checkpoints.append(tracker.count)
    assert checkpoints == old['evaluations']
    np.testing.assert_allclose(tracker.best, old['best_errors'][-1], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(tracker.best_positions, old['best_positions'], rtol=1e-12, atol=1e-10)
    write_result(result_path(output, job), dict(job, evaluations=checkpoints,
                 labels=old['labels'], counts=counts))
    return job


def indices(counts, population):
    assert np.all(counts.sum(axis=-1) == population)
    p = counts / population
    log_p = np.log(p, out=np.zeros_like(p), where=p > 0)
    return -np.sum(p * log_p, axis=-1), 1 - np.sum(p**2, axis=-1)


def report(output, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    bootstrap = np.random.default_rng(20261107).integers(0, len(cfg['seeds']), (2000, len(cfg['seeds'])))
    figures = output / 'figures'
    early = figures / 'first_100_iterations'
    early.mkdir(parents=True, exist_ok=True)
    rows = []
    with PdfPages(figures / 'behavioral_diversity.pdf') as full_pdf, PdfPages(early / 'all_functions.pdf') as early_pdf:
        for function in cfg['functions']:
            records = [read_result(result_path(output, dict(function=function, seed=seed)))
                       for seed in cfg['seeds']]
            evaluations = records[0]['evaluations'][1:]
            assert all(r['evaluations'][1:] == evaluations for r in records)
            iterations = np.asarray(evaluations) // cfg['population'] - 1
            counts = np.asarray([r['counts'][1:] for r in records])
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
            for ax, name, data, maximum in zip(
                axes, ('Shannon H', 'Simpson 1 − D'), indices(counts, cfg['population']), (np.log(10), .9), strict=True
            ):
                for i, label in enumerate(records[0]['labels']):
                    y = data[:, :, i]
                    med = np.median(y, axis=0)
                    lo, hi = interval(y, bootstrap)
                    line, = ax.plot(iterations, med, label=label)
                    ax.fill_between(iterations, lo, hi, alpha=.16, color=line.get_color())
                    rows.extend(dict(function=function, algorithm=label, metric=name, iteration=t,
                                     median=m, ci95_low=l, ci95_high=h)
                                for t, m, l, h in zip(iterations, med, lo, hi, strict=True))
                ax.set(title=name, xlabel='Swarm iteration (0 = initialization)', ylim=(0, maximum * 1.03))
                ax.set_xscale('symlog', linthresh=1)
                ax.set_xlim(0, iterations[-1])
                ax.grid(alpha=.25)
            axes[0].legend()
            title = cfg['function_display_names'][function]
            fig.suptitle(f'{title} · {cfg["dimension"]}D · {len(records)} matched seeds\n'
                         'Median and 95% seed-bootstrap intervals')
            fig.savefig(figures / f'{function}.png', dpi=160)
            full_pdf.savefig(fig)
            for ax in axes:
                ax.set_xlim(0, 100)
            fig.savefig(early / f'{function}.png', dpi=160)
            early_pdf.savefig(fig)
            plt.close(fig)
    with (output / 'trajectory_summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gallery = ''.join(
        f'<h2>{html.escape(cfg["function_display_names"][f])}</h2>'
        f'<img style="width:100%;max-width:1200px" src="figures/{f}.png" loading="lazy">'
        for f in cfg['functions']
    )
    (output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><h1>Behavioral diversity</h1>'
        '<p>At each sampled update, every particle belongs to one of ten categories. '
        'The category combines its largest velocity contribution by normalized Euclidean norm '
        '(inertia, cognitive, social, peer, or all zero) with whether at least one coordinate '
        'received a mutation. Contributions are measured before clipping and reflection. '
        'There is no observation at initialization. Indices use the per-update swarm frequencies. '
        'The index axes are linear to include zero.</p>'
        '<p><a href="figures/behavioral_diversity.pdf">Full-run figures</a> · '
        '<a href="figures/first_100_iterations/all_functions.pdf">First 100 iterations</a> · '
        '<a href="trajectory_summary.csv">Trajectory data</a></p>' + gallery, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    cfg = json.loads((args.source / 'plan.json').read_text())['config']
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(replay, args.source, args.output, cfg, job) for job in jobs(cfg)]
        for i, future in enumerate(futures, 1):
            print(f'[{i}/{len(futures)}] {future.result()}', flush=True)
    report(args.output, cfg)


if __name__ == '__main__':
    main()
