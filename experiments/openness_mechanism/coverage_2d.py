"""Map cumulative coverage in the original coordinates of 2D benchmark runs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path

from experiments.openness_mechanism.run import RULES, BatchedTraitPSO, BatchBudget, BatchFunction, trait_suite
from hppso.benchmarks.classical import problem_definitions
import numpy as np


def trial(function, seed, settings, spec, population, budget):
    problem = trait_suite(2, settings['instance_seed'])[function]
    tracker = BatchBudget(BatchFunction(problem), budget, len(RULES), 2, settings['trace_points'])
    specs = [spec, dict(spec, peer_rule='better'), dict(spec, peer_rule='any'),
             dict(spec, traits=[t for t in spec['traits'] if t != 'openness'])]
    engine = BatchedTraitPSO(tracker, problem['bounds'], population, seed, specs, settings)
    sizes = (50, 100, 200)
    first_visit = {n: np.zeros((len(RULES), n*n), dtype=np.int16) for n in sizes}
    lanes = np.arange(len(RULES))[:, None]
    for epoch in range(1, budget // population):
        engine.step()
        unit = (engine.positions - engine.lower) / engine.width
        for n, grid in first_visit.items():
            bins = np.clip((unit*n).astype(int), 0, n-1)
            ids = bins[..., 1]*n + bins[..., 0]
            old = grid[lanes, ids]
            grid[lanes, ids] = np.where(old == 0, epoch, old)
    assert tracker.count == budget
    assert all(np.count_nonzero(grid, axis=1).max() <= budget-population for grid in first_visit.values())
    return function, seed, first_visit, tracker.best


def report(output, results, functions, seeds, names):
    rows, maps = [], {}
    for function, seed, grids, best in results:
        for size, grid in grids.items():
            maps[f'{function}_{seed:03d}_{size}'] = grid.reshape(len(RULES), size, size)
            for i, rule in enumerate(RULES):
                occupied = np.count_nonzero(grid[i])
                rows.append(dict(function=function, seed=seed, rule=rule, grid_size=size,
                                 cells=occupied, area_fraction=occupied/size**2, final_best=best[i]))
    np.savez_compressed(output/'first_visit_maps.npz', **maps)
    with (output/'coverage_by_seed.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_maps(output, maps, functions, seeds, names)
    for size in (50, 100, 200):
        print(size, [(rule, np.mean([r['area_fraction'] for r in rows if r['rule'] == rule and r['grid_size'] == size]))
                     for rule in RULES], flush=True)
    for function in functions:
        print(function, [(rule, np.mean([r['area_fraction'] for r in rows if r['rule'] == rule and r['grid_size'] == 100
                                       and r['function'] == function])) for rule in RULES], flush=True)


def plot_maps(output, maps, functions, seeds, names, columns=(0, 1, 2, 3)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    labels = ('Worse peer', 'Better peer', 'Random peer', 'Openness off')
    for example in (False, True):
        fig, axes = plt.subplots(len(functions), len(columns), figsize=(2.5*len(columns), 10),
                                 layout='constrained', sharex=True, sharey=True, squeeze=False)
        images = []
        for i, function in enumerate(functions):
            grid = np.stack([maps[f'{function}_{seed:03d}_100'] for seed in seeds])
            occupancy = (grid > 0).mean(axis=(2, 3))
            for position, (j, ax) in enumerate(zip(columns, axes[i], strict=True)):
                if example:
                    first = grid[0, j]
                    layer = np.where(first == 0, 0, np.where(first <= 10, 1, np.where(first <= 100, 2, 3)))
                    images.append(ax.imshow(layer, origin='lower', extent=(0,1,0,1), vmin=0, vmax=3,
                                            cmap=ListedColormap(['#f5f7fa', '#147d78', '#e9a041', '#8056a0']), interpolation='nearest'))
                else:
                    images.append(ax.imshow((grid[:, j] > 0).mean(axis=0), origin='lower', extent=(0,1,0,1),
                                            vmin=0, vmax=1, cmap='Blues', interpolation='nearest'))
                covered = occupancy[0, j] if example else occupancy[:, j].mean()
                ax.set_title(f'{labels[j]} · {covered:.1%}', fontsize=9)
                ax.set_xticks([0,.5,1]); ax.set_yticks([0,.5,1])
                if position == 0:
                    ax.set_ylabel(names[function], fontsize=9)
        fig.supxlabel('Normalized search coordinates')
        if example:
            ticks = [0, 1, 2, 3]
            bar = fig.colorbar(images[0], ax=axes, shrink=.65, ticks=ticks)
            bar.ax.set_yticklabels(['Unvisited', 'Updates 1–10', 'Updates 11–100', 'Updates 101–1199'])
            title, filename = 'When each area was first visited · Seed 0', 'coverage_seed0'
        else:
            fig.colorbar(images[0], ax=axes, shrink=.65, label='Fraction of seeds visiting the cell')
            title, filename = 'Cumulative search coverage · 10 matched seeds', 'coverage_across_seeds'
        fig.suptitle(title, fontweight='bold')
        fig.savefig(output/f'{filename}.pdf')
        fig.savefig(output/f'{filename}.png', dpi=180)
        plt.close(fig)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=base/'full_run_coverage_2d')
    parser.add_argument('--saved-maps', type=Path, help='Plot saved 2D results without the random-peer column')
    args = parser.parse_args()
    if args.saved_maps:
        plan = json.loads((args.saved_maps/'plan.json').read_text())
        columns = tuple(i for i, rule in enumerate(plan['rules']) if rule != 'Random')
        if len(columns) != 3:
            raise ValueError('Expected three rules after excluding Random')
        args.output.mkdir(parents=True, exist_ok=True)
        with np.load(args.saved_maps/'first_visit_maps.npz') as maps:
            plot_maps(args.output, maps, plan['functions'], plan['seeds'],
                      plan['settings']['function_display_names'], columns)
        return
    source = json.loads((base/'config_1000d.json').read_text())
    reference = json.loads((Path(source['reference'])/'plan.json').read_text())
    settings = dict(reference['config'], dimension=2)
    spec = reference['protocol']['variants'][reference['protocol']['factorial']['15']]['spec']
    functions, seeds = source['functions'], source['seeds']
    args.output.mkdir(parents=True, exist_ok=True)
    suite = trait_suite(2, settings['instance_seed'])
    (args.output/'problem_definitions.json').write_text(json.dumps(problem_definitions([suite[f] for f in functions]), indent=2))
    (args.output/'plan.json').write_text(json.dumps(dict(dimension=2, functions=functions, seeds=seeds,
        population=source['population'], budget=source['budget'], reference_fingerprint=reference['fingerprint'],
        settings=settings, hppso_spec=spec, rules=RULES, grid_sizes=[50,100,200]), indent=2))
    jobs = [(f, s, settings, spec, source['population'], source['budget']) for f in functions for s in seeds]
    results = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(trial, *job) for job in jobs]
        for i, task in enumerate(tasks, 1):
            result = task.result()
            results.append(result)
            print(f'[{i}/{len(tasks)}] {result[0]} seed={result[1]}', flush=True)
    report(args.output, results, functions, seeds, settings['function_display_names'])


if __name__ == '__main__':
    main()
