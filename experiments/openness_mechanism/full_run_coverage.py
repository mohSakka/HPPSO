"""Cumulative projected coverage of the full HPPSO search with four peer rules."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from experiments.openness_mechanism.run import RULES, BatchedTraitPSO, BatchBudget, BatchFunction, trait_suite
from experiments.final_benchmark import read_result, write_result, shard_lease
from experiments.trait_study import fingerprint, result_path
import numpy as np


def projections(dimension, config):
    rng = np.random.default_rng(config['projection_seed'])
    blocks = [np.linalg.qr(rng.standard_normal((dimension, config['projection_dimension'])))[0]
              for _ in range(config['projections'])]
    offsets = rng.random((config['projections'], config['projection_dimension']))
    return np.concatenate(blocks, axis=1), offsets


class Coverage:
    def __init__(self, lower, upper, config):
        self.lower = lower
        self.width = upper - lower
        if np.any(self.width <= 0):
            raise ValueError('Coverage study requires positive coordinate widths')
        self.matrix, self.offsets = projections(len(lower), config)
        self.scales = config['cell_widths']
        self.cells = [[[set() for _ in self.offsets] for _ in RULES] for _ in self.scales]
        # The projected centered box is within these symmetric coordinate limits.
        radius = np.abs(self.matrix).sum(axis=0).max() / 2
        self.shift = int(np.ceil(radius / min(self.scales))) + 2
        self.base = 2 * self.shift + 1
        self.powers = self.base ** np.arange(config['projection_dimension'], dtype=np.int64)
        if self.base ** config['projection_dimension'] > np.iinfo(np.int64).max:
            raise ValueError('Projected cell indices do not fit into an integer key')
        self.buffer = []

    def observe(self, positions):
        projected = (positions - self.lower) @ (self.matrix / self.width[:, None])
        projected -= .5 * self.matrix.sum(axis=0)
        self.buffer.append(projected)

    def checkpoint(self):
        points = np.concatenate(self.buffer, axis=1)
        points = points.reshape(*points.shape[:2], *self.offsets.shape)
        self.buffer.clear()
        for width, branches in zip(self.scales, self.cells, strict=True):
            bins = np.floor(points / width + self.offsets).astype(np.int64) + self.shift
            assert bins.min() >= 0 and bins.max() < self.base
            keys = bins @ self.powers
            for branch, row in zip(branches, keys, strict=True):
                for occupied, indices in zip(branch, row.T, strict=True):
                    occupied.update(np.unique(indices).tolist())
        return [[[len(occupied) for occupied in branch] for branch in scale] for scale in self.cells]


def trial(job, output, source, config, settings, spec, digest):
    path = result_path(output, job)
    if path.exists():
        if read_result(path)['fingerprint'] != digest:
            raise ValueError(f'Mismatched saved trial: {path}')
        return job
    problem = trait_suite(source['dimension'], settings['instance_seed'])[job['function']]
    tracker = BatchBudget(BatchFunction(problem), source['budget'], len(RULES), source['dimension'], settings['trace_points'])
    specs = [spec, dict(spec, peer_rule='better'), dict(spec, peer_rule='any'),
             dict(spec, traits=[trait for trait in spec['traits'] if trait != 'openness'])]
    engine = BatchedTraitPSO(tracker, problem['bounds'], source['population'], job['seed'], specs, settings)
    assert np.all(engine.positions == engine.positions[0]) and np.all(engine.velocities == 0)
    coverage = Coverage(engine.lower, engine.upper, config)
    final_update = source['budget'] // source['population'] - 1
    checkpoints, counts, best = [], [], []
    for epoch in range(1, final_update + 1):
        engine.step()
        coverage.observe(engine.positions)
        if epoch in (1, 2, 5, 10, final_update) or epoch % config['checkpoint_every'] == 0:
            checkpoints.append(epoch)
            counts.append(coverage.checkpoint())
            best.append(tracker.best.tolist())
    assert tracker.count == source['budget']
    assert np.all(np.diff(counts, axis=0) >= 0)
    assert np.all(np.diff(best, axis=0) <= 0)
    prior = read_result(result_path(Path(__file__).with_name('results_1000d'), job))
    # The original worse-peer lane must reproduce the earlier 1000D full run.
    np.testing.assert_allclose(tracker.best[0], prior['original_final_best'], rtol=1e-12, atol=1e-12)
    write_result(path, dict(job, fingerprint=digest, epochs=checkpoints, counts=counts, best_objectives=best,
                            evaluations=tracker.count * len(RULES), original_worse_verified=True))
    return job


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=base/'full_run_coverage_1000d')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.report_only:
        from experiments.openness_mechanism.full_run_coverage_report import report
        report(output)
        return
    prior = json.loads((base/'results_1000d'/'plan.json').read_text())
    source, settings = prior['config'], prior['settings']
    config = json.loads((base/'full_run_coverage_config.json').read_text())
    ref = json.loads((Path(source['reference'])/'plan.json').read_text())
    spec = ref['protocol']['variants'][ref['protocol']['factorial']['15']]['spec']
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256((json.dumps(config, sort_keys=True) + prior['fingerprint'] + fingerprint(settings)).encode()
                            + Path(__file__).read_bytes()).hexdigest()
    manifest = dict(config=config, source=source, display_names=settings['function_display_names'], fingerprint=digest)
    with shard_lease(output, 'full_run_coverage'):
        path = output/'plan.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('Output already contains a different experiment')
        path.write_text(json.dumps(manifest, indent=2))
        jobs = [dict(function=function, seed=seed) for function in source['functions'] for seed in source['seeds']]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            tasks = [pool.submit(trial, job, output, source, config, settings, spec, digest) for job in jobs]
            for i, future in enumerate(as_completed(tasks), 1):
                job = future.result()
                print(f'[{i}/{len(jobs)}] {job["function"]} seed={job["seed"]}', flush=True)
        from experiments.openness_mechanism.full_run_coverage_report import report
        report(output)


if __name__ == '__main__':
    main()
