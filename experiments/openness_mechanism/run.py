"""Replay full HPPSO and branch from saved stagnation states."""

import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from experiments.final_benchmark import read_result, write_result, shard_lease
from experiments.trait_batch import BatchBudget, BatchFunction, BatchedTraitPSO, reflect_bounds
from experiments.trait_functions import trait_suite
from experiments.trait_study import fingerprint, result_path


RULES = ('Worse', 'Better', 'Random', 'None')
STATE_ARRAYS = ('positions', 'velocities', 'scores', 'pbests', 'pbest_scores', 'global_best',
                'global_score', 'traits', 'stagnation', 'reset_events', 'initial_best')


class InterventionHPPSO(BatchedTraitPSO):
    """Study-local copy of the full-trait update with a controllable openness term.

    The replay is verified against every archived HPPSO convergence checkpoint.
    Fixed-size RNG draws keep branches aligned even when their refresh masks differ.
    """

    def update(self, *, original=False, focal=None, forced_peers=None):
        n, d = self.n, self.d
        r1, r2 = self.rng.rand(n, d), self.rng.rand(n, d)
        # Preserve the batched engine's draws even for its disabled PSO-RIW arm.
        self.streams[1].uniform(.4, .9)
        inertia = .4 + .5 * self.traits[:, :, 2]
        cognitive = self.settings['c1'] * r1 * (self.pbests - self.positions)
        social = self.settings['c2'] * r2 * (self.global_best[:, None, :] - self.positions)
        conformity = self.traits[:, :, 1]
        cognitive *= (1 - conformity)[:, :, None]
        social *= conformity[:, :, None]
        self.velocities = inertia[:, :, None] * self.velocities + cognitive + social
        peers = self.streams[2].integers(0, n - 1, n)
        peers += peers >= np.arange(n)
        self.streams[3].random(n)
        if original:
            active = (self.traits[:, :, 3] > self.theta[:, None]) & (self.scores[:, peers] > self.scores)
            self.velocities += np.where(active[:, :, None], self.lam[:, None, None]
                                        * (self.pbests[:, peers, :] - self.positions), 0)
        elif forced_peers is not None:
            for branch, peer in enumerate(forced_peers):
                self.velocities[branch, focal] += self.lam[0] * (
                    self.pbests[branch, peer] - self.positions[branch, focal])
        np.clip(self.velocities, -self.limit, self.limit, out=self.velocities)
        self.positions += self.velocities
        uniforms = self.streams[4].random((n, d))
        noise = self.streams[5].standard_normal((n, d))
        mask = uniforms < self.alpha[0] * self.traits[:, :, 0, None]
        self.positions += np.where(mask, noise * self.eta[0] * self.width, 0)
        reflect_bounds(self.positions.reshape(-1, d), self.velocities.reshape(-1, d), self.lower, self.upper)
        self.scores = self.tracker.evaluate(self.positions)
        improved = self.scores < self.pbest_scores
        self.pbest_scores[improved] = self.scores[improved]
        self.pbests[improved] = self.positions[improved]
        rows = np.arange(self.k)
        indices = self.pbest_scores.argmin(axis=1)
        better = self.pbest_scores[rows, indices] < self.global_score
        self.global_best[better] = self.pbests[rows[better], indices[better]]
        self.global_score = np.minimum(self.global_score, self.pbest_scores.min(axis=1))
        self.stagnation = np.where(improved, 0, self.stagnation + 1)
        reset = self.stagnation >= self.settings['stagnation_threshold']
        replacements = self.streams[6].random((n, 4))
        self.traits = np.where(reset[:, :, None], replacements, self.traits)
        self.stagnation[reset] = 0
        self.reset_events += reset.sum(axis=1)


def candidate_sets(engine):
    scores = engine.scores[0]
    worse = scores[None, :] > scores[:, None]
    better = scores[None, :] < scores[:, None]
    eligible = (engine.traits[0, :, 3] > engine.theta[0]) & worse.any(axis=1) & better.any(axis=1)
    return np.flatnonzero(eligible), worse, better


def memory(engine):
    movable = engine.width > 0
    z = (engine.pbests[0][:, movable] - engine.global_best[0, movable]) / engine.width[movable]
    distance = np.linalg.norm(z, axis=1) / np.sqrt(movable.sum())
    return distance, engine.scores[0] - engine.pbest_scores[0]


def summarize_state(engine, focal, worse, better, stagnant, update):
    distance, retained = memory(engine)
    rows = []
    for rule, mask in zip(RULES, (worse[focal], better[focal], focal[:, None] != np.arange(engine.n))):
        weights = mask / mask.sum(axis=1, keepdims=True)
        rows.append(dict(update=update, stagnant=stagnant, eligible_focals=len(focal), rule=rule,
                         distance=float(np.mean(weights @ distance)),
                         retained_improvement=float(np.mean(weights @ retained))))
    return rows


def snapshot(engine, focal, peers, update, streak):
    return dict(update=update, global_stagnation_updates=streak, evaluations=engine.tracker.count,
                focal=int(focal), peers=[int(p) for p in peers],
                state={key: getattr(engine, key).copy() for key in STATE_ARRAYS},
                rng=engine.rng.get_state(), hp_rng=engine.hp_rng.get_state(),
                streams=[deepcopy(rng.bit_generator.state) for rng in engine.streams],
                curiosity_rng=deepcopy(engine.curiosity_rng.bit_generator.state),
                conformity_rng=deepcopy(engine.conformity_rng.bit_generator.state))


def continue_branches(engine, saved, horizon, recovery_threshold):
    # No initialization evaluations. Restore the saved state into four independent lanes.
    branch = deepcopy(engine)
    branch.k = len(RULES)
    for key, array in saved['state'].items():
        setattr(branch, key, np.repeat(array, branch.k, axis=0))
    branch.rng.set_state(saved['rng'])
    branch.hp_rng.set_state(saved['hp_rng'])
    for rng, state in zip(branch.streams, saved['streams'], strict=True):
        rng.bit_generator.state = state
    branch.curiosity_rng.bit_generator.state = saved['curiosity_rng']
    branch.conformity_rng.bit_generator.state = saved['conformity_rng']
    branch.tracker = BatchBudget(engine.tracker.function, horizon * engine.n, branch.k, engine.d, 1)
    branch.tracker.best[:] = branch.global_score
    branch.tracker.best_positions[:] = branch.global_best
    initial = engine.global_score[0]
    history = []
    for update in range(1, horizon + 1):
        branch.update(focal=saved['focal'], forced_peers=saved['peers'] if update == 1 else None)
        history.append(branch.global_score.copy())
    assert branch.tracker.count == horizon * engine.n
    assert np.all(np.diff(np.vstack([np.full(branch.k, initial), history]), axis=0) <= 0)
    distance, retained = memory(engine)
    global_direction = engine.global_best[0] - engine.positions[0, saved['focal']]
    rows = []
    for k, rule in enumerate(RULES):
        peer = saved['peers'][k] if k < len(saved['peers']) else None
        alignment = None
        if peer is not None:
            direction = engine.pbests[0, peer] - engine.positions[0, saved['focal']]
            norm = np.linalg.norm(global_direction) * np.linalg.norm(direction)
            if norm > 0:
                alignment = float(np.clip(direction @ global_direction / norm, -1, 1))
        final = branch.global_score[k]
        improvement = float((initial - final) / max(1., abs(initial)))
        events = np.flatnonzero(np.asarray(history)[:, k] < initial)
        rows.append(dict(update=saved['update'], focal=saved['focal'], rule=rule, peer=peer,
                         start_best=float(initial), final_best=float(final), improvement=improvement,
                         recovered=improvement >= recovery_threshold, event=bool(events.size),
                         time=int(events[0]+1) if events.size else horizon,
                         distance=None if peer is None else float(distance[peer]),
                         retained_improvement=None if peer is None else float(retained[peer]), alignment=alignment))
    return rows


def trial(job, cfg, settings, spec, reference, identifier, output, digest):
    path = result_path(output, job)
    if path.is_file():
        if read_result(path)['fingerprint'] != digest:
            raise ValueError(f'Different configuration for {path}')
        return job
    problem = trait_suite(cfg['dimension'], settings['instance_seed'])[job['function']]
    tracker = BatchBudget(BatchFunction(problem), cfg['budget'], 1, cfg['dimension'], settings['trace_points'])
    engine = InterventionHPPSO(tracker, problem['bounds'], cfg['population'], job['seed'], [spec], settings)
    function_index = cfg['functions'].index(job['function'])
    random = np.random.default_rng(np.random.SeedSequence([cfg['diagnostic_seed'], function_index, job['seed']]))
    horizon = cfg['horizon']
    final_update = cfg['budget'] // cfg['population'] - 1
    streak, next_snapshot = 0, 0
    memories, outcomes, snapshots = [], [], []
    stagnant_states = eligible_stagnant_states = 0
    for update in range(final_update + 1):
        stagnant = streak >= cfg['global_stagnation_updates']
        focal, worse, better = candidate_sets(engine)
        stagnant_states += stagnant
        eligible_stagnant_states += stagnant and bool(len(focal))
        if len(focal):
            memories.extend(summarize_state(engine, focal, worse, better, stagnant, update))
        if (stagnant and len(focal) and next_snapshot <= update <= final_update - horizon
                and len(snapshots) < cfg['max_snapshots']):
            i = int(random.choice(focal))
            peers = [random.choice(np.flatnonzero(worse[i])), random.choice(np.flatnonzero(better[i])),
                     random.choice(np.flatnonzero(np.arange(engine.n) != i))]
            saved = snapshot(engine, i, peers, update, streak)
            outcomes.extend(continue_branches(engine, saved, horizon, cfg['recovery_threshold']))
            snapshots.append(saved)
            next_snapshot = update + horizon
        if update < final_update:
            previous = engine.global_score[0]
            engine.update(original=True)
            streak = 0 if engine.global_score[0] < previous else streak + 1

    assert tracker.count == cfg['budget']
    verify_reference = not cfg.get('reference_settings_only', False)
    if verify_reference:
        old = read_result(result_path(reference, job))
        col = old['variant_ids'].index(identifier)
        assert old['status'] == 'budget_reached'
        np.testing.assert_array_equal([n for n, _ in tracker.trace], old['trace_evaluations'])
        np.testing.assert_array_equal(np.asarray([v[0] for _, v in tracker.trace]), np.asarray(old['trace_best'])[:, col])
        np.testing.assert_array_equal(tracker.best_positions[0], old['best_positions'][col])
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = path.with_suffix('.pkl')
    temporary = checkpoint.with_suffix('.tmp')
    with temporary.open('wb') as file:
        pickle.dump(snapshots, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(checkpoint)
    write_result(path, dict(job, fingerprint=digest, memories=memories, outcomes=outcomes,
                            snapshots=len(snapshots), stagnant_states=stagnant_states,
                            eligible_stagnant_states=eligible_stagnant_states,
                            original_final_best=float(tracker.best[0]), reference_trajectory_verified=verify_reference,
                            replay_evaluations=tracker.count,
                            branch_evaluations=len(snapshots)*len(RULES)*horizon*engine.n))
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    reference = Path(cfg['reference']).resolve()
    plan = json.loads((reference / 'plan.json').read_text())
    settings = dict(plan['config'])
    if cfg.get('reference_settings_only', False):
        settings['dimension'] = cfg['dimension']
    for key in ('dimension', 'population', 'budget', 'seeds'):
        if cfg[key] != settings[key]:
            raise ValueError(f'Mismatched reference setting: {key}')
    if cfg['budget'] % cfg['population']:
        raise ValueError('Budget must contain a whole number of populations')
    assert (settings['theta'], settings['lam']) == (.5, .4) and settings['coordinatewise_hppso_draws']
    identifier = plan['protocol']['factorial']['15']
    spec = plan['protocol']['variants'][identifier]['spec']
    assert spec['reinitialize'] and len(spec['traits']) == 4
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256((json.dumps(cfg, sort_keys=True) + plan['fingerprint'] + fingerprint(settings)).encode()
                            + Path(__file__).read_bytes()).hexdigest()
    with shard_lease(output, 'mechanism'):
        manifest = dict(config=cfg, settings=settings, fingerprint=digest, reference_fingerprint=plan['fingerprint'])
        if (output/'plan.json').exists() and json.loads((output/'plan.json').read_text()) != manifest:
            raise ValueError('This output directory contains a different experiment')
        (output/'plan.json').write_text(json.dumps(manifest, indent=2))
        jobs = [dict(function=function, seed=seed) for function in cfg['functions'] for seed in cfg['seeds']]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = [pool.submit(trial, job, cfg, settings, spec, reference, identifier, output, digest) for job in jobs]
            for i, future in enumerate(as_completed(pending), 1):
                job = future.result()
                print(f'[{i}/{len(jobs)}] {job["function"]} seed={job["seed"]}', flush=True)


if __name__ == '__main__':
    main()
