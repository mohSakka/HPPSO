"""Controlled PSO-only openness ablation with matched random streams."""
from __future__ import annotations

import numpy as np

from ._pso_bounds import reflect_bounds, velocity_limit

VARIANTS = {
    "PSO": ("none", "pbest"),
    "PSO-open-worse-pbest": ("worse", "pbest"),
    "PSO-open-better-pbest": ("better", "pbest"),
    "PSO-open-any-pbest": ("any", "pbest"),
    "PSO-open-worse-current": ("worse", "current"),
}


def openness_delta(positions, personal_bests, scores, peers, enabled, gate, target, lam):
    """Gate using current fitness; never filter/resample the sampled peers."""
    active = enabled.copy()
    if gate == "worse":
        active &= scores[peers] > scores
    elif gate == "better":
        active &= scores[peers] < scores
    elif gate == "none":
        active[:] = False
    elif gate != "any":
        raise ValueError(f"Unknown gate: {gate}")
    if target not in ("pbest", "current"):
        raise ValueError(f"Unknown target: {target}")
    destination = personal_bests if target == "pbest" else positions
    delta = np.zeros_like(positions)
    delta[active] = lam * (destination[peers[active]] - positions[active])
    return delta


class OpennessPSO:
    """Same standard PSO dynamics in every arm; only the extra term changes.

    Base RNG matches numpy.random.seed(seed) in the repository's standard PSO.
    Trait and peer RNGs are independent, so openness consumes no base PSO draws.
    """

    def __init__(self, objective, bounds, population=50, variant="PSO", seed=0,
                 w=0.65, c1=2.0, c2=2.0, lam=0.4, threshold=0.5):
        self.objective = objective
        self.gate, self.target = VARIANTS[variant]
        self.rng = np.random.RandomState(seed)
        self.peer_rng = np.random.default_rng(np.random.SeedSequence([seed, 107]))
        trait_rng = np.random.default_rng(np.random.SeedSequence([seed, 211]))
        self.lower, self.upper = np.asarray(bounds, dtype=float)
        self.population, self.dimension = population, len(self.lower)
        self.w, self.c1, self.c2, self.lam = w, c1, c2, lam
        self.limit = velocity_limit(self.lower, self.upper)
        self.positions = self.rng.uniform(self.lower, self.upper, (population, self.dimension))
        self.velocities = np.zeros_like(self.positions)
        self.openness = trait_rng.random(population)
        self.enabled = self.openness > threshold
        self.scores = np.array([objective(x) for x in self.positions])
        self.personal_bests = self.positions.copy()
        self.best_scores = self.scores.copy()
        i = self.best_scores.argmin()
        self.global_best = self.personal_bests[i].copy()
        self.global_score = float(self.best_scores[i])

    def step(self):
        r1 = self.rng.rand(self.population, self.dimension)
        r2 = self.rng.rand(self.population, self.dimension)
        self.velocities = (self.w * self.velocities
            + self.c1 * r1 * (self.personal_bests - self.positions)
            + self.c2 * r2 * (self.global_best - self.positions))
        if self.population > 1 and self.gate != "none":
            # Sample for all particles to match peer streams across the four arms.
            # Only enabled particles can receive an update.
            peers = self.peer_rng.integers(0, self.population - 1, self.population)
            peers += peers >= np.arange(self.population)
            self.velocities += openness_delta(self.positions, self.personal_bests,
                self.scores, peers, self.enabled, self.gate, self.target, self.lam)
        self.velocities = np.clip(self.velocities, -self.limit, self.limit)
        self.positions += self.velocities
        reflect_bounds(self.positions, self.velocities, self.lower, self.upper)
        self.scores = np.array([self.objective(x) for x in self.positions])
        improved = self.scores < self.best_scores
        self.best_scores[improved] = self.scores[improved]
        self.personal_bests[improved] = self.positions[improved]
        i = self.best_scores.argmin()
        if self.best_scores[i] < self.global_score:
            self.global_score = float(self.best_scores[i])
            self.global_best = self.personal_bests[i].copy()

    def optimize(self):
        # The strict objective interrupts at the exact evaluation budget.
        while True:
            self.step()
