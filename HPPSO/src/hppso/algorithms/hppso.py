"""Human Personality Based Particle Swarm Optimization (HPPSO)."""

from __future__ import annotations

import numpy as np


class HPPSO:
    """
    Human Personality Based PSO (HPPSO).

    Each particle carries four human-inspired personality traits that modulate
    exploration and exploitation: curiosity, confidence, aggressiveness, and
    sociality/openness (also called openness). The algorithm combines adaptive
    velocity updates, socialism-based learning from better peers, curiosity-driven
    Gaussian mutation, and stagnation-triggered personality reinitialization.

    Personality vector layout: [curiosity, confidence, aggressiveness, sociality/openness].
    """

    def __init__(
        self,
        obj_func,
        n_pop: int = 30,
        dimensions: int = 10,
        max_it: int = 100,
        bounds=(-100, 100),
        c1: float = 1.5,
        c2: float = 1.5,
        eta: float = 0.01,
        socialism_threshold: float = 0.5,
        lam: float = 0.4,
        stagnation_threshold: int = 10,
    ):
        self.obj_func = obj_func
        self.n_pop = n_pop
        self.dim = dimensions
        self.max_it = max_it

        lb_in, ub_in = bounds
        self.lb = lb_in * np.ones(dimensions) if np.isscalar(lb_in) else np.asarray(lb_in, dtype=float)
        self.ub = ub_in * np.ones(dimensions) if np.isscalar(ub_in) else np.asarray(ub_in, dtype=float)
        self.range = self.ub - self.lb

        self.c1 = c1
        self.c2 = c2
        self.eta = eta
        self.socialism_threshold = socialism_threshold
        self.lam = lam
        self.v_limit = np.inf * self.range
        self.stagnation_threshold = stagnation_threshold

        self.X = np.random.uniform(self.lb, self.ub, size=(n_pop, dimensions))
        self.V = np.zeros((n_pop, dimensions))
        self.fitness = np.array([obj_func(x) for x in self.X])

        self.P = np.copy(self.X)
        self.p_best_fitness = np.copy(self.fitness)
        best_idx = int(np.argmin(self.p_best_fitness))
        self.g_best = np.copy(self.P[best_idx])
        self.g_best_fitness = float(self.p_best_fitness[best_idx])

        self.personality = np.random.rand(n_pop, 4)
        self.stagnation_counters = np.zeros(n_pop)

    def optimize(self):
        history = []

        for _ in range(self.max_it):
            r1 = np.random.rand(self.n_pop, self.dim)
            r2 = np.random.rand(self.n_pop, self.dim)

            curio = self.personality[:, 0]  # curiosity
            conf = self.personality[:, 1]   # confidence
            agg = self.personality[:, 2]    # aggressiveness
            soc = self.personality[:, 3]    # sociality/openness (openness)

            self.V = (
                (0.4 + 0.5 * agg[:, np.newaxis]) * self.V
                + (1 - conf[:, np.newaxis]) * self.c1 * r1 * (self.P - self.X)
                + conf[:, np.newaxis] * self.c2 * r2 * (self.g_best - self.X)
            )

            social_mask = soc > self.socialism_threshold
            for i in np.where(social_mask)[0]:
                better = [
                    idx
                    for idx in range(self.n_pop)
                    if idx != i and self.fitness[idx] < self.fitness[i]
                ]
                if better:
                    j = int(np.random.choice(better))
                    self.V[i] += self.lam * (self.P[j] - self.X[i])

            self.V = np.clip(self.V, -self.v_limit, self.v_limit)
            new_X = self.X + self.V

            mutation_prob = 0.05 * curio[:, np.newaxis]
            mutation_mask = np.random.rand(self.n_pop, self.dim) < mutation_prob
            if np.any(mutation_mask):
                noise = np.random.normal(0, self.eta * self.range, size=self.X.shape)
                new_X[mutation_mask] += noise[mutation_mask]

            new_X = np.clip(new_X, self.lb, self.ub)
            new_fitness = np.array([self.obj_func(x) for x in new_X])
            old_p_best = np.copy(self.p_best_fitness)

            self.X = new_X
            self.fitness = new_fitness

            improved_p = self.fitness < self.p_best_fitness
            self.P[improved_p] = np.copy(self.X[improved_p])
            self.p_best_fitness[improved_p] = self.fitness[improved_p]

            if float(np.min(self.p_best_fitness)) < self.g_best_fitness:
                best_idx = int(np.argmin(self.p_best_fitness))
                self.g_best_fitness = float(self.p_best_fitness[best_idx])
                self.g_best = np.copy(self.P[best_idx])

            stagnated = self.p_best_fitness >= old_p_best
            self.stagnation_counters[stagnated] += 1
            self.stagnation_counters[~stagnated] = 0

            reinit = self.stagnation_counters >= self.stagnation_threshold
            if np.any(reinit):
                self.personality[reinit] = np.random.rand(int(np.sum(reinit)), 4)
                self.stagnation_counters[reinit] = 0

            history.append(self.g_best_fitness)

        return self.g_best_fitness, history

    def get_best_position(self):
        return self.g_best


# Backward-compatible alias used in original notebooks
HPPSO_Modified = HPPSO
