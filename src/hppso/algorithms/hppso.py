"""Human Personality Based Particle Swarm Optimization (HPPSO)."""

from __future__ import annotations

import numpy as np

from ._pso_bounds import reflect_bounds, velocity_limit


class HPPSO:
    """
    Human Personality Based PSO (HPPSO).

    Each particle carries four human-inspired personality traits that modulate
    exploration and exploitation: curiosity, confidence, aggressiveness, and
    openness. The algorithm combines adaptive
    velocity updates, attraction to a randomly sampled worse peer, curiosity-driven
    Gaussian mutation, and stagnation-triggered personality reinitialization.

    Personality vector layout: [curiosity, confidence, aggressiveness, openness].
    """

    def __init__(
        self,
        obj_func,
        n_pop: int = 30,
        dimensions: int = 10,
        max_it: int = 100,
        bounds=(-100, 100),
        c1: float = 2.0,
        c2: float = 2.0,
        eta: float = 0.001,
        socialism_threshold: float = 0.5,
        lam: float = 0.4,
        stagnation_threshold: int = 10,
        alpha_curio: float = 0.05,
        v_max: float | np.ndarray | None = None,
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
        self.alpha_curio = alpha_curio
        self.v_limit = velocity_limit(self.lb, self.ub, v_max)
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
        self.stagnation_counters = np.zeros(n_pop, dtype=int)

    def _start_epoch(self):
        """Allow variants to update shared parameters before a swarm update."""

    def _effective_aggressiveness(self):
        return self.personality[:, 2]

    def _attraction_weights(self):
        conformity = self.personality[:, 1, np.newaxis]
        return 1 - conformity, conformity

    def _refresh_personalities(self, improved, previous_best, reinit):
        if np.any(reinit):
            self.personality[reinit] = np.random.rand(int(np.sum(reinit)), 4)

    def optimize(self):
        history = []

        for _ in range(self.max_it):
            self._start_epoch()
            r1 = np.random.rand(self.n_pop, self.dim)
            r2 = np.random.rand(self.n_pop, self.dim)

            curio = self.personality[:, 0]  # curiosity
            cognitive_weight, social_weight = self._attraction_weights()
            agg = self._effective_aggressiveness()
            openness = self.personality[:, 3]

            self.V = (
                (0.4 + 0.5 * agg[:, np.newaxis]) * self.V
                + cognitive_weight * self.c1 * r1 * (self.P - self.X)
                + social_weight * self.c2 * r2 * (self.g_best - self.X)
            )

            # Sample one peer from ALL other particles, then test its fitness.
            # Mapping draws in [0, n_pop - 1) past each particle's own index
            # excludes self uniformly without rejection loops or an N x N matrix.
            if self.n_pop > 1:
                open_indices = np.flatnonzero(openness > self.socialism_threshold)
                peers = np.random.randint(0, self.n_pop - 1, size=open_indices.size)
                peers += peers >= open_indices
                worse = self.fitness[peers] > self.fitness[open_indices]
                active = open_indices[worse]
                self.V[active] += self.lam * (self.P[peers[worse]] - self.X[active])

            self.V = np.clip(self.V, -self.v_limit, self.v_limit)
            new_X = self.X + self.V

            mutation_prob = self.alpha_curio * curio[:, np.newaxis]
            mutation_mask = np.random.rand(self.n_pop, self.dim) < mutation_prob
            if np.any(mutation_mask):
                noise = np.random.normal(0, self.eta * self.range, size=self.X.shape)
                new_X[mutation_mask] += noise[mutation_mask]

            reflect_bounds(new_X, self.V, self.lb, self.ub)
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

            stagnated = self.p_best_fitness == old_p_best
            self.stagnation_counters[stagnated] += 1
            self.stagnation_counters[~stagnated] = 0

            reinit = self.stagnation_counters >= self.stagnation_threshold
            self._refresh_personalities(improved_p, old_p_best, reinit)
            self.stagnation_counters[reinit] = 0

            history.append(self.g_best_fitness)

        return self.g_best_fitness, history

    def get_best_position(self):
        return self.g_best


# Backward-compatible alias used in original notebooks
HPPSO_Modified = HPPSO
