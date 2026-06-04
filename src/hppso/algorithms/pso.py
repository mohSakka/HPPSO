"""Standard Particle Swarm Optimization and variants (PSO-m, PSO-RIW)."""

from __future__ import annotations

import numpy as np


class PSO:
    """Particle Swarm Optimization with optional Gaussian mutation and random inertia."""

    def __init__(
        self,
        objective_function,
        bounds,
        num_particles: int,
        max_iterations: int,
        w: float = 0.65,
        c1: float = 1.5,
        c2: float = 1.5,
        mutation_rate: float = 0.01,
        gaussian_mutation_strength: float = 0.01,
        w_random_range: tuple[float, float] | None = None,
    ):
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.mutation_rate = mutation_rate
        self.gaussian_mutation_strength = gaussian_mutation_strength
        self.w_random_range = w_random_range

        self.dimension = len(bounds)
        bounds_array = np.array(bounds)
        self.lower_bounds = bounds_array[:, 0]
        self.upper_bounds = bounds_array[:, 1]

        self.positions = np.random.uniform(
            self.lower_bounds, self.upper_bounds, size=(self.num_particles, self.dimension)
        )
        self.velocities = np.random.uniform(-1, 1, size=(self.num_particles, self.dimension))

        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.array([self.objective_function(p) for p in self.positions])

        self.global_best_score = float(np.min(self.personal_best_scores))
        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_scores)]

    def optimize(self):
        history = []

        for _ in range(self.max_iterations):
            if self.w_random_range is not None:
                self.w = np.random.uniform(self.w_random_range[0], self.w_random_range[1])

            r1 = np.random.rand(self.num_particles, self.dimension)
            r2 = np.random.rand(self.num_particles, self.dimension)

            cognitive = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social = self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = self.w * self.velocities + cognitive + social
            self.positions = self.positions + self.velocities

            if self.mutation_rate > 0:
                mutation_mask = np.random.rand(self.num_particles) < self.mutation_rate
                if np.any(mutation_mask):
                    ranges = self.upper_bounds - self.lower_bounds
                    std_devs = ranges * self.gaussian_mutation_strength
                    noise = np.random.normal(0, std_devs, size=(np.sum(mutation_mask), self.dimension))
                    self.positions[mutation_mask] = self.positions[mutation_mask] + noise
                    self.positions[mutation_mask] = np.clip(
                        self.positions[mutation_mask], self.lower_bounds, self.upper_bounds
                    )

            self.positions = np.clip(self.positions, self.lower_bounds, self.upper_bounds)
            current_scores = np.array([self.objective_function(p) for p in self.positions])

            improved = current_scores < self.personal_best_scores
            self.personal_best_scores[improved] = current_scores[improved]
            self.personal_best_positions[improved] = self.positions[improved]

            min_idx = int(np.argmin(current_scores))
            min_score = current_scores[min_idx]
            if min_score < self.global_best_score:
                self.global_best_score = float(min_score)
                self.global_best_position = self.positions[min_idx]

            history.append(self.global_best_score)

        return self.global_best_position, self.global_best_score, history
