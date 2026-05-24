"""Circle Search Algorithm (CSA)."""

from __future__ import annotations

import numpy as np


class CSA:
    def __init__(self, objective_function, bounds, num_individuals: int, max_iterations: int, c: float = 0.8):
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_individuals = num_individuals
        self.max_iterations = max_iterations
        self.c = c
        self.dimension = len(bounds)

        bounds_array = np.array(bounds)
        self.lower_bounds = bounds_array[:, 0]
        self.upper_bounds = bounds_array[:, 1]

        self.positions = np.random.uniform(
            self.lower_bounds, self.upper_bounds, size=(self.num_individuals, self.dimension)
        )
        self.fitness = np.array([self.objective_function(p) for p in self.positions])

        self.global_best_fitness = float(np.min(self.fitness))
        self.global_best_position = np.copy(self.positions[np.argmin(self.fitness)])

    def optimize(self):
        history = []

        for iteration in range(1, self.max_iterations + 1):
            a = np.pi - np.pi * (iteration / self.max_iterations) ** 2
            p = 1 - 0.9 * (iteration / self.max_iterations) ** 0.5
            w = a * np.random.rand() - a
            rand_tan = np.random.rand(self.num_individuals, self.dimension)
            diff = self.global_best_position - self.positions

            if iteration > (self.c * self.max_iterations):
                self.positions = self.global_best_position + diff * np.tan(w * rand_tan)
            else:
                self.positions = self.global_best_position - diff * np.tan(w * p)

            self.positions = np.clip(self.positions, self.lower_bounds, self.upper_bounds)
            self.fitness = np.array([self.objective_function(p) for p in self.positions])

            min_idx = int(np.argmin(self.fitness))
            min_score = self.fitness[min_idx]
            if min_score < self.global_best_fitness:
                self.global_best_fitness = float(min_score)
                self.global_best_position = np.copy(self.positions[min_idx])

            history.append(self.global_best_fitness)

        return self.global_best_position, self.global_best_fitness, history
