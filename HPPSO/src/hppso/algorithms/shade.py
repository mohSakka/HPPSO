"""Success-History based Adaptive Differential Evolution (SHADE)."""

from __future__ import annotations

import numpy as np


class SHADE:
    def __init__(
        self,
        objective_function,
        bounds,
        num_individuals: int,
        max_iterations: int,
        H: int = 100,
    ):
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_individuals = num_individuals
        self.max_iterations = max_iterations
        self.H = H
        self.dimension = len(bounds)

        bounds_array = np.array(bounds)
        self.lower_bounds = bounds_array[:, 0]
        self.upper_bounds = bounds_array[:, 1]

        self.population = np.random.uniform(
            self.lower_bounds, self.upper_bounds, size=(self.num_individuals, self.dimension)
        )
        self.fitness = np.array([self.objective_function(ind) for ind in self.population])

        self.M_CR = np.full(self.H, 0.5)
        self.M_F = np.full(self.H, 0.5)
        self.archive: list = []

        self.global_best_fitness = float(np.min(self.fitness))
        self.global_best_position = self.population[np.argmin(self.fitness)]

    def optimize(self):
        history = []
        r_idx = 0

        for _ in range(self.max_iterations):
            k_indices = np.random.randint(0, self.H, self.num_individuals)
            m_cr = self.M_CR[k_indices]
            m_f = self.M_F[k_indices]

            CR_i = np.clip(np.random.normal(loc=m_cr, scale=0.1), 0, 1)

            F_i = np.zeros(self.num_individuals)
            for i in range(self.num_individuals):
                while True:
                    F_i[i] = np.random.standard_cauchy() * 0.1 + m_f[i]
                    if F_i[i] > 0:
                        break
            F_i = np.clip(F_i, 0, 1)

            p_best_size = max(2, int(0.05 * self.num_individuals))
            sorted_indices = np.argsort(self.fitness)
            p_best_indices = np.random.choice(sorted_indices[:p_best_size], self.num_individuals, replace=True)
            p_best_positions = self.population[p_best_indices]

            union = np.vstack((self.population, np.array(self.archive))) if self.archive else self.population
            union_size = len(union)

            r1 = np.random.randint(0, union_size, self.num_individuals)
            r2 = np.random.randint(0, union_size, self.num_individuals)
            for i in range(self.num_individuals):
                while r1[i] == i or r2[i] == i or r1[i] == r2[i]:
                    r1[i] = np.random.randint(0, union_size)
                    r2[i] = np.random.randint(0, union_size)

            x_r1 = union[r1]
            x_r2 = union[r2]
            F = F_i[:, np.newaxis]
            mutant = self.population + F * (p_best_positions - self.population) + F * (x_r1 - x_r2)

            j_rand = np.random.randint(0, self.dimension, self.num_individuals)
            rand_cross = np.random.rand(self.num_individuals, self.dimension)
            crossover_mask = (rand_cross < CR_i[:, np.newaxis]) | (
                np.arange(self.dimension) == j_rand[:, np.newaxis]
            )

            trial = np.copy(self.population)
            trial[crossover_mask] = mutant[crossover_mask]
            trial = np.clip(trial, self.lower_bounds, self.upper_bounds)

            trial_fitness = np.array([self.objective_function(ind) for ind in trial])
            improved = trial_fitness < self.fitness

            S_CR = CR_i[improved]
            S_F = F_i[improved]
            delta_f = np.abs(self.fitness[improved] - trial_fitness[improved])

            self.archive.extend(self.population[improved].tolist())
            self.population[improved] = trial[improved]
            self.fitness[improved] = trial_fitness[improved]

            if len(self.archive) > self.num_individuals:
                self.archive = self.archive[-self.num_individuals :]

            if len(S_CR) > 0:
                weights = delta_f / np.sum(delta_f)
                self.M_CR[r_idx] = np.sum(S_CR * weights)
                self.M_F[r_idx] = np.sum(S_F * weights)
                r_idx = (r_idx + 1) % self.H

            min_fit = float(np.min(self.fitness))
            if min_fit < self.global_best_fitness:
                self.global_best_fitness = min_fit
                self.global_best_position = self.population[np.argmin(self.fitness)]

            history.append(self.global_best_fitness)

        return self.global_best_position, self.global_best_fitness, history
