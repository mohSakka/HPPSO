"""Grey Wolf Optimizer (GWO)."""

from __future__ import annotations

import numpy as np


class GWO:
    def __init__(self, objective_function, bounds, num_wolves: int, max_iterations: int):
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_wolves = num_wolves
        self.max_iterations = max_iterations
        self.dimension = len(bounds)

        bounds_array = np.array(bounds)
        self.lower_bounds = bounds_array[:, 0]
        self.upper_bounds = bounds_array[:, 1]

        self.positions = np.random.uniform(
            self.lower_bounds, self.upper_bounds, size=(self.num_wolves, self.dimension)
        )

        self.alpha_pos = np.zeros(self.dimension)
        self.alpha_score = float("inf")
        self.beta_pos = np.zeros(self.dimension)
        self.beta_score = float("inf")
        self.delta_pos = np.zeros(self.dimension)
        self.delta_score = float("inf")

    def optimize(self):
        history = []

        for iteration in range(self.max_iterations):
            current_scores = np.array([self.objective_function(p) for p in self.positions])
            sorted_indices = np.argsort(current_scores)

            new_alpha_score = current_scores[sorted_indices[0]]
            new_alpha_pos = self.positions[sorted_indices[0]]
            new_beta_score = current_scores[sorted_indices[1]]
            new_beta_pos = self.positions[sorted_indices[1]]
            new_delta_score = current_scores[sorted_indices[2]]
            new_delta_pos = self.positions[sorted_indices[2]]

            if new_alpha_score < self.alpha_score:
                self.alpha_score = float(new_alpha_score)
                self.alpha_pos = new_alpha_pos
            if new_beta_score < self.beta_score:
                self.beta_score = float(new_beta_score)
                self.beta_pos = new_beta_pos
            if new_delta_score < self.delta_score:
                self.delta_score = float(new_delta_score)
                self.delta_pos = new_delta_pos

            a = 2 - iteration * (2 / self.max_iterations)
            r1 = np.random.rand(self.num_wolves, self.dimension)
            r2 = np.random.rand(self.num_wolves, self.dimension)

            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * self.alpha_pos - self.positions)
            X1 = self.alpha_pos - A1 * D_alpha

            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * self.beta_pos - self.positions)
            X2 = self.beta_pos - A2 * D_beta

            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * self.delta_pos - self.positions)
            X3 = self.delta_pos - A3 * D_delta

            self.positions = (X1 + X2 + X3) / 3
            self.positions = np.clip(self.positions, self.lower_bounds, self.upper_bounds)
            history.append(self.alpha_score)

        return self.alpha_pos, self.alpha_score, history
