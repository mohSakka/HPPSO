"""Separable CMA-ES (sep-CMA-ES)."""

from __future__ import annotations

import numpy as np


class SepCMAES:
    """Diagonal (separable) variant of CMA-ES for high-dimensional problems."""

    def __init__(
        self,
        objective_function,
        N: int,
        lower_bounds,
        upper_bounds,
        initial_sigma: float,
        max_iterations: int,
        pop_size: int,
        max_total_evaluations: int,
    ):
        self.objective_function = objective_function
        self.N = N
        self.xmean = np.random.uniform(lower_bounds, upper_bounds, N).astype(float).reshape(-1, 1)
        self.sigma = initial_sigma
        self.stopfitness = 1e-10
        self.max_iterations = max_iterations
        self.lam = pop_size
        self.stopeval = max_total_evaluations

        self.mu = int(np.floor(self.lam / 2))
        weights_orig = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights_orig / np.sum(weights_orig)
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights**2)

        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        self.c1 = 1 / (3 * (N + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, (self.mueff - 2 + 1 / self.mueff) / ((N + 2) ** 2 + self.mueff))
        self.c1_sep = self.c1 * (N + 2) / 3
        self.cmu_sep = self.cmu * (N + 2) / 3
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (N + 1)) - 1) + self.cs

        self.pc = np.zeros((N, 1))
        self.ps = np.zeros((N, 1))
        self.C_diag = np.ones((N, 1))
        self.chiN = N**0.5 * (1 - 1 / (4 * N) + 1 / (21 * N**2))

        self.lower_bounds = np.atleast_1d(lower_bounds)
        self.upper_bounds = np.atleast_1d(upper_bounds)

    def optimize(self):
        counteval = 0
        history = []
        best_fitness = float("inf")
        best_x = self.xmean.flatten()

        for _ in range(self.max_iterations):
            if counteval >= self.stopeval:
                break

            z = np.random.randn(self.N, self.lam)
            y = np.sqrt(self.C_diag) * z
            arx = self.xmean + self.sigma * y

            arfitness = np.zeros(self.lam)
            for k in range(self.lam):
                arx[:, k] = np.clip(arx[:, k], self.lower_bounds, self.upper_bounds)
                arfitness[k] = self.objective_function(arx[:, k])
                counteval += 1

            arindex = np.argsort(arfitness)
            xold = self.xmean
            selected = arindex[: self.mu]
            self.xmean = arx[:, selected] @ self.weights.reshape(-1, 1)

            if arfitness[arindex[0]] < best_fitness:
                best_fitness = float(arfitness[arindex[0]])
                best_x = arx[:, arindex[0]].copy()
            history.append(best_fitness)

            dx = (self.xmean - xold) / self.sigma
            invsqrtC_dx = dx / np.sqrt(self.C_diag)
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrtC_dx

            hsig = (
                np.linalg.norm(self.ps)
                / np.sqrt(1 - (1 - self.cs) ** (2 * counteval / self.lam))
                / self.chiN
                < 1.4 + 2 / (self.N + 1)
            )
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * dx

            artmp = (1 / self.sigma) * (arx[:, selected] - xold)
            rank_mu_diag = (artmp**2) @ self.weights.reshape(-1, 1)
            self.C_diag = (
                (1 - self.c1_sep - self.cmu_sep) * self.C_diag
                + self.c1_sep * (self.pc**2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C_diag)
                + self.cmu_sep * rank_mu_diag
            )

            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
            self.C_diag = np.clip(self.C_diag, 1e-20, 1e20)

            if arfitness[arindex[0]] <= self.stopfitness:
                break

        return best_x, best_fitness, history


SepCMAES_Python = SepCMAES
