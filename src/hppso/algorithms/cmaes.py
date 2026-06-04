"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES)."""

from __future__ import annotations

import numpy as np


class CMAES:
    """Pure Python implementation of CMA-ES (Hansen & Ostermeier)."""

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
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((N + 2) ** 2 + self.mueff),
        )
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (N + 1)) - 1) + self.cs

        self.pc = np.zeros((N, 1))
        self.ps = np.zeros((N, 1))
        self.B = np.eye(N)
        self.D = np.ones((N, 1))
        self.C = self.B @ np.diag(self.D[:, 0] ** 2) @ self.B.T
        self.invsqrtC = self.B @ np.diag(self.D[:, 0] ** -1) @ self.B.T
        self.eigeneval = 0
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

            arx = np.zeros((self.N, self.lam))
            arfitness = np.zeros(self.lam)

            for k in range(self.lam):
                if counteval >= self.stopeval:
                    break
                z = np.random.randn(self.N, 1)
                y = self.B @ (self.D * z)
                x_candidate = np.clip(
                    (self.xmean + self.sigma * y).flatten(),
                    self.lower_bounds,
                    self.upper_bounds,
                )
                arx[:, k] = x_candidate
                arfitness[k] = self.objective_function(arx[:, k])
                counteval += 1

            if counteval >= self.stopeval:
                break

            arindex = np.argsort(arfitness)
            xold = self.xmean
            self.xmean = arx[:, arindex[: self.mu]] @ self.weights.reshape(-1, 1)

            if arfitness[arindex[0]] < best_fitness:
                best_fitness = float(arfitness[arindex[0]])
                best_x = arx[:, arindex[0]]
            history.append(best_fitness)

            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.invsqrtC @ (
                self.xmean - xold
            ) / self.sigma
            hsig = (
                np.linalg.norm(self.ps)
                / np.sqrt(1 - (1 - self.cs) ** (2 * counteval / self.lam))
                / self.chiN
                < 1.4 + 2 / (self.N + 1)
            )
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (
                self.xmean - xold
            ) / self.sigma

            artmp = (1 / self.sigma) * (arx[:, arindex[: self.mu]] - xold)
            self.C = (
                (1 - self.c1 - self.cmu) * self.C
                + self.c1 * (self.pc @ self.pc.T + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                + self.cmu * artmp @ np.diag(self.weights) @ artmp.T
            )

            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

            if counteval - self.eigeneval > (self.lam / (self.c1 + self.cmu) / self.N / 10) and counteval > 0:
                self.eigeneval = counteval
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                D_raw, B_raw = np.linalg.eigh(self.C)
                idx = D_raw.argsort()[::-1]
                self.D = np.sqrt(D_raw[idx]).reshape(-1, 1)
                self.B = B_raw[:, idx]
                self.invsqrtC = self.B @ np.diag(self.D[:, 0] ** -1) @ self.B.T

            if arfitness[arindex[0]] <= self.stopfitness:
                break

        return best_x, best_fitness, history


PureCMAES_Python = CMAES
