"""Genetic Algorithm with Multi-parent Crossover (GA-MPC)."""

from __future__ import annotations

import random
from itertools import count

import numpy as np


def run_ga_mpc(
    obj_func,
    dim: int,
    lower_bound,
    upper_bound,
    ps: int = 90,
    max_gen: int = 1000,
    cr: float = 1.0,
    p: float = 0.1,
    mu: float = 0.7,
    sigma: float = 0.1,
    archive_ratio: float = 0.5,
    tc_options=None,
):
    """Run GA-MPC and return (best_solution, best_fitness, history)."""
    if tc_options is None:
        tc_options = [2, 3]

    lb = np.atleast_1d(lower_bound)
    ub = np.atleast_1d(upper_bound)

    def tournament_selection(pop, fitness_vals, k):
        selected = random.sample(range(len(pop)), k)
        best = min(selected, key=lambda i: fitness_vals[i])
        return best

    def mpc_crossover(indices, pop, fitness):
        ranked = sorted(indices, key=lambda i: fitness[i])
        x1, x2, x3 = pop[ranked]
        beta = np.random.normal(mu, sigma)
        return [
            x1 + beta * (x2 - x3),
            x2 + beta * (x3 - x1),
            x3 + beta * (x1 - x2),
        ]

    def randomized_operator(offspring, archive):
        if len(archive) == 0:
            return offspring
        mutation_mask = np.random.rand(*offspring.shape) < p
        rows, cols = np.where(mutation_mask)
        if len(rows) > 0:
            archive_indices = np.random.randint(0, len(archive), size=len(rows))
            offspring[rows, cols] = archive[archive_indices, cols]
        return offspring

    def perturb_duplicates(pop):
        _, inv = np.unique(pop, axis=0, return_inverse=True)
        counts = np.bincount(inv)
        dup_unique = np.where(counts > 1)[0]
        dup_mask = np.isin(inv, dup_unique)
        if np.any(dup_mask):
            n = int(np.sum(dup_mask))
            u = np.random.rand(n, 1)
            pop[dup_mask] += np.random.normal(0.5 * u, 0.25 * u, size=(n, dim))
            pop[dup_mask] = np.clip(pop[dup_mask], lb, ub)
        return pop

    population = np.random.uniform(lb, ub, (ps, dim))
    history = []
    global_best = float("inf")
    best_solution = None

    cache = {}

    def evaluate(points):
        nonlocal global_best, best_solution
        values = np.empty(len(points))
        for i, point in enumerate(points):
            key = np.ascontiguousarray(point).tobytes()
            if key not in cache:
                cache[key] = obj_func(point)
            values[i] = cache[key]
            if values[i] < global_best:
                global_best = float(values[i])
                best_solution = point.copy()
        return values

    fitness_vals = evaluate(population)

    for _ in (count() if max_gen is None else range(max_gen)):
        cache = {np.ascontiguousarray(x).tobytes(): value for x, value in zip(population, fitness_vals)}
        m = int(archive_ratio * ps)
        archive_indices = np.argsort(fitness_vals)[:m]
        archive = population[archive_indices]
        archive_fitness = fitness_vals[archive_indices]

        selection_pool = np.array(
            [tournament_selection(population, fitness_vals, random.choice(tc_options)) for _ in range(ps)]
        )

        offspring = []
        for i in range(0, ps - 2, 3):
            if random.random() < cr:
                offspring.extend(mpc_crossover(selection_pool[i:i + 3], population, fitness_vals))

        if offspring:
            offspring = np.clip(randomized_operator(np.array(offspring), archive), lb, ub)
            combined = np.vstack((offspring, archive))
            combined_fitness = np.r_[evaluate(offspring), archive_fitness]
        else:
            combined = archive
            combined_fitness = archive_fitness

        best_idx = np.argsort(combined_fitness)[:ps]
        population = combined[best_idx]
        fitness_vals = combined_fitness[best_idx].copy()
        if len(population) < ps:
            n_add = ps - len(population)
            additions = np.random.uniform(lb, ub, (n_add, dim))
            population = np.vstack((population, additions))
            fitness_vals = np.r_[fitness_vals, evaluate(additions)]

        previous = population.copy()
        population = np.clip(population, lb, ub)
        population = perturb_duplicates(population)
        changed = np.any(population != previous, axis=1)
        fitness_vals[changed] = evaluate(population[changed])
        history.append(global_best)

    return best_solution, global_best, history


run_ga_mpc_cec = run_ga_mpc
