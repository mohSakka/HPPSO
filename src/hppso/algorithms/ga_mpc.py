"""Genetic Algorithm with Multi-parent Crossover (GA-MPC)."""

from __future__ import annotations

import random

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
        return pop[best]

    def mpc_crossover(x1, x2, x3):
        parents = sorted([x1, x2, x3], key=obj_func)
        x1, x2, x3 = parents
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

    for _ in range(max_gen):
        fitness_vals = np.array([obj_func(ind) for ind in population])
        m = int(archive_ratio * ps)
        archive = population[np.argsort(fitness_vals)[:m]]

        selection_pool = np.array(
            [tournament_selection(population, fitness_vals, random.choice(tc_options)) for _ in range(ps)]
        )

        offspring = []
        for i in range(0, ps - 2, 3):
            if random.random() < cr:
                x1, x2, x3 = selection_pool[i], selection_pool[i + 1], selection_pool[i + 2]
                offspring.extend(mpc_crossover(x1, x2, x3))

        if offspring:
            offspring = np.clip(randomized_operator(np.array(offspring), archive), lb, ub)
            combined = np.vstack((offspring, archive))
        else:
            combined = archive

        combined_fitness = np.array([obj_func(ind) for ind in combined])
        best_idx = np.argsort(combined_fitness)[:ps]
        population = combined[best_idx]
        if len(population) < ps:
            n_add = ps - len(population)
            population = np.vstack((population, np.random.uniform(lb, ub, (n_add, dim))))

        population = np.clip(population, lb, ub)
        population = perturb_duplicates(population)

        best_val = float(np.min([obj_func(ind) for ind in population]))
        global_best = min(global_best, best_val)
        history.append(global_best)

    final_idx = int(np.argmin([obj_func(ind) for ind in population]))
    best_solution = population[final_idx]
    return best_solution, obj_func(best_solution), history


run_ga_mpc_cec = run_ga_mpc
