"""Standardized algorithm runners for benchmarking experiments."""

from __future__ import annotations

import random

import numpy as np

from hppso.algorithms import (
    CMAES,
    CSA,
    GWO,
    HPPSO,
    PSO,
    SepCMAES,
    SHADE,
    run_ga_mpc,
)


def _bounds_list(dim, lower_bound, upper_bound):
    if np.isscalar(lower_bound):
        return [(lower_bound, upper_bound)] * dim
    return list(zip(lower_bound, upper_bound))


def run_pso(obj_func, dim, lower_bound, upper_bound, pop_size, max_iters, seed):
    random.seed(seed)
    np.random.seed(seed)
    pso = PSO(obj_func, _bounds_list(dim, lower_bound, upper_bound), pop_size, max_iters, mutation_rate=0)
    _, score, history = pso.optimize()
    return score, history


def run_pso_m(obj_func, dim, lower_bound, upper_bound, pop_size, max_iters, seed):
    random.seed(seed)
    np.random.seed(seed)
    pso = PSO(
        obj_func,
        _bounds_list(dim, lower_bound, upper_bound),
        pop_size,
        max_iters,
        mutation_rate=0.05,
        gaussian_mutation_strength=0.1,
    )
    _, score, history = pso.optimize()
    return score, history


def run_pso_riw(obj_func, dim, lower_bound, upper_bound, pop_size, max_iters, seed):
    random.seed(seed)
    np.random.seed(seed)
    pso = PSO(
        obj_func,
        _bounds_list(dim, lower_bound, upper_bound),
        pop_size,
        max_iters,
        w_random_range=(0.4, 0.9),
        mutation_rate=0,
    )
    _, score, history = pso.optimize()
    return score, history


def run_hppso(obj_func, dim, lower_bound, upper_bound, pop_size, max_iters, seed):
    random.seed(seed)
    np.random.seed(seed)
    hppso = HPPSO(obj_func, n_pop=pop_size, dimensions=dim, max_it=max_iters, bounds=(lower_bound, upper_bound))
    score, history = hppso.optimize()
    return score, history


def run_ga_mpc_algo(obj_func, dim, lower_bound, upper_bound, pop_size, max_iters, seed):
    random.seed(seed)
    np.random.seed(seed)
    _, score, history = run_ga_mpc(obj_func, dim, lower_bound, upper_bound, ps=pop_size, max_gen=max_iters)
    return score, history


def run_gwo(obj_func, dim, lower_bound, upper_bound, pop_size, max_iters, seed):
    random.seed(seed)
    np.random.seed(seed)
    gwo = GWO(obj_func, _bounds_list(dim, lower_bound, upper_bound), pop_size, max_iters)
    _, score, history = gwo.optimize()
    return score, history


def run_shade(obj_func, dim, lower_bound, upper_bound, pop_size, max_iters, seed, H: int = 50):
    random.seed(seed)
    np.random.seed(seed)
    shade = SHADE(obj_func, _bounds_list(dim, lower_bound, upper_bound), pop_size, max_iters, H=H)
    _, score, history = shade.optimize()
    return score, history


def run_csa(obj_func, dim, lower_bound, upper_bound, pop_size, max_iters, seed):
    random.seed(seed)
    np.random.seed(seed)
    csa = CSA(obj_func, _bounds_list(dim, lower_bound, upper_bound), pop_size, max_iters)
    _, score, history = csa.optimize()
    return score, history


def run_cmaes(obj_func, dim, lower_bound, upper_bound, pop_size, max_iters, max_evals, seed):
    random.seed(seed)
    np.random.seed(seed)
    lb = np.atleast_1d(lower_bound)
    ub = np.atleast_1d(upper_bound)
    if lb.size == 1:
        lb = np.full(dim, lb.item())
        ub = np.full(dim, ub.item())
    bounds_range = ub - lb
    sigma = 0.25 * np.mean(bounds_range[bounds_range > 0]) if np.any(bounds_range > 0) else 0.3

    cmaes = CMAES(
        obj_func,
        N=dim,
        lower_bounds=lower_bound,
        upper_bounds=upper_bound,
        initial_sigma=sigma,
        max_iterations=max_iters,
        pop_size=pop_size,
        max_total_evaluations=max_evals,
    )
    _, score, history = cmaes.optimize()
    return score, history


def run_sep_cmaes(obj_func, dim, lower_bound, upper_bound, pop_size, max_iters, max_evals, seed):
    random.seed(seed)
    np.random.seed(seed)
    lb = np.atleast_1d(lower_bound)
    ub = np.atleast_1d(upper_bound)
    if lb.size == 1:
        lb = np.full(dim, lb.item())
        ub = np.full(dim, ub.item())
    bounds_range = ub - lb
    sigma = 0.25 * np.mean(bounds_range[bounds_range > 0]) if np.any(bounds_range > 0) else 0.3

    sep = SepCMAES(
        obj_func,
        N=dim,
        lower_bounds=lower_bound,
        upper_bounds=upper_bound,
        initial_sigma=sigma,
        max_iterations=max_iters,
        pop_size=pop_size,
        max_total_evaluations=max_evals,
    )
    _, score, history = sep.optimize()
    return score, history


DEFAULT_ALGORITHMS = {
    "Original PSO": run_pso,
    "PSO-m": run_pso_m,
    "PSO-RIW": run_pso_riw,
    "HPPSO": run_hppso,
    "GA-MPC": run_ga_mpc_algo,
    "GWO": run_gwo,
    "SHADE": run_shade,
    "CSA": run_csa,
    "CMA-ES": run_cmaes,
    "Sep-CMA-ES": run_sep_cmaes,
}


def run_benchmark(
    algorithms,
    problems,
    num_runs: int = 20,
    pop_size: int = 30,
    max_iters: int = 500,
):
    """Run all algorithm-problem pairs and return structured results."""
    max_evals = pop_size * max_iters
    results = {}

    for algo_name, runner in algorithms.items():
        results[algo_name] = {}
        print(f"\n--- {algo_name} ---")

        for problem in problems:
            name = problem["name"]
            obj_func = problem["function"]
            dim = problem["dimension"]
            lb, ub = problem["bounds"]

            scores, histories = [], []
            print(f"  {name} (dim={dim})")

            for run in range(num_runs):
                try:
                    if algo_name in ("CMA-ES", "Sep-CMA-ES"):
                        score, history = runner(obj_func, dim, lb, ub, pop_size, max_iters, max_evals, run)
                    else:
                        score, history = runner(obj_func, dim, lb, ub, pop_size, max_iters, run)
                    scores.append(score)
                    histories.append(history)
                except Exception as exc:
                    print(f"    Run {run + 1} failed: {exc}")
                    scores.append(float("nan"))
                    histories.append([])

            avg_score = float(np.nanmean(scores)) if scores else float("nan")
            max_len = max((len(h) for h in histories if h), default=0)
            if max_len:
                padded = [
                    np.pad(h, (0, max_len - len(h)), mode="edge") if h else np.full(max_len, np.nan)
                    for h in histories
                ]
                avg_history = np.nanmean(padded, axis=0).tolist()
            else:
                avg_history = []

            results[algo_name][name] = {
                "avg_final_score": avg_score,
                "all_final_scores": scores,
                "avg_history": avg_history,
            }
            print(f"    Avg score: {avg_score:.5e}")

    return results
