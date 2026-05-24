"""Fix HPPSO NN training script."""

from __future__ import annotations

import argparse
import random

import numpy as np
from sklearn.datasets import load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hppso.algorithms import HPPSO, PSO
from hppso.nn.simple_mlp import SimpleNeuralNetwork, mean_squared_error, nn_objective_function


def load_dataset(name: str):
    if name == "diabetes":
        data = load_diabetes()
        X, y = data.data, data.target.reshape(-1, 1)
    elif name == "wine":
        data = load_wine()
        X = data.data
        y = (data.target == 0).astype(float).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_pso_variant(algo: str, X_train, y_train, pop: int, iters: int):
    nn = SimpleNeuralNetwork(X_train.shape[1], 16, 8, 1)
    bounds = [(-2.0, 2.0)] * len(nn.get_weights_flat())

    def objective(weights):
        return nn_objective_function(weights, nn, X_train, y_train)

    if algo == "HPPSO":
        opt = HPPSO(objective, n_pop=pop, dimensions=len(bounds), max_it=iters, bounds=(-2.0, 2.0))
        best_score, history = opt.optimize()
        nn.set_weights_flat(opt.get_best_position())
        return nn, best_score, history

    kwargs = {"mutation_rate": 0}
    if algo == "PSO-m":
        kwargs = {"mutation_rate": 0.05, "gaussian_mutation_strength": 0.1}
    elif algo == "PSO-RIW":
        kwargs = {"w_random_range": (0.4, 0.9), "mutation_rate": 0}

    pso = PSO(objective, bounds, pop, iters, **kwargs)
    best_weights, best_score, history = pso.optimize()
    nn.set_weights_flat(best_weights)
    return nn, float(best_score), history


def main():
    parser = argparse.ArgumentParser(description="Neural network weight training with metaheuristics")
    parser.add_argument("--dataset", choices=["diabetes", "wine"], default="diabetes")
    parser.add_argument("--algorithm", choices=["PSO", "PSO-m", "PSO-RIW", "HPPSO"], default="HPPSO")
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    X_train, X_test, y_train, y_test = load_dataset(args.dataset)
    nn, train_mse, history = train_pso_variant(args.algorithm, X_train, y_train, args.pop_size, args.max_iters)

    test_mse = mean_squared_error(y_test, nn.forward(X_test))
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Final convergence: {history[-1]:.6f}")


if __name__ == "__main__":
    main()
