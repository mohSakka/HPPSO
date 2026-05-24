"""Plotting helpers for benchmark experiments."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def save_figure(name: str, folder: str = "results", dpi: int = 150):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.png")
    plt.savefig(path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_convergence(curves, labels, title: str, filename: str, folder: str = "results"):
    plt.figure(figsize=(10, 6))
    for curve, label in zip(curves, labels):
        c = np.asarray(curve, dtype=float)
        c[c <= 0] = np.finfo(float).eps
        plt.plot(c, label=label)

    plt.yscale("log")
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Global Best Fitness")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    save_figure(filename, folder=folder)


def run_avg(model_factory, seeds=None):
    """Average convergence histories over multiple random seeds."""
    if seeds is None:
        seeds = list(range(20))

    histories = []
    for seed in seeds:
        np.random.seed(seed)
        optimizer = model_factory()
        _, history = optimizer.optimize()
        histories.append(history)

    if not histories:
        return np.array([])

    max_len = max(len(h) for h in histories)
    padded = []
    for h in histories:
        if h:
            padded.append(h + [h[-1]] * (max_len - len(h)))
        else:
            padded.append([np.nan] * max_len)
    return np.mean(padded, axis=0)
