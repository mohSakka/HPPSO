"""Plot generation for experiment results.

Two entry points:

  make_plots(results, cfg, out_dir)
      Called from run.py at the end of a fresh run.

  replot_from_run(run_dir)
      Loads metrics.json + config_used.yaml from a finished run dir and
      regenerates the plots/ subfolder. Useful for iterating on plots
      without rerunning the experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Functions used for per-function convergence plots, if present in results.
HIGHLIGHT_FUNCTIONS = (
    "f1_sphere",
    "f4_rosenbrock",
    "f6_rastrigin",
    "f7_ackley",
    "f11_schwefel_2_26",
    "f13_michalewicz",
)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_plots(results: dict, cfg: dict, out_dir: Path) -> list[Path]:
    """Generate plots into out_dir/plots/ based on experiment_type."""
    import matplotlib

    matplotlib.use("Agg")  # headless

    plots_dir = _ensure_dir(out_dir / "plots")
    etype = cfg.get("experiment_type")
    if etype in ("classical", "cec2011"):
        return _plot_benchmark(results, cfg, plots_dir)
    if etype == "nn_training":
        return _plot_nn(results, cfg, plots_dir)
    return []


def _plot_benchmark(results: dict, cfg: dict, plots_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    name = cfg.get("name", "")

    # Long-form -> pivot (problem x algorithm) of mean scores.
    rows = []
    for algo, probs in results.items():
        for prob, data in probs.items():
            rows.append(
                {
                    "algorithm": algo,
                    "problem": prob,
                    "mean": data.get("avg_final_score"),
                }
            )
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="problem", columns="algorithm", values="mean")

    # 1. Average-rank bar chart.
    ranks = pivot.rank(axis=1, method="min")
    avg_ranks = ranks.mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    avg_ranks.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_ylabel("Average rank (lower is better)")
    ax.set_title(f"Average rank across {len(pivot)} functions — {name}")
    ax.set_xticklabels(avg_ranks.index, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots_dir / "avg_ranks_bar.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 2. Wins per algorithm (best mean on each function).
    wins = pivot.idxmin(axis=1).value_counts()
    for algo in pivot.columns:
        if algo not in wins.index:
            wins[algo] = 0
    wins = wins.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    wins.plot(kind="bar", ax=ax, color="seagreen", edgecolor="black")
    ax.set_ylabel(f"# functions where best (of {len(pivot)})")
    ax.set_title(f"Wins per algorithm — {name}")
    ax.set_xticklabels(wins.index, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots_dir / "wins_per_algorithm.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 3. Per-function convergence curves for highlight functions.
    funcs_present = list(next(iter(results.values())).keys())
    highlights = [f for f in HIGHLIGHT_FUNCTIONS if f in funcs_present]
    if not highlights:
        highlights = funcs_present[:6]

    for func in highlights:
        fig, ax = plt.subplots(figsize=(10, 6))
        any_history = False
        for algo, probs in results.items():
            history = probs.get(func, {}).get("avg_history") or []
            if not history:
                continue
            any_history = True
            arr = np.asarray(history, dtype=float)
            arr = np.where(arr <= 0, np.finfo(float).eps, arr)
            ax.plot(arr, label=algo, linewidth=1.2)
        if not any_history:
            plt.close(fig)
            continue
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Average best fitness (log)")
        ax.set_title(f"{func} — convergence ({name})")
        ax.legend(fontsize=8, ncol=2, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        p = plots_dir / f"convergence_{func}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    return paths


def _plot_nn(results: dict, cfg: dict, plots_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    rows = []
    for algo, m in results.items():
        rows.append(
            {
                "algorithm": algo,
                "train_mse": m.get("train_mse"),
                "test_mse": m.get("test_mse"),
            }
        )
    df = pd.DataFrame(rows).sort_values("test_mse")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    w = 0.4
    ax.bar(x - w / 2, df["train_mse"], width=w, label="Train MSE", color="steelblue", edgecolor="black")
    ax.bar(x + w / 2, df["test_mse"], width=w, label="Test MSE", color="indianred", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(df["algorithm"], rotation=30, ha="right")
    ax.set_ylabel("MSE")
    ax.set_title(f"Train vs test MSE — {cfg.get('name', '')}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots_dir / "train_test_mse.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return [p]


def replot_from_run(run_dir: Path) -> list[Path]:
    """Regenerate plots from a finished run directory."""
    cfg_path = run_dir / "config_used.yaml"
    metrics_path = run_dir / "metrics.json"
    if not cfg_path.exists() or not metrics_path.exists():
        raise SystemExit(
            f"{run_dir} is not a finished experiment run "
            f"(missing config_used.yaml or metrics.json)"
        )
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    with metrics_path.open("r", encoding="utf-8") as fh:
        results = json.load(fh)
    return make_plots(results, cfg, run_dir)
