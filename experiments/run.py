"""Run an HPPSO experiment from a YAML config.

Usage:
    uv run python experiments/run.py experiments/configs/classical_30d.yaml
    uv run python experiments/run.py classical_30d              # shorthand
    uv run hppso-run-experiment experiments/configs/cec2011.yaml

    # Skip plots on a fresh run:
    uv run hppso-run-experiment classical_30d --no-plots

    # Regenerate plots for an existing run without rerunning the experiment:
    uv run hppso-run-experiment --plots-from experiments/results/classical_30d/20260604_164332

Outputs land in experiments/results/<name>/<timestamp>/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def select_algorithms(names: list[str], registry: dict) -> dict:
    missing = [n for n in names if n not in registry]
    if missing:
        raise SystemExit(
            f"Unknown algorithms: {missing}\nAvailable: {sorted(registry.keys())}"
        )
    return {n: registry[n] for n in names}


def run_classical(cfg: dict, out_dir: Path) -> dict:
    from hppso.benchmarks.classical import build_problem_suite
    from hppso.experiments.runners import DEFAULT_ALGORITHMS, run_benchmark

    np.random.seed(cfg.get("seed", 42))
    problems = build_problem_suite(dim=cfg["dim"], random_shift=False)
    algos = select_algorithms(cfg["algorithms"], DEFAULT_ALGORITHMS)
    return run_benchmark(
        algos,
        problems,
        num_runs=cfg["runs"],
        pop_size=cfg["pop_size"],
        max_iters=cfg["max_iters"],
    )


def run_cec2011(cfg: dict, out_dir: Path) -> dict:
    from hppso.benchmarks.cec2011 import load_cec2011_problems
    from hppso.experiments.runners import DEFAULT_ALGORITHMS, run_benchmark

    np.random.seed(cfg.get("seed", 42))
    problems = load_cec2011_problems()
    algos = select_algorithms(cfg["algorithms"], DEFAULT_ALGORITHMS)
    return run_benchmark(
        algos,
        problems,
        num_runs=cfg["runs"],
        pop_size=cfg["pop_size"],
        max_iters=cfg["max_iters"],
    )


def run_nn_training(cfg: dict, out_dir: Path) -> dict:
    import random

    from hppso.experiments.train_neural_network import load_dataset, train_pso_variant
    from hppso.nn.simple_mlp import mean_squared_error

    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test = load_dataset(cfg["dataset"])

    out: dict = {}
    for algo in cfg["algorithms"]:
        nn, train_mse, history = train_pso_variant(
            algo, X_train, y_train, cfg["pop_size"], cfg["max_iters"]
        )
        test_mse = float(mean_squared_error(y_test, nn.forward(X_test)))
        out[algo] = {
            "train_mse": float(train_mse),
            "test_mse": test_mse,
            "final_history": float(history[-1]) if history else None,
        }
    return out


DISPATCH = {
    "classical": run_classical,
    "cec2011": run_cec2011,
    "nn_training": run_nn_training,
}


def write_outputs(
    results: dict,
    cfg: dict,
    out_dir: Path,
    plots: bool = True,
) -> list[Path]:
    """Write metrics/CSV/config snapshot and (optionally) plots. Returns plot paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    output_opts = cfg.get("output", {}) or {}

    # Snapshot the config used (provenance).
    (out_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    if output_opts.get("save_json", True):
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)

    if output_opts.get("save_csv", True) and cfg["experiment_type"] in (
        "classical",
        "cec2011",
    ):
        rows = []
        for algo, prob_results in results.items():
            for prob, data in prob_results.items():
                scores = data.get("all_final_scores", [])
                rows.append(
                    {
                        "experiment": cfg.get("name", ""),
                        "algorithm": algo,
                        "problem": prob,
                        "mean_best_score": data.get("avg_final_score"),
                        "std_best_score": float(np.nanstd(scores)) if scores else float("nan"),
                        "n_runs": len(scores),
                    }
                )
        pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    plot_paths: list[Path] = []
    if plots and output_opts.get("save_plots", True):
        from experiments.plots import make_plots

        try:
            plot_paths = make_plots(results, cfg, out_dir)
        except Exception as exc:  # plots are secondary — never fail the run
            print(f"WARNING: plot generation failed: {exc}")
    return plot_paths


def resolve_config_path(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    # shorthand: 'classical_30d' -> experiments/configs/classical_30d.yaml
    candidate = CONFIGS_DIR / (p.name if p.suffix else f"{p.name}.yaml")
    if candidate.exists():
        return candidate
    raise SystemExit(f"Config not found: {arg}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an HPPSO experiment from a YAML config."
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to YAML config file (or name under experiments/configs/). "
        "Omit when using --plots-from.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output dir (default: experiments/results/<name>/<timestamp>/)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation on a fresh run.",
    )
    parser.add_argument(
        "--plots-from",
        default=None,
        help="Regenerate plots from a finished run dir (no rerun). "
        "Reads <dir>/metrics.json + <dir>/config_used.yaml.",
    )
    args = parser.parse_args(argv)

    # Replot-only mode: skip the experiment, just rebuild plots.
    if args.plots_from:
        from experiments.plots import replot_from_run

        run_dir = Path(args.plots_from)
        if not run_dir.exists():
            raise SystemExit(f"Run dir not found: {run_dir}")
        print(f"Regenerating plots in {run_dir / 'plots'} from existing metrics.json")
        plot_paths = replot_from_run(run_dir)
        print(f"\nWrote {len(plot_paths)} plot(s).")
        return 0

    if not args.config:
        parser.error("config is required unless --plots-from <dir> is given")

    cfg_path = resolve_config_path(args.config)
    cfg = load_config(cfg_path)

    name = cfg.get("name") or cfg_path.stem
    etype = cfg.get("experiment_type")
    if etype not in DISPATCH:
        raise SystemExit(
            f"Unknown experiment_type '{etype}'. Valid: {sorted(DISPATCH.keys())}"
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else RESULTS_ROOT / name / timestamp
    )

    print(f"=== HPPSO experiment ===")
    print(f"  config:  {cfg_path}")
    print(f"  name:    {name}")
    print(f"  type:    {etype}")
    print(f"  output:  {out_dir}")
    print(f"  plots:   {'no' if args.no_plots else 'yes'}")
    print()

    results = DISPATCH[etype](cfg, out_dir)
    plot_paths = write_outputs(results, cfg, out_dir, plots=not args.no_plots)

    print(f"\nDone. Results written to: {out_dir}")
    if plot_paths:
        print(f"Plots ({len(plot_paths)}): {out_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
