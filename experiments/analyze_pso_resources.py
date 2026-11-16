"""Analyze paired PSO–HPPSO resource measurements."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def compare(pso, hppso):
    difference = hppso - pso
    n = len(difference)
    mean = difference.mean()
    sd = difference.std(ddof=1)
    margin = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    test = stats.ttest_rel(hppso, pso) if sd > 0 else None
    return {
        "pairs": n,
        "pso_mean": pso.mean(),
        "pso_sd": pso.std(ddof=1),
        "hppso_mean": hppso.mean(),
        "hppso_sd": hppso.std(ddof=1),
        "mean_difference": mean,
        "difference_ci_low": mean - margin,
        "difference_ci_high": mean + margin,
        "median_difference": difference.median(),
        "t_statistic": test.statistic if test else np.nan,
        "p_value": test.pvalue if test else np.nan,
    }


def holm(p_values):
    ordered = p_values.dropna().sort_values()
    adjusted = ordered * np.arange(len(ordered), 0, -1)
    return adjusted.cummax().clip(upper=1).reindex(p_values.index)


def analyze_server(runs_path, output):
    runs = pd.read_csv(runs_path)
    runs = runs[runs["algorithm"].isin(["Original PSO", "HPPSO"])]
    valid = runs[(runs.status == "budget_reached") & (runs.evaluations == runs.budget)
                 & np.isfinite(runs.elapsed_seconds) & (runs.elapsed_seconds > 0)]
    keys = ["dimension", "function", "seed", "population", "budget", "fingerprint"]
    pairs = valid[valid.algorithm == "Original PSO"].merge(
        valid[valid.algorithm == "HPPSO"], on=keys, suffixes=("_pso", "_hppso"),
        validate="one_to_one",
    )
    same_host = pairs[pairs.host_pso == pairs.host_hppso].copy()
    results = []
    for (dimension, function), group in same_host.groupby(["dimension", "function"]):
        pso, hppso = group.elapsed_seconds_pso, group.elapsed_seconds_hppso
        results.append({
            "dimension": dimension,
            "function": function,
            "hosts": ", ".join(sorted(group.host_pso.unique())),
            **compare(pso, hppso),
            "median_paired_ratio": (hppso / pso).median(),
        })
    tests = pd.DataFrame(results)
    tests["p_holm"] = holm(tests.p_value)
    tests.to_csv(output / "server_paired_runtime_by_problem.csv", index=False)
    summary = tests.assign(
        hppso_slower=tests.mean_difference > 0,
        hppso_slower_holm=(tests.p_holm < 0.05) & (tests.mean_difference > 0),
        hppso_faster_holm=(tests.p_holm < 0.05) & (tests.mean_difference < 0),
    ).groupby("dimension").agg(
        problems=("function", "size"),
        matched_pairs=("pairs", "sum"),
        median_problem_ratio=("median_paired_ratio", "median"),
        hppso_slower=("hppso_slower", "sum"),
        hppso_slower_holm=("hppso_slower_holm", "sum"),
        hppso_faster_holm=("hppso_faster_holm", "sum"),
    )
    summary.to_csv(output / "server_runtime_by_dimension.csv")
    print(f"{len(same_host)} same-host pairs. {len(pairs) - len(same_host)} cross-host pairs excluded.")
    print(summary.to_string())


def analyze_nn(output):
    trials = pd.read_csv(output / "nn_trials.csv")
    if len(trials) != 60 or set(trials.seed) != set(range(30)) or not trials.evaluations.eq(3000).all():
        raise ValueError("Expected 30 paired seeds at 3,000 evaluations per trial")
    trials["peak_rss_mib"] = trials.peak_rss_bytes / 2**20
    trials["peak_above_baseline_mib"] = (trials.peak_rss_bytes - trials.baseline_rss_bytes) / 2**20
    metrics = ["wall_seconds", "peak_rss_mib", "cpu_seconds", "peak_above_baseline_mib"]
    wide = trials.pivot(index="seed", columns="algorithm", values=metrics)
    if wide.isna().any().any():
        raise ValueError("Missing paired measurements")
    tests = pd.DataFrame({
        metric: compare(wide[metric].PSO, wide[metric].HPPSO) for metric in metrics
    }).T.rename_axis("metric")
    # Wall time and absolute peak working set are the two primary endpoints.
    tests["p_holm_primary"] = holm(tests.loc[metrics[:2], "p_value"])
    tests.to_csv(output / "nn_paired_resources.csv")
    plot_nn(wide, tests, output)
    print(tests.to_string())


def plot_nn(wide, tests, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), layout="constrained")
    panels = (
        ("wall_seconds", "Training wall time", "Seconds"),
        ("peak_rss_mib", "Peak process working set", "MiB"),
    )
    for ax, (metric, title, unit) in zip(axes, panels):
        values = wide[metric].HPPSO - wide[metric].PSO
        summary = tests.loc[metric]
        mean = summary.mean_difference
        interval = [[mean - summary.difference_ci_low], [summary.difference_ci_high - mean]]
        ax.axvline(0, color="#81909b", linewidth=1, linestyle="--", zorder=0)
        ax.scatter(values, wide.index, color="#265e83", alpha=0.76, s=19, label="Paired seed")
        ax.errorbar(mean, 32, xerr=interval, fmt="D", color="#b34a24", capsize=4,
                    markersize=5, linewidth=1.6, label="Mean and 95% CI")
        ax.set(title=title, xlabel=f"HPPSO minus PSO ({unit})", ylim=(-1.5, 34.5))
        ax.set_yticks([0, 5, 10, 15, 20, 25, 29])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Seed")
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    figure.suptitle("Paired resource measurements on Digits", fontweight="semibold")
    figure.savefig(output / "nn_paired_resources.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-runs", type=Path)
    parser.add_argument("--nn", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.server_runs:
        analyze_server(args.server_runs, args.output)
    if args.nn:
        analyze_nn(args.output)


if __name__ == "__main__":
    main()
