"""Supplement the openness report with linear axes and individual seed scores."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from experiments.openness_plot_pages import make_pages, save_pages

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/openness_ablation_100d"
LABELS = ["PSO", "Worse → pbest\n(original)", "Better → pbest", "Any → pbest", "Worse → current"]


def plot_details(output, start_fraction=0.8, completed_only=False):
    output = Path(output)
    if not 0 <= start_fraction < 1:
        raise ValueError("start_fraction must be in [0, 1)")
    plan = json.loads((output / "plan.json").read_text())
    cfg = plan["config"]
    curves = pd.read_csv(output / "convergence_summary.csv")
    summary = pd.read_csv(output / "summary.csv")
    runs = pd.read_csv(output / "runs.csv")
    expected_fingerprints = {name: plan["fingerprint"] for name in cfg["functions"]}
    for source in plan.get("analysis_sources", []):
        for name in source["config"]["functions"]:
            expected_fingerprints[name] = source["fingerprint"]
    if not runs["fingerprint"].eq(runs.function.map(expected_fingerprints)).all():
        raise ValueError("Mixed result fingerprints")
    algorithms = list(plan["variants"])
    function_names = cfg["functions"]
    if completed_only:
        function_names = [name for name in function_names if name in set(summary.function)]
        if not function_names:
            raise ValueError("No complete function blocks available")
    start = start_fraction * cfg["budget"]
    folder = output / "convergence"
    zooms, zoom_axes = make_pages(len(function_names), figsize=(14, 9))
    scores, score_axes = make_pages(len(function_names), figsize=(14, 9))
    for function, zoom_ax, score_ax in zip(function_names, zoom_axes, score_axes):
        subset = runs[runs.function.eq(function)]
        if (len(subset) != len(algorithms) * len(cfg["seeds"])
                or not subset.status.eq("budget_reached").all()
                or not subset.evaluations.eq(cfg["budget"]).all()):
            raise ValueError(f"Incomplete function: {function}")
        title = function.split("_", 1)[1].replace("_", " ").title()
        for index, algorithm in enumerate(algorithms):
            curve = curves[curves.function.eq(function) & curves.algorithm.eq(algorithm)]
            # Retain the preceding observation so the step curve covers the zoom boundary.
            previous = curve[curve.evaluations.lt(start)].tail(1)
            curve = pd.concat([previous, curve[curve.evaluations.ge(start)]])
            zoom_ax.step(curve.evaluations, curve["median"], where="post",
                         color=f"C{index}", linewidth=1.8, label=LABELS[index].replace("\n", " "))
            zoom_ax.fill_between(curve.evaluations, curve.median_ci95_low, curve.median_ci95_high,
                                 step="post", color=f"C{index}", alpha=0.13)
            values = subset[subset.algorithm.eq(algorithm)].sort_values("seed").best_score.to_numpy()
            record = summary[summary.function.eq(function) & summary.algorithm.eq(algorithm)].iloc[0]
            # Identical deterministic seed offsets across arms; no scores are jittered vertically.
            offsets = np.linspace(-0.17, 0.17, len(values))
            score_ax.scatter(index + offsets, values, s=24, color=f"C{index}", alpha=0.6)
            score_ax.vlines(index, record.median_ci95_low, record.median_ci95_high,
                            color="black", linewidth=2.2, zorder=4)
            score_ax.hlines([record.median_ci95_low, record.median_ci95_high],
                           index - .08, index + .08, color="black", linewidth=1.7, zorder=4)
            score_ax.scatter(index, record["median"], marker="D", s=45,
                             color="black", edgecolor="white", zorder=5)
        zoom_ax.set(title=title, xlabel="Objective evaluations", ylabel="Best-so-far objective error (linear)",
                    xlim=(start, cfg["budget"]))
        zoom_ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        zoom_ax.legend(fontsize=8)
        score_ax.set(title=title, ylabel="Final objective error (linear)")
        score_ax.set_xticks(range(len(algorithms)), LABELS, fontsize=9)
        for axis in (zoom_ax, score_ax):
            axis.set_yscale("linear")
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
    suffix = f"\n{len(function_names)}/{len(cfg['functions'])} function blocks complete" if completed_only else ""
    with PdfPages(folder / "linear_detail.pdf") as pdf:
        zoom_files = save_pages(zooms, folder, "late_convergence_linear",
            f"Final {100*(1-start_fraction):g}% of evaluations — linear y-axes\n"
            f"{cfg['dimension']}D, {len(cfg['seeds'])} seeds; median and pointwise 95% bootstrap CI" + suffix, pdf)
        score_files = save_pages(scores, folder, "final_scores_linear",
            f"Final scores — {cfg['dimension']}D, {cfg['budget']:,} evaluations\n"
            "Colored dots: individual seeds; black diamond: median; black interval: 95% bootstrap CI for median" + suffix, pdf)
    html = output / "index.html"
    marker = "<!-- linear-details -->"
    existing = html.read_text().split(marker)[0] if html.exists() else "<!doctype html>"
    html.write_text(existing + marker + '<h2>Linear-scale details</h2>'
        '<p>Late convergence and final seed scores. Intervals describe uncertainty in the median.</p>'
        + ''.join(f'<img src="convergence/{name}" style="max-width:100%">' for name in zoom_files + score_files)
        + '<p><a href="convergence/linear_detail.pdf">Linear details PDF</a></p>')
    print(folder / "linear_detail.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-fraction", type=float, default=0.8)
    parser.add_argument("--completed-only", action="store_true", help="Plot only complete blocks in a partial report")
    args = parser.parse_args()
    plot_details(args.output, args.start_fraction, args.completed_only)
