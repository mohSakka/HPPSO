"""Plot completed two-seed comparisons from a live experiment journal."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from hppso.experiments.runners import DEFAULT_ALGORITHMS


def read_snapshot(folder):
    rows = []
    for line in (folder/"trials.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass  # A live writer may not yet have finished the last line.
    latest = {}
    for r in rows:
        latest[(r["phase"], r["dimension"], r["function"], r["algorithm"], r["seed"])] = r
    return [r for r in latest.values() if r["phase"] == "full"]


def completed_functions(rows, dim, algorithms=None):
    if algorithms is None:
        algorithms = [a for a in DEFAULT_ALGORITHMS if dim == 30 or a != "CMA-ES"]
    functions = sorted({r["function"] for r in rows if r["dimension"] == dim})
    return [f for f in functions if all(
        {r["seed"] for r in rows if r["dimension"] == dim and r["function"] == f
         and r["algorithm"] == a and r["status"] != "failed"} == {0, 1}
        for a in algorithms)]


def step_curve(row):
    trace = list(row["trace"])
    if not trace:
        return np.array([row["evaluations"]]), np.array([row["best_score"]])
    if trace[-1][0] < row["evaluations"]:
        trace.append([row["evaluations"], row["best_score"]])
    a = np.asarray(trace, float)
    return a[:, 0], a[:, 1]


def align_curves(rows):
    curves = [step_curve(r) for r in rows]
    start = max(x[0] for x, _ in curves)
    end = min(x[-1] for x, _ in curves)
    grid = np.unique(np.concatenate([x[(x >= start) & (x <= end)] for x, _ in curves]+[np.array([end])]))
    values = np.vstack([y[np.searchsorted(x, grid, side="right")-1] for x, y in curves])
    return grid, np.median(values, axis=0), np.min(values, axis=0), np.max(values, axis=0)


def draw(ax, rows, dim, function, algorithms):
    negative = False
    extents = []
    for algo in algorithms:
        group = sorted([r for r in rows if r["dimension"] == dim and r["function"] == function
                        and r["algorithm"] == algo], key=lambda r:r["seed"])
        if len(group) != 2:
            continue
        x, median, lower, upper = align_curves(group)
        color = f"C{list(DEFAULT_ALGORITHMS).index(algo)}"
        ax.step(x, median, where="post", label=algo, color=color, lw=1.8 if algo=="HPPSO" else 1.15)
        ax.fill_between(x, lower, upper, step="post", alpha=.09, color=color)
        negative |= bool(np.any(lower < -1e-10))
        extents.append((float(lower.min()), float(upper.max())))
    ax.set_xscale("log")
    if not negative and extents:
        low = min(v[0] for v in extents)
        high = max(v[1] for v in extents)
        if low > 0:
            ax.set_yscale("log")
            ax.set_ylim(low*.7, high*1.3)
        else:
            ax.set_yscale("symlog", linthresh=1e-12)
            ax.set_ylim(min(0., low*1.3), max(1e-12, high*1.3))
    ax.set_xlabel("Actual objective evaluations")
    ax.set_ylabel("Best objective value" if function != "f2_schwefel_2_22" else "Best ln(1 + Schwefel 2.22)")
    ax.set_title(f"{dim}D — {function}", fontsize=10)
    ax.grid(alpha=.25)


def create_snapshot(folder, output, dimensions=(30, 1000)):
    output.mkdir(parents=True, exist_ok=True)
    frozen = output/"snapshot.json"
    rows = json.loads(frozen.read_text()) if frozen.exists() else read_snapshot(folder)
    rows = [r for r in rows if r["dimension"] in dimensions]
    plan = json.loads((folder/"plan.json").read_text()) if (folder/"plan.json").exists() else {}
    (output/"snapshot.json").write_text(json.dumps(rows, allow_nan=False), encoding="utf-8")
    ranks, winners, included = [], [], {}
    highlight = ("f1_sphere", "f3_zakharov", "f4_rosenbrock", "f6_rastrigin", "f7_ackley", "f11_schwefel_2_26")
    for dim in dimensions:
        algorithms = plan.get("algorithms_by_dimension", {}).get(str(dim), [a for a in DEFAULT_ALGORITHMS if dim==30 or a!="CMA-ES"])
        functions = completed_functions(rows, dim, algorithms)
        included[str(dim)] = functions
        if not functions:
            continue
        frame = pd.DataFrame([r for r in rows if r["dimension"]==dim and r["function"] in functions])
        pivot = frame.groupby(["function", "algorithm"]).best_score.mean().unstack()
        r = pivot.rank(axis=1, method="average")
        for algo, rank in r.mean().sort_values().items():
            ranks.append(dict(dimension=dim, algorithm=algo, average_rank=rank, n_functions=len(functions)))
        for f, a in pivot.idxmin(axis=1).items():
            winners.append(dict(dimension=dim, function=f, algorithm=a, mean_score=pivot.loc[f,a]))
        with PdfPages(output/f"convergence_{dim}d_all.pdf") as pdf:
            for function in functions:
                fig, ax = plt.subplots(figsize=(10, 6))
                draw(ax, rows, dim, function, algorithms)
                ax.legend(fontsize=8, ncol=2)
                fig.suptitle("Two-seed median; shading is the range across seeds", fontsize=10)
                fig.tight_layout()
                fig.savefig(output/f"{dim}d_{function}.png", dpi=150)
                pdf.savefig(fig)
                plt.close(fig)
        selected = [f for f in highlight if f in functions]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
        for ax, function in zip(axes.flat, selected):
            draw(ax, rows, dim, function, [a for a in ("HPPSO", "Original PSO", "SHADE", "Sep-CMA-ES" if "Sep-CMA-ES" in algorithms else "CMA-ES") if a in algorithms])
        for ax in list(axes.flat)[len(selected):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"{dim}D convergence snapshot — two-seed median and seed range", fontsize=15)
        fig.tight_layout()
        fig.savefig(output/f"convergence_{dim}d_overview.png", dpi=150)
        plt.close(fig)
    pd.DataFrame(ranks).to_csv(output/"average_ranks.csv", index=False)
    pd.DataFrame(winners).to_csv(output/"function_winners.csv", index=False)
    counts = {str(d):sum(r["dimension"]==d for r in rows) for d in (30,1000)}
    manifest = dict(completed_trials=counts, failed_trials=sum(r["status"]=="failed" for r in rows),
                    included_functions=included, note="Only fully completed two-seed, all-algorithm functions are compared. No extrapolation past either seed's last evaluation.")
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(pd.DataFrame(ranks).to_string(index=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[30,1000])
    args=parser.parse_args()
    create_snapshot(args.run_dir, args.output_dir, args.dimensions)
