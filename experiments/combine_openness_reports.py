"""Combine disjoint completed ablation batches without changing raw fingerprints."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from experiments.openness_plot_pages import make_pages, save_pages
from experiments.plot_openness_detail import LABELS, plot_details


def combine(sources, output):
    sources = [Path(p).resolve() for p in sources]
    output = Path(output).resolve()
    if output in sources:
        raise ValueError("Combined report must have its own output directory")
    output.mkdir(parents=True, exist_ok=True)
    frames = {name: [] for name in ("runs", "summary", "paired_comparisons", "convergence_summary", "function_ranks")}
    manifest, definitions, functions = [], {}, []
    config, variants = None, None
    for source in sources:
        plan = json.loads((source / "plan.json").read_text())
        cfg = plan["config"]
        settings = {key: value for key, value in cfg.items() if key != "functions"}
        if config is None:
            config, variants = dict(cfg), plan["variants"]
        if settings != {key: value for key, value in config.items() if key != "functions"} or variants != plan["variants"]:
            raise ValueError("Batches must have identical settings and variants")
        if set(functions) & set(cfg["functions"]):
            raise ValueError("Function selections must be disjoint")
        tables = {name: pd.read_csv(source / f"{name}.csv") for name in frames}
        runs = tables["runs"]
        expected = len(cfg["functions"]) * len(variants) * len(cfg["seeds"])
        if (len(runs) != expected or not runs.status.eq("budget_reached").all()
                or not runs.evaluations.eq(cfg["budget"]).all()
                or not runs.fingerprint.eq(plan["fingerprint"]).all()):
            raise ValueError(f"Incomplete or incompatible source: {source}")
        expected_keys = {(f, a, s) for f in cfg["functions"] for a in variants for s in cfg["seeds"]}
        if set(zip(runs.function, runs.algorithm, runs.seed)) != expected_keys:
            raise ValueError(f"Missing or duplicate trial keys: {source}")
        if set(tables["summary"].function) != set(cfg["functions"]) or not tables["summary"].n.eq(len(cfg["seeds"])).all():
            raise ValueError(f"Incomplete summary: {source}")
        for name, table in tables.items():
            table["source_directory"] = str(source)
            table["source_fingerprint"] = plan["fingerprint"]
            frames[name].append(table)
        functions.extend(cfg["functions"])
        definitions.update(json.loads((source / "problem_definitions.json").read_text()))
        manifest.append(dict(directory=str(source), fingerprint=plan["fingerprint"],
                             config=cfg, imported_trials=plan.get("imported_trials")))
    if not manifest:
        raise ValueError("At least one source is required")
    config["functions"] = functions
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    plan = dict(config=config, variants=variants, fingerprint=digest, analysis_sources=manifest,
                note="Analysis manifest fingerprint; raw trials retain each execution/import plan fingerprint")
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    (output / "problem_definitions.json").write_text(json.dumps(definitions, indent=2))
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in frames.items()}
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    tables["function_ranks"].groupby("algorithm")["rank"].mean().sort_values().to_csv(output / "average_ranks.csv")
    folder = output / "convergence"
    folder.mkdir(exist_ok=True)
    overviews, overview_axes = make_pages(len(functions))
    curves = tables["convergence_summary"]
    with PdfPages(folder / "all_functions.pdf") as pdf:
        for function, overview in zip(functions, overview_axes):
            fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
            for index, algorithm in enumerate(variants):
                curve = curves[curves.function.eq(function) & curves.algorithm.eq(algorithm)]
                for axis in (ax, overview):
                    axis.step(curve.evaluations, curve["median"], where="post", color=f"C{index}",
                              label=LABELS[index].replace("\n", " "), linewidth=1.6)
                    axis.fill_between(curve.evaluations, curve.median_ci95_low, curve.median_ci95_high,
                                      step="post", color=f"C{index}", alpha=.13)
            for axis in (ax, overview):
                axis.set(title=function.split("_", 1)[1].replace("_", " ").title(),
                         xlabel="Objective evaluations (including initialization)",
                         ylabel="Best-so-far objective error, f(x) − f* (f*=0)", xlim=(0, config["budget"]))
                axis.set_yscale("symlog", linthresh=1e-12)
                axis.grid(alpha=.2)
                axis.legend(fontsize=8)
            fig.suptitle("Median; pointwise 95% bootstrap confidence interval", fontsize=11)
            fig.savefig(folder / f"{function}.png", dpi=170)
            pdf.savefig(fig)
            plt.close(fig)
    with PdfPages(folder / "overview_pages.pdf") as pdf:
        names = save_pages(overviews, folder, "overview",
            f"Openness ablation — {len(functions)} functions, {config['dimension']}D, {len(config['seeds'])} seeds\n"
            "Median; pointwise 95% bootstrap confidence intervals", pdf)
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Combined openness study</title>'
        f'<h1>Openness study: {len(functions)} completed functions</h1>'
        '<p>Raw execution fingerprints are retained in runs.csv and plan.json.</p>'
        + ''.join(f'<img src="convergence/{name}" style="max-width:100%">' for name in names)
        + '<p><a href="convergence/all_functions.pdf">All convergence curves</a> · '
        '<a href="summary.csv">Summary and confidence intervals</a> · '
        '<a href="average_ranks.csv">Average ranks</a> · <a href="runs.csv">Raw results</a></p>')
    (output / "report_status.json").write_text(json.dumps(dict(saved_trials=len(tables["runs"]),
        expected_trials=len(tables["runs"]), complete_functions=functions, failed_trials=0), indent=2))
    plot_details(output)
    print(f"Combined {len(functions)} functions and {len(tables['runs'])} valid trials: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    combine(args.sources, args.output)
