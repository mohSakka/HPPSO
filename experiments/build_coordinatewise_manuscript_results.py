"""Rebuild the mathematical-results supplement from the coordinate-wise archive.

No optimizers are run. The paired-significance convention is shared with
final_benchmark_wilcoxon.py; Friedman/Nemenyi use per-function seed means.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
from itertools import combinations, product
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.final_benchmark_wilcoxon import exact_signed_rank, report

DIMENSIONS = [30, 300, 600, 1000]
COMMON = ["Original PSO", "PSO-m", "PSO-RIW", "HPPSO", "GA-MPC", "GWO", "SHADE", "CSA"]
LABEL = {"Original PSO": "PSO", **{a: a for a in COMMON[1:]}, "CMA-ES": "CMA-ES", "Sep-CMA-ES": "Sep-CMA-ES"}
FUNCTION_NAMES = [
    "Sphere", "Schwefel 2.22", "Zakharov", "Rosenbrock", "Powell Sum (implemented variant)",
    "Rastrigin", "Ackley", "Griewank", "Weierstrass", "Levy", "Schwefel 2.26",
    "Styblinski--Tang", "Michalewicz", "Alpine 1", "Happy Cat", "HGBat", "Bent Cigar",
    "Discus", "Penalized 1", "Zakharov 2 (implemented variant)", "Expanded Schaffer F6",
    "Rotated Ellipsoid", "Rotated Rastrigin", "Rotated Ackley", "Composition I",
    "Rotated Griewank", "Rotated Discus", "Schwefel 1.2", "Rotated Schaffer F7", "Composition II",
]
SELECTED = ["f1_sphere", "f13_michalewicz", "f21_expanded_schaffer_f6", "f30_composition_multimodal"]
COLORS = dict(zip(COMMON + ["CMA-ES", "Sep-CMA-ES"],
                  ["#777777", "#b07aa1", "#56a5b6", "#c43627", "#d49716", "#92a662", "#24734b", "#876647", "#265997", "#265997"]))
STYLES = dict(zip(COMMON + ["CMA-ES", "Sep-CMA-ES"], ["--", ":", "-.", "-", "--", ":", "-", "-.", "-", "--"]))
MARKERS = dict(zip(COMMON + ["CMA-ES", "Sep-CMA-ES"], ["o", "v", "^", "s", "P", "X", "D", "<", "d", "d"]))


def names(d):
    return COMMON + ["CMA-ES" if d == 30 else "Sep-CMA-ES"]


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def validate_signed_rank():
    # Independent enumeration catches tied-rank, zero, and tail-probability errors.
    cases = [np.zeros(5), np.array([1., -1, 2, -3, 0]), np.array([1., 2, 3, 4]), np.array([-2., -2, -1, 0])]
    for d in cases:
        p = exact_signed_rank(d)[2]
        nonzero = d[d != 0]
        r = rankdata(abs(nonzero))
        observed = abs(np.dot(np.sign(nonzero), r))
        stats = [abs(np.dot(s, r)) for s in product([-1, 1], repeat=len(r))]
        assert np.isclose(p, np.mean(np.asarray(stats) >= observed)), (d, p)
    d = np.arange(1., 21.) * np.array([1, -1] * 10)
    assert np.isclose(exact_signed_rank(d)[2], wilcoxon(d, method="exact").pvalue)


def prepare_statistics(source, out):
    validate_signed_rank()
    report(source)
    plan = json.loads((source / "plan.json").read_text())
    runs = pd.read_csv(source / "runs.csv", float_precision="round_trip")
    expected = {(d, f, a, s) for d in DIMENSIONS for f in plan["functions"] for a in names(d) for s in range(20)}
    assert len(runs) == len(expected) == 21600
    assert set(runs[["dimension", "function", "algorithm", "seed"]].itertuples(index=False, name=None)) == expected
    assert runs.status.eq("budget_reached").all() and runs.evaluations.eq(runs.budget).all()
    assert runs.budget.eq(1000 * runs.dimension).all() and np.isfinite(runs.best_score).all()
    assert runs.population.eq(runs.dimension.map({30: 30, 300: 90, 600: 180, 1000: 300})).all()
    def stable_std(values):
        values = np.asarray(values, dtype=float)
        deviations = values - values[0]
        scale = np.max(np.abs(deviations))
        return float(scale * np.std(deviations / scale, ddof=1)) if scale else 0.0

    assert np.isclose(stable_std([1e-300, 2e-300, 3e-300]) / 1e-300, 1.0)
    summary = runs.groupby(["dimension", "function", "algorithm"], as_index=False).agg(
        n=("best_score", "size"), mean=("best_score", "mean"), std=("best_score", stable_std),
        median=("best_score", "median"), minimum=("best_score", "min"), maximum=("best_score", "max"))
    assert np.isfinite(summary[["mean", "std", "median"]].to_numpy()).all()
    summary.to_csv(out / "numerical_results.csv", index=False)
    ranks = pd.read_csv(source / "wilcoxon/rank_details.csv")
    ranks["nonsignificant"] = 8 - ranks.significant_wins - ranks.significant_losses
    ranks.to_csv(out / "per_function_significance_ranks.csv", index=False)
    pairs = pd.read_csv(source / "wilcoxon/pairwise_tests.csv", float_precision="round_trip").fillna({"significant_winner": ""})
    pairs.to_csv(out / "paired_wilcoxon_tests.csv", index=False)
    aggregates = ranks.groupby(["dimension", "algorithm"], as_index=False).agg(
        mean_rank=("rank", "mean"), median_rank=("rank", "median"))
    aggregates.to_csv(out / "wilcoxon_rank_summary.csv", index=False)
    headtohead, omnibus, friedman_ranks, posthoc = [], [], [], []
    for d in DIMENSIONS:
        for a in names(d):
            if a == "HPPSO":
                continue
            t = pairs[pairs.dimension.eq(d) & (
                (pairs.algorithm_a.eq("HPPSO") & pairs.algorithm_b.eq(a)) |
                (pairs.algorithm_a.eq(a) & pairs.algorithm_b.eq("HPPSO")))]
            w = t.significant_winner.eq("HPPSO").sum()
            loss = t.significant_winner.eq(a).sum()
            assert len(t) == 30
            headtohead.append(dict(dimension=d, comparator=a, wins=int(w), losses=int(loss), ties=int(30-w-loss)))
        matrix = summary[summary.dimension.eq(d)].pivot(index="function", columns="algorithm", values="mean")
        matrix = matrix.reindex(index=plan["functions"], columns=names(d))
        rr = np.vstack([rankdata(row) for row in matrix.to_numpy()])
        avg = rr.mean(axis=0)
        test = friedmanchisquare(*matrix.to_numpy().T)
        omnibus.append(dict(dimension=d, blocks=30, algorithms=9, df=8,
                            chi_square=test.statistic, p_raw=test.pvalue))
        for a, r in zip(names(d), avg):
            friedman_ranks.append(dict(dimension=d, algorithm=a, mean_rank=r))
        se = np.sqrt(9*10/(6*30))
        for i, j in combinations(range(9), 2):
            q = np.sqrt(2) * abs(avg[i] - avg[j]) / se
            p = float(studentized_range.sf(q, 9, np.inf))
            posthoc.append(dict(dimension=d, algorithm_a=names(d)[i], algorithm_b=names(d)[j],
                                mean_rank_a=avg[i], mean_rank_b=avg[j], p_nemenyi=p,
                                significant=p <= .05))
    omnibus = pd.DataFrame(omnibus)
    omnibus["p_holm_four_dimensions"] = holm(omnibus.p_raw)
    h = pd.DataFrame(headtohead)
    fr = pd.DataFrame(friedman_ranks)
    ph = pd.DataFrame(posthoc)
    for fn, frame in [("hppso_win_loss_tie.csv", h), ("friedman_tests.csv", omnibus),
                      ("friedman_mean_ranks.csv", fr), ("nemenyi_posthoc.csv", ph)]:
        frame.to_csv(out / fn, index=False)
    audit = dict(source=str(source.resolve()), sha256_runs=hashlib.sha256((source / "runs.csv").read_bytes()).hexdigest(),
                 plan_fingerprint=plan["fingerprint"], trials=len(runs), blocks=120, seeds=20,
                 significant_pairs=int(pairs.p_holm.le(.05).sum()), total_pairs=len(pairs),
                 numerical_std_ddof=1, numerical_std_method="Rescale deviations from the first score before variance calculation to avoid underflow",
                 signed_rank="Exact sign enumeration by subset-sum dynamic programming; zeros omitted; average tied ranks",
                 within_function_family="36 two-sided comparisons, Holm 0.05, independently in each function-dimension block",
                 rank="Descending significant wins minus losses against eight opponents; average rank for equal net scores",
                 friedman="30 function blocks per dimension; one mean over 20 seeds per algorithm and block; tie-corrected chi-square, df=8",
                 posthoc="Nemenyi all 36 pairs within each dimension; studentized range with infinite df; alpha=.05",
                 omnibus_multiplicity="Holm across four dimensions; dimensions not pooled as independent replications",
                 methods_reference="https://www.jmlr.org/papers/v7/demsar06a.html")
    (out / "analysis_manifest.json").write_text(json.dumps(audit, indent=2))
    print("HPPSO pairwise comparisons\n" + h.to_string(index=False), flush=True)
    print("Friedman\n" + omnibus.to_string(index=False), flush=True)
    print("HPPSO Nemenyi\n" + ph[ph.algorithm_a.eq("HPPSO") | ph.algorithm_b.eq("HPPSO")].to_string(index=False), flush=True)
    return plan, summary, ranks, aggregates, h, omnibus, fr, ph


def rank_text(v):
    return f"{v:g}"


def sci(v):
    return f"{v:.2e}".replace("e+0", "e+").replace("e-0", "e-")


def ptex(v):
    if v < .0001:
        mantissa, exponent = f"{v:.2e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"
    return f"{v:.4f}"


def write_tables(out, plan, summary, ranks, agg, h, omnibus, fr, ph):
    for d in DIMENSIONS:
        aa = names(d)
        header = "Function & " + " & ".join(LABEL[a] for a in aa[:-1]) + r" & CMA$^{*}$ & HPPSO \\" + "\n"
        header += r"& \multicolumn{9}{c}{Significance-based rank, $R_{W}$} & W/L/T \\" + "\n"
        cap = (rf"Within-function Wilcoxon results at $d={d}$. $R_W$ ranks significant wins minus losses; equal scores share average ranks. "
               r"HPPSO's W/L/T counts refer to its eight opponents. Holm correction covers 36 pairs per function ($\alpha=0.05$).")
        lines = [r"\begin{table}[p]", r"\centering\scriptsize", rf"\caption{{{cap}}}\label{{tab:coordinate_wilcoxon_{d}}}",
                 r"\setlength{\tabcolsep}{3.2pt}", r"\renewcommand{\arraystretch}{1.03}",
                 r"\begin{tabular}{@{}lrrrrrrrrrc@{}}", r"\toprule", header, r"\midrule"]
        for i, f in enumerate(plan["functions"], 1):
            block = ranks[ranks.dimension.eq(d) & ranks.function.eq(f)].set_index("algorithm")
            hp = block.loc["HPPSO"]
            cells = [rank_text(block.loc[a, "rank"]) for a in aa]
            cells[3] = r"\textbf{" + cells[3] + "}"
            lines.append(rf"$f_{{{i}}}$ & " + " & ".join(cells) + f" & {int(hp.significant_wins)}/{int(hp.significant_losses)}/{int(hp.nonsignificant)}" + r" \\")
        lines.append(r"\midrule")
        ag = agg[agg.dimension.eq(d)].set_index("algorithm")
        for label, col in [("Mean rank", "mean_rank"), ("Median rank", "median_rank")]:
            lines.append(label + " & " + " & ".join(f"{ag.loc[a,col]:.2f}" for a in aa) + r" & --- \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip", r"\begin{minipage}{0.98\linewidth}\footnotesize",
                  (r"$^{*}$CMA denotes CMA-ES at 30D. " if d == 30 else r"$^{*}$CMA denotes Sep-CMA-ES at this dimension. ") +
                  r"Bold identifies HPPSO. Rank differences alone do not imply significance. Function names appear in Appendix~\ref{app:coordinate_numerical}.",
                  r"\end{minipage}", r"\end{table}"]
        (out / f"wilcoxon_{d}d.tex").write_text("\n".join(lines) + "\n")

        # Each cell occupies two lines; this keeps all nine algorithms on one page.
        lines = [r"\begin{table}[p]", r"\centering\scriptsize", rf"\caption{{Final best objective values at $d={d}$ over 20 seeds. Each cell gives the mean, followed by the sample standard deviation in parentheses. Values are rounded to three significant figures; analyses use full precision. $f_2$ is reported on the implemented $\ln(1+f)$ scale.}}\label{{tab:coordinate_numeric_{d}}}",
                 r"\setlength{\tabcolsep}{2.8pt}", r"\renewcommand{\arraystretch}{1.08}", r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
                 "Function & " + " & ".join(LABEL[a] for a in aa[:-1]) + r" & CMA$^{*}$ \\", r"\midrule"]
        for i, f in enumerate(plan["functions"], 1):
            b = summary[summary.dimension.eq(d) & summary.function.eq(f)].set_index("algorithm")
            cells = [rf"\shortstack[r]{{{sci(b.loc[a, 'mean'])}\\({sci(b.loc[a, 'std'])})}}" for a in aa]
            lines.append(rf"$f_{{{i}}}$ & " + " & ".join(cells) + r" \\[2pt]")
        lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip\footnotesize",
                  (r"$^{*}$CMA-ES." if d == 30 else r"$^{*}$Sep-CMA-ES."), r"\end{table}"]
        (out / f"numerical_{d}d.tex").write_text("\n".join(lines) + "\n")

    # Combine two dimensions into one float, sharing the caption and notes.
    for first, second in [(30, 300), (600, 1000)]:
        parts = [r"\begin{table}[p]", r"\centering\scriptsize",
                 rf"\caption{{Within-function Wilcoxon results at {first}D and {second}D. $R_W$ ranks significant wins minus losses, with averaged ranks for equal scores. HPPSO's W/L/T counts refer to its eight opponents. Holm correction covers all 36 pairs per function ($\alpha=0.05$).}}",
                 r"\setlength{\tabcolsep}{3.2pt}\renewcommand{\arraystretch}{1.0}"]
        for panel, d in zip(["a", "b"], [first, second]):
            original = (out / f"wilcoxon_{d}d.tex").read_text()
            tabular = original[original.index(r"\begin{tabular}"):original.index(r"\end{tabular}")+len(r"\end{tabular}")]
            parts += [rf"\label{{tab:coordinate_wilcoxon_{d}}}", rf"\textbf{{({panel}) {d}D}}\par\smallskip", tabular]
            if d == first:
                parts.append(r"\par\medskip")
        parts += [r"\par\smallskip\begin{minipage}{0.98\linewidth}\footnotesize",
                  r"$^{*}$CMA is CMA-ES at 30D and Sep-CMA-ES otherwise. Bold identifies HPPSO. Rank differences alone do not imply significance. Function names appear in Appendix~\ref{app:coordinate_numerical}.",
                  r"\end{minipage}\end{table}"]
        (out / f"wilcoxon_{first}_{second}d.tex").write_text("\n".join(parts)+"\n")

    lines = [r"\begin{table}[htbp]", r"\centering\scriptsize", r"\caption{Wilcoxon summaries across 30 functions per dimension. $\overline R_W$ is the mean of the within-function significance-based ranks. W/L/T counts functions on which HPPSO significantly wins, significantly loses, or has no detected difference against the row algorithm. Each triplet sums to 30.}\label{tab:coordinate_rank_summary}",
             r"\setlength{\tabcolsep}{3.5pt}", r"\renewcommand{\arraystretch}{1.15}", r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
             r"& \multicolumn{2}{c}{30D} & \multicolumn{2}{c}{300D} & \multicolumn{2}{c}{600D} & \multicolumn{2}{c}{1000D} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"Algorithm & $\overline R_W$ & W/L/T & $\overline R_W$ & W/L/T & $\overline R_W$ & W/L/T & $\overline R_W$ & W/L/T \\", r"\midrule"]
    for a in COMMON + ["CMA"]:
        cells = [LABEL.get(a, r"CMA$^{*}$")]
        for d in DIMENSIONS:
            ar = names(d)[-1] if a == "CMA" else a
            v = agg[agg.dimension.eq(d) & agg.algorithm.eq(ar)].iloc[0].mean_rank
            cells.append(rf"\textbf{{{v:.2f}}}" if a == "HPPSO" else f"{v:.2f}")
            if a == "HPPSO":
                cells.append("---")
            else:
                row = h[h.dimension.eq(d) & h.comparator.eq(ar)].iloc[0]
                cells.append(f"{row.wins}/{row.losses}/{row.ties}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip\footnotesize $^{*}$CMA-ES at 30D; Sep-CMA-ES at 300D, 600D, and 1000D.", r"\end{table}"]
    (out / "rank_summary.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{table}[htbp]", r"\centering\small", r"\caption{Friedman tests across all 30 functions at each dimension, using the mean final cost over 20 seeds per algorithm and function. The four omnibus $p$-values are Holm-adjusted. $\overline R_F$ denotes a Friedman mean rank.}\label{tab:coordinate_friedman}",
             r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{@{}rrrrrrr@{}}", r"\toprule",
             r"$d$ & $\chi_F^2(8)$ & $p_{\mathrm{Holm}}$ & \multicolumn{3}{c}{$\overline R_F$} & $N$ \\",
             r"\cmidrule(lr){4-6}", r"& & & HPPSO & SHADE & CMA$^{*}$ & \\", r"\midrule"]
    for d in DIMENSIONS:
        o = omnibus[omnibus.dimension.eq(d)].iloc[0]
        r = fr[fr.dimension.eq(d)].set_index("algorithm").mean_rank
        lines.append(f"{d} & {o.chi_square:.2f} & {ptex(o.p_holm_four_dimensions)} & {r['HPPSO']:.2f} & {r['SHADE']:.2f} & {r[names(d)[-1]]:.2f} & 30" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip\footnotesize $^{*}$CMA-ES at 30D; Sep-CMA-ES otherwise. $N$ is the number of function blocks.", r"\end{table}"]
    (out / "friedman.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{table}[htbp]", r"\centering\small", r"\caption{Nemenyi post hoc comparisons between HPPSO and each competitor after the Friedman tests. Reported $p$-values account for all 36 algorithm pairs within each dimension. A superscript $+$ indicates significantly lower mean rank for HPPSO; $-$ indicates significantly higher mean rank ($\alpha=0.05$).}\label{tab:coordinate_nemenyi}",
             r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule", r"Comparator & 30D & 300D & 600D & 1000D \\", r"\midrule"]
    for a in [x for x in COMMON if x != "HPPSO"] + ["CMA"]:
        cells = [LABEL.get(a, r"CMA$^{*}$")]
        for d in DIMENSIONS:
            ar = names(d)[-1] if a == "CMA" else a
            row = ph[ph.dimension.eq(d) & (
                (ph.algorithm_a.eq(ar) & ph.algorithm_b.eq("HPPSO")) |
                (ph.algorithm_b.eq(ar) & ph.algorithm_a.eq("HPPSO")))].iloc[0]
            p = row.p_nemenyi
            val = r"$<0.0001$" if p < .0001 else f"{p:.4f}"
            if p <= .05:
                rh, ra = ((row.mean_rank_a, row.mean_rank_b) if row.algorithm_a == "HPPSO" else (row.mean_rank_b, row.mean_rank_a))
                val += r"$^{+}$" if rh < ra else r"$^{-}$"
            cells.append(val)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip\footnotesize $^{*}$CMA-ES at 30D; Sep-CMA-ES otherwise. Nonsignificance does not establish equivalence.", r"\end{table}"]
    (out / "nemenyi.tex").write_text("\n".join(lines) + "\n")


def configure_plots():
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 240})


def curve_bundle(source, out, d, f):
    cache = out / "curve_data" / f"{d}d_{f}.npz"
    if cache.exists():
        with np.load(cache) as data:
            return {key: data[key] for key in data.files}
    b = d * 1000
    pop = {30:30, 300:90, 600:180, 1000:300}[d]
    # Start beyond the shared initialization allocation. Checkpoint values are
    # carried forward only; future improvements are never interpolated backward.
    grid = np.unique(np.r_[pop+1, np.geomspace(pop+1, b, 450).astype(int), np.linspace(pop+1, b, 450).astype(int), b])
    data = {"evaluations": grid}
    for a in names(d):
        curves = []
        for s in range(20):
            file = source / "trials" / str(d) / f / a.replace(" ", "_") / f"seed_{s:03d}.json.gz"
            with gzip.open(file, "rt") as stream:
                trial = json.load(stream)
            t = np.asarray(trial["trace"], dtype=float)
            assert t[-1,0] == b and t[-1,1] == trial["best_score"]
            assert np.all(np.diff(t[:,0]) > 0) and np.all(np.diff(t[:,1]) <= 0)
            curves.append(t[np.searchsorted(t[:,0], grid, side="right") - 1, 1])
        data[a] = np.quantile(curves, [.25,.5,.75], axis=0)
    np.savez_compressed(cache, **data)
    return data


def draw_curves(ax, data, d, *, compact=False, ribbons=False):
    for a in names(d):
        q1, med, q3 = data[a]
        ax.step(data["evaluations"] / (1000*d), med, where="post", color=COLORS[a], linestyle=STYLES[a],
                linewidth=1.6 if a == "HPPSO" else (.85 if compact else 1.1), zorder=4 if a == "HPPSO" else 2)
        if ribbons:
            ax.fill_between(data["evaluations"]/(1000*d), q1, q3, step="post", color=COLORS[a], alpha=.07, linewidth=0)
    low = min(np.min(data[a][1]) for a in names(d))
    if low > 0:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(numticks=4 if compact else 6))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if compact and ax.get_ylim()[1] / ax.get_ylim()[0] < 100:
            # Narrow log ranges may contain no powers of ten. Label ordinary
            # cost values while retaining the logarithmic axis transformation.
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
            ax.yaxis.set_major_formatter(ScalarFormatter())
        scale = "log"
    elif low == 0:
        ax.set_yscale("symlog", linthresh=1e-12)
        scale = "symlog, linear threshold 1e-12"
    else:
        ax.yaxis.set_major_locator(MaxNLocator(4 if compact else 6))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3,4), useMathText=True)
        scale = "linear"
    ax.set_xlim(0,1)
    ax.set_xticks([0,.5,1] if compact else [0,.25,.5,.75,1])
    ax.grid(axis="y", which="major", color="#dedede", linewidth=.45)
    ax.tick_params(which="major", length=2.5, pad=2)
    return scale


def legend_items():
    return [Line2D([0],[0], color=COLORS[a], linestyle=STYLES[a], lw=1.7 if a == "HPPSO" else 1.1,
                   label=LABEL[a]) for a in COMMON+["CMA-ES","Sep-CMA-ES"]]


def build_representative_figure(source, out):
    configure_plots()
    (out / "curve_data").mkdir(exist_ok=True)
    (out / "convergence").mkdir(exist_ok=True)
    fig, axs = plt.subplots(4,4, figsize=(7.25,6.7))
    titles = ["Sphere\n($f_1$)", "Michalewicz\n($f_{13}$)", "Expanded Schaffer F6\n($f_{21}$)", "Composition II\n($f_{30}$)"]
    for row,d in enumerate(DIMENSIONS):
        for col,f in enumerate(SELECTED):
            ax=axs[row,col]
            draw_curves(ax, curve_bundle(source,out,d,f), d, compact=True)
            if row == 0: ax.set_title(titles[col], fontsize=8)
            if col == 0: ax.set_ylabel(f"{d}D\nBest cost", fontsize=8)
            ax.tick_params(labelbottom=row==3)
    fig.supxlabel("Fraction of evaluation budget", x=.54,y=.026, fontsize=9)
    fig.legend(handles=legend_items(), loc="lower center", bbox_to_anchor=(.53,.92), ncol=5, frameon=False,
               fontsize=7.1, handlelength=2.1, columnspacing=1.1)
    fig.subplots_adjust(left=.11,right=.985,bottom=.095,top=.82,wspace=.48,hspace=.31)
    fig.savefig(out/"representative_convergence.pdf")
    fig.savefig(out/"representative_convergence.png")
    plt.close(fig)
    print("Representative convergence figure complete", flush=True)


def build_figures(source, out, plan, agg):
    build_representative_figure(source, out)
    fig,ax=plt.subplots(figsize=(6.0,3.25))
    for a in COMMON+["CMA-ES","Sep-CMA-ES"]:
        sub=agg[agg.algorithm.eq(a)].sort_values("dimension")
        ax.plot(sub.dimension,sub.mean_rank,label=LABEL[a], color=COLORS[a], linestyle=STYLES[a],
                marker=MARKERS[a],ms=4,linewidth=1.8 if a=="HPPSO" else 1, markerfacecolor="white" if a in ["CMA-ES","Sep-CMA-ES"] else COLORS[a])
    ax.set(xlabel="Dimension",ylabel="Mean Wilcoxon-based rank",xticks=DIMENSIONS,yticks=range(1,10),ylim=(9.15,.8),xlim=(0,1030))
    ax.grid(axis="y",alpha=.25,linewidth=.5)
    fig.legend(loc="lower center",bbox_to_anchor=(.53,0),ncol=5,frameon=False,fontsize=7.5,columnspacing=1)
    fig.subplots_adjust(left=.105,right=.985,top=.98,bottom=.29)
    fig.savefig(out/"rank_by_dimension.pdf")
    fig.savefig(out/"rank_by_dimension.png")
    plt.close(fig)

    # All 120 curves in a single 30-page atlas, four dimensions per page.
    modes=[]
    with PdfPages(out/"all_convergence_curves.pdf") as pdf:
        for i,f in enumerate(plan["functions"],1):
            fig,axs=plt.subplots(2,2,figsize=(8.0,6.3))
            for ax,d in zip(axs.flat,DIMENSIONS):
                data=curve_bundle(source,out,d,f)
                scale=draw_curves(ax,data,d,ribbons=True)
                modes.append(dict(function=f,dimension=d,yscale=scale))
                ax.set_title(f"{d}D | budget {1000*d:,}")
                ax.set_xlabel("Fraction of evaluation budget")
                ax.set_ylabel("Best ln(1 + f)" if f=="f2_schwefel_2_22" else "Best cost")
            fig.suptitle(f"f{i}: {FUNCTION_NAMES[i-1].replace('--','–')}\n20-seed medians and interquartile ranges",fontsize=11,y=.985)
            fig.legend(handles=legend_items(),loc="lower center",bbox_to_anchor=(.52,.005),ncol=5,frameon=False,fontsize=8)
            fig.subplots_adjust(left=.10,right=.98,bottom=.16,top=.87,wspace=.30,hspace=.42)
            pdf.savefig(fig)
            fig.savefig(out/"convergence"/f"{f}.png",dpi=160)
            plt.close(fig)
            print(f"Convergence atlas {i}/30: {f}",flush=True)
    pd.DataFrame(modes).to_csv(out/"convergence_axis_scales.csv",index=False)
    gallery = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Coordinate-wise benchmark curves</title>',
               '<style>body{font:16px Georgia;margin:2rem auto;max-width:1100px;line-height:1.5}img{width:100%}nav{columns:3}a{color:#204e86}</style>',
               '<h1>Mathematical benchmark convergence curves</h1><p>All 30 functions at 30D, 300D, 600D, and 1000D. Curves show medians over 20 seeds; bands show the interquartile range. The horizontal axis is evaluations divided by budget (1,000 times dimension). Initialization-only points are omitted. The earliest shown checkpoint can retain the last recorded value from initialization. Values between saved checkpoints are carried forward. Positive curves use log axes; zero-valued curves use a signed-log axis with linear threshold 10<sup>−12</sup>; negative-cost functions use linear axes. See the axis-scale CSV for each panel.</p>',
               '<p><a href="../all_convergence_curves.pdf">Complete PDF atlas</a> · <a href="../convergence_axis_scales.csv">Axis scales</a></p><nav>']
    gallery += [f'<a href="#{f}">f{i}: {html.escape(FUNCTION_NAMES[i-1])}</a><br>' for i,f in enumerate(plan['functions'],1)]
    gallery.append('</nav>')
    for i,f in enumerate(plan['functions'],1):
        gallery.append(f'<section id="{f}"><h2>f{i}: {html.escape(FUNCTION_NAMES[i-1])}</h2><img loading="lazy" src="{f}.png" alt="Convergence at four dimensions"></section>')
    (out/"convergence/index.html").write_text('\n'.join(gallery)+'</html>',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"hppso_results_coordinatewise_hppso (2)/final_results")
    parser.add_argument("--output",type=Path,default=ROOT/"Springer_Nature_LaTeX_Template (6)/coordinatewise_results")
    parser.add_argument("--skip-figures",action="store_true")
    parser.add_argument("--representative-only",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.representative_only:
        build_representative_figure(args.source,args.output)
        return
    stats=prepare_statistics(args.source,args.output)
    write_tables(args.output,*stats)
    if not args.skip_figures:
        build_figures(args.source,args.output,stats[0],stats[3])
    shutil.copyfile(args.source/"plan.json",args.output/"source_plan.json")
    print("Completed",args.output,flush=True)


if __name__=="__main__":
    main()
