"""Trait study statistics, parameter sweeps, convergence and factorial interactions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from experiments.trait_study import jobs,result_path
from experiments.final_benchmark import read_result
from experiments.trait_interactions import bootstrap_ci,normalized_response,factorial_contrasts
from experiments.trait_curve_plots import draw_curve


def interaction_plot(effects,title,path,names,response_label="final/initial error fraction"):
    fig,ax=plt.subplots(figsize=(8,6),constrained_layout=True)
    matrix=np.zeros((4,4));intervals={}
    for i in range(4):
        for j in range(i+1,4):
            row=effects[(1<<i)|(1<<j)]
            matrix[i,j]=matrix[j,i]=row[0];intervals[i,j]=intervals[j,i]=row
    scale=max(np.abs(matrix).max(),1e-12)
    shown=np.ma.array(matrix,mask=np.eye(4,dtype=bool))
    im=ax.imshow(shown,cmap="RdBu_r",vmin=-scale,vmax=scale)
    for i in range(4):
        for j in range(4):
            if i==j:ax.text(j,i,"—",ha="center",va="center")
            else:
                mean,lo,hi=intervals[i,j]
                ax.text(j,i,f"{mean:.2g}\n[{lo:.2g}, {hi:.2g}]",ha="center",va="center",fontsize=9,
                        color="white" if abs(mean)>scale*.65 else "black")
    ax.set_xticks(range(4),names,rotation=20);ax.set_yticks(range(4),names)
    ax.set_title(title+"\nMean pair interaction [95% seed-bootstrap CI]",fontsize=11)
    fig.colorbar(im,ax=ax,label=f"Interaction in {response_label} (lower favors the combination)")
    fig.savefig(path,dpi=170);fig.savefig(path.with_suffix(".pdf"));plt.close(fig)


def report(output,sweep_curves=False):
    output=Path(output);plan=json.loads((output/"plan.json").read_text())
    cfg,p=plan["config"],plan["protocol"];identifiers=list(p["variants"]);index={v:i for i,v in enumerate(identifiers)}
    expected_fingerprints={name:plan["fingerprint"] for name in cfg["functions"]}
    for source in plan.get("analysis_sources",[]):
        for name in source["config"]["functions"]:expected_fingerprints[name]=source["fingerprint"]
    draws=np.random.default_rng(cfg["bootstrap_seed"]).integers(0,len(cfg["seeds"]),
        (cfg["bootstrap_resamples"],len(cfg["seeds"])))
    records={};rows=[]
    for job in jobs(cfg):
        path=result_path(output,job)
        if not path.exists():continue
        r=read_result(path)
        if r["fingerprint"]!=expected_fingerprints[job["function"]] or r["variant_ids"]!=identifiers:raise ValueError("Mixed batch results")
        records[(job["function"],job["seed"])]=r
        for i,v in enumerate(identifiers):
            rows.append(dict(function=job["function"],seed=job["seed"],variant=v,label=p["variants"][v]["label"],
                status=r["status"],evaluations=r["evaluations_per_path"],best_score=r["best_scores"][i],
                initial_best_score=None if r["initial_best"] is None else r["initial_best"][i],
                reset_events=None if r["reset_events"] is None else r["reset_events"][i],
                reused_baseline=bool(r.get("reused_baseline_fingerprint") and v == p["baseline"]),
                reused_control=v in r.get("reused_control_fingerprints", {}),
                batch_elapsed_seconds=r["batch_elapsed_seconds"],batch_function=r["function"],batch_seed=r["seed"]))
    if not rows:raise ValueError("No saved batches")
    pd.DataFrame(rows).to_csv(output/"runs.csv",index=False)
    pd.DataFrame([dict(function=r["function"],seed=r["seed"],status=r["status"],
        paths=len(r["variant_ids"]),evaluations_per_path=r["evaluations_per_path"],
        logical_evaluations=r["logical_evaluations"],
        executed_evaluations=r.get("executed_evaluations", r["logical_evaluations"]),
        objective_batch_calls=r["objective_batch_calls"],
        batch_elapsed_seconds=r["batch_elapsed_seconds"]) for r in records.values()]).to_csv(output/"batch_timings.csv",index=False)
    figures=output/"figures";figures.mkdir(exist_ok=True)
    curves_dir=figures/"convergence";curves_dir.mkdir(exist_ok=True)
    matrices=figures/"interactions";matrices.mkdir(exist_ok=True)
    summaries=[];curve_rows=[];paired=[];rank_rows=[];effects_rows=[];seed_effect_rows=[];interaction_seed_arrays=[]
    complete=[];gallery=[]
    relative_change = cfg.get("factorial_response") == "relative_objective_change"
    response_label = "relative objective change" if relative_change else "final/initial error fraction"
    contextual_pairs = []
    def line_plot(entries,curves,grid,title,filename,pdf,late=False):
        draw_curve(entries,curves,grid,title,filename,pdf,cfg,late)
    with PdfPages(figures/"convergence.pdf") as full_pdf, PdfPages(figures/"late_convergence_linear.pdf") as late_pdf, PdfPages(figures/"parameter_convergence.pdf") as sweep_pdf:
        for function in cfg["functions"]:
            rs=[records.get((function,seed)) for seed in cfg["seeds"]]
            if any(r is None or r["status"]!="budget_reached" or r["evaluations_per_path"]!=cfg["budget"] for r in rs):continue
            complete.append(function)
            grid=np.asarray(rs[0]["trace_evaluations"])
            if any(not np.array_equal(grid,r["trace_evaluations"]) for r in rs):raise ValueError("Inconsistent checkpoints")
            values=np.stack([r["trace_best"] for r in rs]) # seeds, checkpoints, variants
            final=np.array([r["best_scores"] for r in rs]);initial=np.array([r["initial_best"] for r in rs])
            curves={};stats={}
            displayed={e["variant"] for g,entries in p["groups"].items() if g not in ("factorial","curiosity_grid","openness_grid") for e in entries}
            if sweep_curves:displayed=set(identifiers)
            for i,v in enumerate(identifiers):
                m=np.median(values[:,:,i],axis=0);q1,q3=np.quantile(values[:,:,i],[.25,.75],axis=0)
                lo,hi=bootstrap_ci(values[:,:,i],draws) if v in displayed else (np.full(len(grid),np.nan),np.full(len(grid),np.nan))
                final_lo,final_hi=bootstrap_ci(final[:,i,None],draws)[:,0]
                stats[v]=dict(function=function,variant=v,label=p["variants"][v]["label"],n=len(rs),
                    mean=float(final[:,i].mean()),std=float(final[:,i].std(ddof=1)) if len(rs)>1 else 0.,
                    median=float(np.median(final[:,i])),q25=float(q1[-1]),q75=float(q3[-1]),
                    median_ci95_low=float(final_lo),median_ci95_high=float(final_hi),
                    mean_personality_resets=float(np.mean([r["reset_events"][i] for r in rs])))
                summaries.append(stats[v]);curves[v]=(m,lo,hi)
                curve_rows.extend(dict(function=function,variant=v,evaluations=int(x),median=float(y),
                    q25=float(a),q75=float(b),median_ci95_low=float(l),median_ci95_high=float(h))
                    for x,y,a,b,l,h in zip(grid,m,q1,q3,lo,hi))
            for group,entries in p["groups"].items():
                if group in ("factorial","curiosity_grid","openness_grid"):continue
                ref=entries[0]["variant"];ids=list(dict.fromkeys(e["variant"] for e in entries))
                ranking=pd.Series({v:stats[v]["median"] for v in ids}).rank()
                for e in entries:
                    v=e["variant"];rank_rows.append(dict(function=function,group=group,variant=v,label=e["label"],rank=ranking[v]))
                    difference=final[:,index[v]]-final[:,index[ref]]
                    low,high=bootstrap_ci(difference[:,None],draws)[:,0]
                    paired.append(dict(function=function,group=group,label=e["label"],reference=entries[0]["label"],
                        median_paired_difference=float(np.median(difference)),ci95_low=float(low),ci95_high=float(high),
                        wins=int((difference<0).sum()),ties=int((difference==0).sum()),losses=int((difference>0).sum())))
                filename=curves_dir/f"{function}_{group}.png"
                name=cfg.get("function_display_names", {}).get(function, function)
                title=f"{name} — {group}"
                line_plot(entries,curves,grid,title,filename,full_pdf)
                line_plot(entries,curves,grid,title+" (final 20%, linear)",curves_dir/f"{function}_{group}_late.png",late_pdf,True)
                gallery.append((title,filename.relative_to(output).as_posix()))
            # OAT response plots, with both personalization controls and seed uncertainty.
            fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
            sweeps=[("curiosity_grid","eta","alpha_mut",cfg["alpha_mut"],cfg["etas"]),
                    ("curiosity_grid","alpha_mut","eta",cfg["eta"],cfg["alphas"]),
                    ("openness_grid","theta","lam",cfg["lam"],cfg["thetas"]),
                    ("openness_grid","lam","theta",cfg["theta"],cfg["lambdas"])]
            any_sweep=False
            for ax,(group,param,fixed,level,candidates) in zip(axes.flat,sweeps):
                if group not in p["groups"]:ax.set_visible(False);continue
                any_sweep=True
                for mode in ("individual","shared"):
                    entries=[e for e in p["groups"][group] if e["settings"].get("personalization")==mode
                        and e["settings"].get(fixed)==level and e["settings"].get(param) in candidates]
                    entries.sort(key=lambda e:e["settings"][param])
                    if not entries:continue
                    x=[e["settings"][param] for e in entries]
                    mid=[stats[e["variant"]]["median"] for e in entries]
                    lo=[stats[e["variant"]]["median_ci95_low"] for e in entries];hi=[stats[e["variant"]]["median_ci95_high"] for e in entries]
                    ax.plot(x,mid,"o-",label=mode);ax.fill_between(x,lo,hi,alpha=.15)
                    if sweep_curves:
                        refs=p["groups"][group][:2 if group=="curiosity_grid" else 1]
                        line_plot(refs+entries,curves,grid,f"{function}: {param}, {mode}",
                            curves_dir/f"{function}_{param}_{mode}.png",sweep_pdf)
                baseline=p["baseline"]
                baseline_label = p["variants"][baseline]["label"]
                ax.axhline(stats[baseline]["median"],color="black",linestyle="--",label=f"Default {baseline_label} median" if cfg.get("full_hppso_sensitivity") else "PSO median")
                if param=="eta":ax.set_xscale("log")
                ax.set_yscale("linear")
                ax.set(title=f"Vary {param}; {fixed}={level:g}",xlabel=param,ylabel="Final objective error, linear (median; 95% CI)")
                ax.grid(alpha=.2);ax.legend(fontsize=8)
            if any_sweep:
                fig.suptitle(f"{function} — {cfg['dimension']}D, {cfg['budget']:,} FE/path; one-parameter slices",fontsize=12)
                file=figures/f"{function}_parameter_response.png";fig.savefig(file,dpi=150);fig.savefig(file.with_suffix(".pdf"))
                gallery.append((function+" parameters",file.relative_to(output).as_posix()))
            plt.close(fig)
            # The 16-arm design identifies all main, pair, triple and four-way contrasts.
            if len(p["factorial"]) == 16 and all(key in index for key in p["factorial"].values()):
                order=[index[p["factorial"][str(mask)]] for mask in range(16)]
                if relative_change:
                    reference = initial[:,order[0],None]
                    if np.any(reference == 0) or not np.allclose(initial[:,order], reference):
                        raise ValueError(f"Expected nonzero matched initial objectives for {function}")
                    response = (final[:,order] - reference) / np.abs(reference)
                    for a in range(4):
                        for b in range(a+1,4):
                            pair = (1<<a) | (1<<b)
                            names = f"{p['trait_order'][a]} + {p['trait_order'][b]}"
                            for other in range(16):
                                if other & pair:
                                    continue
                                values = response[:,other|pair] - response[:,other|(1<<a)] - response[:,other|(1<<b)] + response[:,other]
                                low, high = bootstrap_ci(values[:,None], draws, "mean")[:,0]
                                background = " + ".join(name for i,name in enumerate(p["trait_order"]) if other & (1<<i))
                                contextual_pairs.append(dict(function=function, pair=names, other_traits_mask=other,
                                                             other_traits_on=background or "None",
                                                             mean_contrast=values.mean(), ci95_low=low, ci95_high=high))
                else:
                    response = normalized_response(final[:,order],initial[:,order])
                contrasts=factorial_contrasts(response)
                interaction_seed_arrays.append(contrasts)
                bounds=bootstrap_ci(contrasts,draws,"mean");means=contrasts.mean(axis=0);matrix_effects={}
                for subset in range(1,16):
                    names=[name for i,name in enumerate(p["trait_order"]) if subset&(1<<i)]
                    effects_rows.append(dict(function=function,subset=subset,traits=" + ".join(names),order=len(names),
                        mean_contrast=means[subset-1],ci95_low=bounds[0,subset-1],ci95_high=bounds[1,subset-1],n_seeds=len(rs)))
                    matrix_effects[subset]=(means[subset-1],bounds[0,subset-1],bounds[1,subset-1])
                    seed_effect_rows.extend(dict(function=function,seed=seed,subset=subset,contrast=float(value))
                        for seed,value in zip(cfg["seeds"],contrasts[:,subset-1]))
                file=matrices/f"{function}.png";interaction_plot(matrix_effects,
                    f"{cfg.get('function_display_names', {}).get(function, function)} — {cfg['dimension']}D, {cfg['budget']:,} FE/path",
                    file,p["trait_order"],response_label)
                gallery.append((function+" interactions",file.relative_to(output).as_posix()))
    if interaction_seed_arrays:
        aggregate=np.mean(interaction_seed_arrays,axis=0)
        bounds=bootstrap_ci(aggregate,draws,"mean");means=aggregate.mean(axis=0);matrix_effects={}
        for subset in range(1,16):
            matrix_effects[subset]=(means[subset-1],bounds[0,subset-1],bounds[1,subset-1])
            effects_rows.append(dict(function="aggregate_fixed_functions",subset=subset,
                traits=" + ".join(n for i,n in enumerate(p["trait_order"]) if subset&(1<<i)),order=subset.bit_count(),
                mean_contrast=means[subset-1],ci95_low=bounds[0,subset-1],ci95_high=bounds[1,subset-1],n_seeds=len(cfg["seeds"])))
        interaction_plot(matrix_effects,f"{cfg['dimension']}D, {cfg['budget']:,} FE/path; mean of {len(interaction_seed_arrays)} fixed functions",
            matrices/"aggregate.png",p["trait_order"],response_label)
    pd.DataFrame(summaries).to_csv(output/"summary.csv",index=False)
    pd.DataFrame(curve_rows).to_csv(output/"convergence_summary.csv",index=False)
    pd.DataFrame(paired).to_csv(output/"paired_comparisons.csv",index=False)
    pd.DataFrame(rank_rows).to_csv(output/"group_ranks.csv",index=False)
    if rank_rows:pd.DataFrame(rank_rows).groupby(["group","label"])["rank"].mean().to_csv(output/"average_group_ranks.csv")
    pd.DataFrame(effects_rows).to_csv(output/"factorial_effects.csv",index=False)
    pd.DataFrame(seed_effect_rows).to_csv(output/"factorial_seed_contrasts.csv",index=False)
    if contextual_pairs:
        pd.DataFrame(contextual_pairs).to_csv(output/"pair_interactions_by_context.csv",index=False)
    # Long-form parameter results keep alias labels and disabled endpoints explicit.
    summary_lookup={(s["function"],s["variant"]):s for s in summaries}
    parameter_rows=[]
    for f in complete:
        for group in ("curiosity_grid","openness_grid"):
            for e in p["groups"].get(group,[]):
                parameter_rows.append(dict(summary_lookup[f,e["variant"]],group=group,setting_label=e["label"],**e["settings"]))
    pd.DataFrame(parameter_rows).to_csv(output/"parameter_results.csv",index=False)
    status=dict(saved_batches=len(records),expected_batches=len(jobs(cfg)),paths_per_batch=len(identifiers),
        valid_path_runs=sum(r["status"]=="budget_reached" and r["evaluations_per_path"]==cfg["budget"] for r in records.values())*len(identifiers),
        failed_batches=sum(r["status"]!="budget_reached" for r in records.values()),complete_functions=complete,
        budget_per_path=cfg["budget"],sweep_convergence_included=sweep_curves)
    (output/"report_status.json").write_text(json.dumps(status,indent=2))
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Trait experiments</title>'
        f'<h1>Trait experiments — {cfg["dimension"]}D; {cfg["budget"]:,} evaluations per path</h1>'
        f'<p>{cfg["population"]} particles · {len(cfg["seeds"])} matched seeds. Median and pointwise 95% seed-bootstrap intervals.</p>'
        '<p><a href="figures/convergence.pdf">Convergence PDF</a> · '
        '<a href="figures/late_convergence_linear.pdf">Linear late curves</a> · '
        '<a href="summary.csv">Summary</a> · <a href="paired_comparisons.csv">Paired comparisons</a> · '
        '<a href="group_ranks.csv">Descriptive ranks by function</a> · <a href="average_group_ranks.csv">Descriptive mean ranks</a> · '
        '<a href="runs.csv">Per-seed results</a></p>'
        + ('<p><a href="figures/parameter_convergence.pdf">Parameter-slice convergence</a></p>' if sweep_curves else '')
        +
        ('<p><a href="factorial_effects.csv">All factorial effects</a></p><h2>Aggregate pair interactions</h2><img src="figures/interactions/aggregate.png" style="max-width:900px;width:100%">' if interaction_seed_arrays else '')
        +''.join(f'<h2>{title}</h2><img loading="lazy" src="{file}" style="width:100%;max-width:1100px">' for title,file in gallery),
        encoding="utf-8")
    print(json.dumps(status,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--sweep-curves",action="store_true",help="Also render every OAT parameter-slice convergence curve")
    args=parser.parse_args();report(args.output,args.sweep_curves)
