"""Render saved convergence statistics without rerunning optimization or bootstrap."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def draw_curve(entries,curves,grid,title,filename,pdf,cfg,late=False):
    fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True)
    used=set();view=slice(None);limits=[]
    if late:view=slice(max(0,int(np.searchsorted(grid,.8*cfg["budget"]))-1),None)
    for entry in entries:
        v=entry["variant"]
        if v in used:continue
        used.add(v);m,lo,hi=curves[v]
        ax.step(grid[view],m[view],where="post",label=entry["label"],linewidth=1.5)
        color=ax.lines[-1].get_color();ax.fill_between(grid[view],lo[view],hi[view],step="post",color=color,alpha=.12)
        values=np.concatenate([m[view],lo[view],hi[view]])
        values=values[np.isfinite(values)]
        if len(values):limits.extend([values.min(),values.max()])
    low,high=min(limits),max(limits)
    ax.set_xlim(.8*cfg["budget"] if late else 0,cfg["budget"])
    if low<0 or late:
        margin=.05*max(high-low,abs(high)*.01,1e-12)
        ax.set_ylim(low-margin if low<0 else max(0.,low-margin),high+margin)
    else:
        if low>0:
            ax.set_yscale("log");ax.set_ylim(low/1.15,high*1.15)
        else:
            ax.set_yscale("symlog",linthresh=1e-12)
            ax.set_ylim(min(0.,low*1.15),max(high*1.15,1e-12))
    ax.set(title=title,xlabel="Objective evaluations per path",ylabel="Best-so-far objective value")
    ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2,loc="upper center",bbox_to_anchor=(.5,-.16))
    fig.suptitle(f"{cfg['dimension']}D; {len(cfg['seeds'])} matched seeds; median and pointwise 95% bootstrap CI",fontsize=10)
    fig.savefig(filename,dpi=150);pdf.savefig(fig);plt.close(fig)


def refresh(output):
    output=Path(output);plan=json.loads((output/"plan.json").read_text());cfg=plan["config"];p=plan["protocol"]
    data=pd.read_csv(output/"convergence_summary.csv")
    folder=output/"figures";curvedir=folder/"convergence"
    with PdfPages(folder/"convergence.pdf") as full, PdfPages(folder/"late_convergence_linear.pdf") as late, PdfPages(folder/"parameter_convergence.pdf") as sweeps:
        for function in data.function.unique():
            subset=data[data.function.eq(function)]
            curves={};grid=None
            for v,frame in subset.groupby("variant",sort=False):
                frame=frame.sort_values("evaluations")
                if grid is None:grid=frame.evaluations.to_numpy()
                else:np.testing.assert_array_equal(frame.evaluations.to_numpy(),grid)
                curves[v]=tuple(frame[column].to_numpy() for column in ("median","median_ci95_low","median_ci95_high"))
            for group,entries in p["groups"].items():
                if group in ("factorial","curiosity_grid","openness_grid"):continue
                name=cfg.get("function_display_names", {}).get(function, function)
                title=f"{name} — {group}"
                draw_curve(entries,curves,grid,title,curvedir/f"{function}_{group}.png",full,cfg)
                draw_curve(entries,curves,grid,title+" (final 20%, linear)",curvedir/f"{function}_{group}_late.png",late,cfg,True)
            for group,param,fixed,level,candidates in (
                ("curiosity_grid","eta","alpha_mut",cfg["alpha_mut"],cfg["etas"]),
                ("curiosity_grid","alpha_mut","eta",cfg["eta"],cfg["alphas"]),
                ("openness_grid","theta","lam",cfg["lam"],cfg["thetas"]),
                ("openness_grid","lam","theta",cfg["theta"],cfg["lambdas"])):
                if group not in p["groups"]:continue
                for mode in ("individual","shared"):
                    entries=[e for e in p["groups"][group] if e["settings"].get("personalization")==mode
                        and e["settings"].get(fixed)==level and e["settings"].get(param) in candidates]
                    entries.sort(key=lambda e:e["settings"][param])
                    refs=p["groups"][group][:2 if group=="curiosity_grid" else 1]
                    if entries and all(np.isfinite(curves[e["variant"]][1]).all() for e in entries):
                        draw_curve(refs+entries,curves,grid,f"{function}: {param}, {mode}",
                                   curvedir/f"{function}_{param}_{mode}.png",sweeps,cfg)
    print("Refreshed full, late and parameter-slice curves from saved statistics.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--output",type=Path,required=True)
    refresh(parser.parse_args().output)
