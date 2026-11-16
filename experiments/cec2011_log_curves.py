"""Render saved CEC2011 traces with logarithmic/symmetric-log y axes."""
import argparse
import gzip
import json
import os
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def publish(fig,path):
    tmp=path.with_name(path.stem+'.render_tmp'+path.suffix)
    fig.savefig(tmp,dpi=160)
    for attempt in range(100):
        try:os.replace(tmp,path);return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.05)


def render(output,number,plan,exclude_initialization=False):
    records=[]
    for path in (output/'trials'/f'T{number}').rglob('*.json.gz'):
        with gzip.open(path,'rt') as stream:r=json.load(stream)
        if r['status']=='budget_reached':records.append(r)
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained');values=[]
    for algo in plan['config']['algorithms']:
        rs=[r for r in records if r['algorithm']==algo]
        if len(rs)!=3:continue
        traces=[dict(r['trace']) for r in rs]
        common=set(traces[0])
        for trace in traces[1:]:common.intersection_update(trace)
        population=plan['config']['population']
        x=sorted(n for n in common if not exclude_initialization or n>=2*population)
        if not x:continue
        y=np.array([[trace[n] for n in x] for trace in traces]);values.append(y.ravel())
        display_x=np.asarray(x)-(population if exclude_initialization else 0)
        line,=ax.plot(display_x,np.median(y,axis=0),label=algo,lw=1.5)
        ax.fill_between(display_x,y.min(axis=0),y.max(axis=0),color=line.get_color(),alpha=.1)
    all_values=np.concatenate(values) if values else np.array([1.])
    if np.all(all_values>0):ax.set_yscale('log');scale='log';threshold=None
    else:
        absolute=np.abs(all_values);positive=absolute[absolute>0];maximum=max(float(absolute.max()),1e-15)
        threshold=float(np.clip(positive.min()/10 if len(positive) else maximum*1e-6,maximum*1e-12,maximum*1e-3))
        ax.set_yscale('symlog',linthresh=threshold);scale='symlog'
    ax.set_xscale('log');ax.set_xlabel('Objective evaluations after initialization (log scale)' if exclude_initialization else 'Objective evaluations (log scale)')
    ax.set_ylabel('Best objective value ('+('log scale' if scale=='log' else f'symmetric log; linear within ±{threshold:.2g}')+')')
    ax.grid(alpha=.25);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3,fontsize=9)
    cfg=plan['config'];problem=next(p for p in cfg['problems'] if p['problem']==number)
    caption=f'Median and min–max over seeds 0–2; {len(records)}/27 valid trials'
    if exclude_initialization:caption+='; initialization hidden'
    ax.set_title(f'CEC2011 T{number}: D={problem["dimension"]}, population {cfg["population"]}, budget {problem["budget"]:,}\n'+caption)
    folder=output/'figures';folder.mkdir(exist_ok=True)
    publish(fig,folder/f'T{number}.png');publish(fig,folder/f'T{number}.pdf');plt.close(fig)
    return dict(y_scale=scale,linthresh=threshold,exclude_initialization=exclude_initialization,
                total_evaluation_cutoff=2*cfg['population'] if exclude_initialization else 1,
                x_offset=cfg['population'] if exclude_initialization else 0)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results/cec2011_pop90_fe150k')
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--exclude-initialization',action='store_true');args=parser.parse_args()
    plan=json.loads((args.output/'plan.json').read_text());rendered={}
    while True:
        try:progress=json.loads((args.output/'progress.json').read_text())
        except (OSError,ValueError):time.sleep(2);continue
        for n in progress['finished_problems']:
            if str(n) in rendered:continue
            rendered[str(n)]=render(args.output,n,plan,args.exclude_initialization)
            (args.output/'log_plot_settings.json').write_text(json.dumps(dict(
                raw_values_unchanged=True,optimization_rerun=False,problems=rendered),indent=2))
            print(f'T{n}: {rendered[str(n)]}',flush=True)
        if not args.watch or progress['stage']=='complete':break
        time.sleep(3)


if __name__=='__main__':main()
