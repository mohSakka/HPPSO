"""Plot saved diversity observations over the first N optimizer iterations."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator,ScalarFormatter
from experiments.diversity_study import LABELS,METRICS,interval
from experiments.final_benchmark import read_result
from experiments.trait_study import result_path
from experiments.trait_plan import ROOT


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/results/diversity_zero_velocity13')
    parser.add_argument('--iterations',type=int,default=100)
    args=parser.parse_args()
    if args.iterations<1:raise ValueError('iterations must be positive')
    output=args.output;cfg=json.loads((output/'plan.json').read_text())['config']
    folder=output/'figures'/f'first_{args.iterations}_iterations';folder.mkdir(exist_ok=True)
    bootstrap=np.random.default_rng(20261107).integers(0,len(cfg['seeds']),(2000,len(cfg['seeds'])))
    gallery=[]
    with PdfPages(folder/'all_functions.pdf') as pdf:
        for function in cfg['functions']:
            records=[read_result(result_path(output,dict(function=function,seed=s))) for s in cfg['seeds']]
            iterations=np.asarray(records[0]['evaluations'])/cfg['population']-1
            mask=(iterations>=0)&(iterations<=args.iterations);x=iterations[mask]
            data=np.asarray([r['diversity'] for r in records])[:,mask]
            error=np.asarray([r['best_errors'] for r in records])[:,mask]
            fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
            for m,ax in enumerate(axes.flat):
                log_y = m < 3 or np.all(error > 0)
                for k,label in enumerate(LABELS):
                    y=data[:,:,k,m] if m<3 else error[:,:,k]
                    med=np.median(y,axis=0);lo,hi=interval(y,bootstrap)
                    line,=ax.plot(x,np.ma.masked_less_equal(med,0) if log_y else med,label=label,lw=1.6)
                    ax.fill_between(x,np.ma.masked_less_equal(lo,0) if log_y else lo,np.ma.masked_less_equal(hi,0) if log_y else hi,
                                    color=line.get_color(),alpha=.16)
                ax.set_title(METRICS[m].replace('_',' ').capitalize() if m<3 else 'Best-so-far objective')
                ax.set_ylabel('Normalized distance' if m<3 else 'Objective value')
                ax.set_yscale('log' if log_y else 'linear')
                # Shifted logarithm includes the shared initialization at iteration zero.
                ax.set_xscale('function',functions=(np.log1p,np.expm1))
                ax.set_xlim(0,args.iterations)
                ticks=[t for t in [0,1,2,5,10,20,50,100,200,500,1000] if t<=args.iterations]
                ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.set_xlabel('Iteration (log(1 + iteration) spacing)');ax.grid(alpha=.25)
            axes[0,0].legend(fontsize=9)
            title=cfg.get('function_display_names', {}).get(function, function)
            fig.suptitle(f'{title} · First {args.iterations} iterations · {cfg["dimension"]}D · '
                         f'{len(records)} matched seeds\nMedian and 95% seed-bootstrap intervals')
            filename=folder/(function+'.png');fig.savefig(filename,dpi=160);pdf.savefig(fig);plt.close(fig)
            gallery.append(filename)
    (folder/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><h1>First {args.iterations} iterations</h1>'
        '<p>Shifted logarithmic x-axis includes initialization at iteration 0. '
        'Diversity y-axes are logarithmic and omit zero values. The objective axis is linear when nonpositive values occur. '
        'Iteration t corresponds to population × (t + 1) evaluations. '
        'Only recorded checkpoints are shown.</p>'
        '<p><a href="all_functions.pdf">All functions PDF</a></p>'+
        ''.join(f'<h2>{cfg.get("function_display_names", {}).get(p.stem, p.stem)}</h2><img loading="lazy" src="{p.name}" style="width:100%;max-width:1200px">' for p in gallery), encoding='utf-8')
    index=output/'index.html';s=index.read_text(encoding='utf-8');link=f'figures/{folder.name}/index.html'
    if link not in s:s=s.replace('<h1>',f'<p><a href="{link}">First {args.iterations} iterations</a></p><h1>',1)
    index.write_text(s, encoding='utf-8')
    print('Saved '+str(folder),flush=True)


if __name__=='__main__':main()
