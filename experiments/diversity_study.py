"""Matched four-algorithm diversity trajectories; read-only instrumentation."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import html
import json
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from scipy.spatial.distance import pdist
from experiments.trait_plan import ROOT, build_plan, load_config
from experiments.trait_study import fingerprint, jobs, result_path
from experiments.trait_functions import trait_suite
from experiments.trait_batch import BatchFunction, BatchBudget, BatchedTraitPSO
from experiments.final_benchmark import read_result, write_result, shard_lease

LABELS=('HPPSO','PSO','PSO-m','PSO-RIW')
METRICS=('dispersion','mean_pairwise_distance','mean_step_distance')


def measure(positions, width, previous=None):
    """Distances divided by box widths and sqrt(D); no random draws."""
    z=positions/np.asarray(width)
    centered=z-z.mean(axis=1,keepdims=True)
    dispersion=np.sqrt(np.mean(centered**2,axis=(1,2)))
    pairwise=np.array([pdist(block).mean()/np.sqrt(block.shape[1]) for block in z])
    step=np.zeros(len(z)) if previous is None else np.linalg.norm(
        (positions-previous)/width,axis=2).mean(axis=1)/np.sqrt(z.shape[-1])
    return np.stack((dispersion,pairwise,step),axis=1)


def protocol(cfg):
    p=build_plan(cfg)
    if cfg.get('factorial_pso_control'):
        entries = {entry['label']: entry['variant'] for entry in p['groups']['hp_ablation']}
        keys = [entries[label] for label in LABELS]
        return keys, [p['variants'][key]['spec'] for key in keys]
    keys=[p['factorial']['15'],p['baseline'],
          next(e['variant'] for e in p['groups']['curiosity'] if e['label']=='PSO-m'),
          next(e['variant'] for e in p['groups']['aggressiveness'] if e['label']=='PSO-RIW')]
    return keys,[p['variants'][key]['spec'] for key in keys]


def trial(job,cfg,output,digest,reference):
    keys,specs=protocol(cfg)
    problem=trait_suite(cfg['dimension'],cfg['instance_seed'])[job['function']]
    tracker=BatchBudget(BatchFunction(problem),cfg['budget'],4,cfg['dimension'],cfg['trace_points'])
    engine=BatchedTraitPSO(tracker,problem['bounds'],cfg['population'],job['seed'],specs,cfg)
    # Sample only complete evaluated populations, including the final generation.
    generations=int(np.ceil(cfg['budget']/cfg['population']))
    sample_generations=np.unique(np.r_[1,np.geomspace(1,generations,100).astype(int),
                                       np.linspace(1,generations,201).astype(int)])
    targets=set((sample_generations*cfg['population']).clip(max=cfg['budget']).tolist())
    counts=[tracker.count];values=[measure(engine.positions,engine.width)]
    objective=[tracker.best.copy()];started=time.perf_counter()
    while tracker.count<tracker.budget:
        previous=engine.positions.copy()
        engine.step()
        if tracker.count in targets or tracker.count==tracker.budget:
            counts.append(tracker.count);values.append(measure(engine.positions,engine.width,previous))
            objective.append(tracker.best.copy())
    values=np.asarray(values);objective=np.asarray(objective)
    assert np.isfinite(values).all() and (values>=0).all()
    assert np.all(np.diff(objective,axis=0)<=1e-8)
    assert np.all(engine.positions>=engine.lower) and np.all(engine.positions<=engine.upper)
    old=read_result(result_path(reference,job));order=[old['variant_ids'].index(k) for k in keys]
    np.testing.assert_allclose(tracker.best,np.asarray(old['best_scores'])[order],rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(tracker.best_positions,np.asarray(old['best_positions'])[order],rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(values[0,:,0],np.full(4,values[0,0,0]),rtol=0,atol=0)
    result=dict(job,status='budget_reached',fingerprint=digest,labels=LABELS,variant_ids=keys,
        metrics=METRICS,evaluations=counts,diversity=values.tolist(),best_errors=objective.tolist(),
        best_positions=tracker.best_positions.tolist(),evaluations_per_path=tracker.count,
        initial_velocities_zero=True,reference_ablation_matched=True,
        batch_elapsed_seconds=time.perf_counter()-started)
    write_result(result_path(output,job),result)
    return job


def interval(data,indices):
    return np.quantile(np.median(data[indices],axis=1),[.025,.975],axis=0)


def report(output, integrated=False):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    output=Path(output);plan=json.loads((output/'plan.json').read_text());cfg=plan['config']
    labels=("HPPSO","PSO") if integrated else LABELS
    if integrated:
        source=output
        output=output/"diversity"
        output.mkdir(exist_ok=True)
    else:
        source=output
    folder=output/'figures';folder.mkdir(exist_ok=True)
    runs=[];trajectory=[];paired=[];summary=[];gallery=[];checked=0
    bootstrap=np.random.default_rng(20261107).integers(0,len(cfg['seeds']),(2000,len(cfg['seeds'])))
    early_folder=folder/'first_100_iterations'
    early_folder.mkdir(exist_ok=True)
    with PdfPages(folder/'diversity_trajectories.pdf') as pdf, PdfPages(early_folder/'all_functions.pdf') as early_pdf:
        for function in cfg['functions']:
            records=[read_result(result_path(source,dict(function=function,seed=s))) for s in cfg['seeds']]
            assert all(r['fingerprint']==plan['fingerprint'] and r['status']=='budget_reached' and
                       r['evaluations_per_path']==cfg['budget'] for r in records)
            if integrated:
                records=[r["diversity_record"] for r in records]
                assert all(tuple(r["labels"])==labels for r in records)
            else:
                assert all(r['reference_ablation_matched'] for r in records)
            checked+=len(records)
            x=np.asarray(records[0]['evaluations'])
            assert all(r['evaluations']==x.tolist() for r in records)
            data=np.asarray([r['diversity'] for r in records])
            errors=np.asarray([r['best_errors'] for r in records])
            iterations=x/cfg['population']-1
            fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
            for metric,ax in zip(range(3),axes.flat):
                for k,label in enumerate(labels):
                    y=data[:,:,k,metric];med=np.median(y,axis=0);lo,hi=interval(y,bootstrap)
                    line,=ax.plot(iterations,np.ma.masked_less_equal(med,0),label=label)
                    ax.fill_between(iterations,np.ma.masked_less_equal(lo,0),
                                    np.ma.masked_less_equal(hi,0),color=line.get_color(),alpha=.16)
                    for t,n in enumerate(x):
                        trajectory.append(dict(function=function,algorithm=label,metric=METRICS[metric],
                            evaluations=n,median=med[t],ci95_low=lo[t],ci95_high=hi[t]))
                ax.set_title(METRICS[metric].replace('_',' ').capitalize())
                ax.set_ylabel('Normalized distance')
                ax.set_yscale('log')
                ax.grid(alpha=.25)
            ax=axes[1,1]
            for k,label in enumerate(labels):
                y=errors[:,:,k];med=np.median(y,axis=0);lo,hi=interval(y,bootstrap)
                line,=ax.plot(iterations,med,label=label)
                ax.fill_between(iterations,lo,hi,color=line.get_color(),alpha=.16)
            if np.all(errors>0):ax.set_yscale('log')
            elif np.any(errors>=0):ax.set_yscale('symlog',linthresh=max(float(np.abs(errors).max())*1e-10,1e-12))
            ax.set_title('Best-so-far objective');ax.grid(alpha=.25)
            for panel in axes.flat:
                panel.set_xscale('symlog',linthresh=1)
                panel.set_xlim(0,iterations[-1])
                panel.set_xlabel('Swarm iteration (0 = initialization)')
            axes[0,0].legend(fontsize=9)
            title = cfg.get('function_display_names', {}).get(function, function)
            fig.suptitle(f'{title} · {cfg["dimension"]}D · {len(records)} matched seeds · {cfg["budget"]:,} evaluations\n'
                         'Median and 95% seed-bootstrap intervals · log diversity axes')
            filename=folder/(function+'.png')
            fig.savefig(filename,dpi=160)
            pdf.savefig(fig)
            # Save the same curves and y-axis limits with a shorter x-axis.
            for panel in axes.flat:
                panel.set_xlim(0,min(100,iterations[-1]))
            fig.savefig(early_folder/filename.name,dpi=160)
            early_pdf.savefig(fig)
            plt.close(fig)
            gallery.append(filename)
            # Integrals start after the common evaluated initialization, and are divided by interval length.
            late=x>=.8*cfg['budget']
            for k,label in enumerate(labels):
                for s,seed in enumerate(cfg['seeds']):
                    row=dict(function=function,algorithm=label,seed=seed,final_objective=errors[s,-1,k])
                    for m,name in enumerate(METRICS):
                        y=data[s,:,k,m]
                        row[name+'_final']=y[-1]
                        row[name+'_auc']=np.trapz(y,x)/(x[-1]-x[0])
                        row[name+'_late']=np.trapz(y[late],x[late])/(x[late][-1]-x[late][0])
                        early_end = min(x[-1], (100 + 1) * cfg['population'])
                        early = x < early_end
                        row[name+'_first100']=np.trapz(np.r_[y[early],np.interp(early_end,x,y)],np.r_[x[early],early_end])/(early_end-x[0])
                    runs.append(row)
    df=pd.DataFrame(runs);df.to_csv(output/'runs.csv',index=False)
    for function in cfg['functions']:
        block=df[df.function==function]
        for label in labels:
            rows=block[block.algorithm==label].sort_values('seed')
            for metric in [c for c in df.columns if c not in ('function','algorithm','seed')]:
                y=rows[metric].to_numpy();lo,hi=interval(y,bootstrap)
                summary.append(dict(function=function,algorithm=label,metric=metric,n=len(y),
                    median=np.median(y),mean=np.mean(y),std=np.std(y,ddof=1),ci95_low=lo,ci95_high=hi))
                if label!='HPPSO':
                    hp=block[block.algorithm=='HPPSO'].sort_values('seed')[metric].to_numpy()
                    diff=hp-y;low,high=interval(diff,bootstrap)
                    paired.append(dict(function=function,comparator=label,metric=metric,
                        median_paired_hp_minus_comparator=np.median(diff),ci95_low=low,ci95_high=high,
                        hp_greater=int((diff>0).sum()),hp_lower=int((diff<0).sum()),ties=int((diff==0).sum())))
    pd.DataFrame(summary).to_csv(output/'summary.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'paired_comparisons.csv',index=False)
    pd.DataFrame(trajectory).to_csv(output/'trajectory_summary.csv',index=False)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Swarm diversity</h1>'
        f'<p>{cfg["dimension"]}D · {cfg["population"]} particles · {len(cfg["seeds"])} matched seeds · {cfg["budget"]:,} evaluations. Median and 95% bootstrap intervals. '
        'The iteration axis includes initialization at 0 and uses a symmetric log scale. '
        'Diversity axes are logarithmic and omit zero distances.</p>'
        '<p>Distances use current particles and are normalized by coordinate bounds and sqrt(D). '
        'Higher diversity alone does not establish better optimization or prevention of premature convergence. '
        'Intervals are unadjusted and conditional on these fixed function instances.</p>'
        '<p><a href="figures/diversity_trajectories.pdf">All figures PDF</a> · '
        '<a href="figures/first_100_iterations/index.html">First 100 iterations</a> · '
        '<a href="summary.csv">Summary</a> · <a href="paired_comparisons.csv">Paired differences</a> · '
        '<a href="runs.csv">Per-seed summaries</a></p>'+
        ''.join('<h2>'+html.escape(cfg.get('function_display_names', {}).get(p.stem, p.stem))+'</h2><img style="width:100%;max-width:1200px" loading="lazy" src="'+p.relative_to(output).as_posix()+'">' for p in gallery), encoding='utf-8')
    (early_folder/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><h1>First 100 iterations</h1>'
        '<p>The same curves, aggregation, and axis scales as the full-run figures, zoomed to iterations 0–100. '
        'Initialization is iteration 0. Logarithmic diversity axes omit zero distances.</p>'
        '<p><a href="all_functions.pdf">All functions PDF</a> · <a href="../../index.html">Full-run figures</a></p>'
        + ''.join(f'<h2>{html.escape(cfg.get("function_display_names", {}).get(p.stem, p.stem))}</h2>'
                  f'<img style="width:100%;max-width:1200px" loading="lazy" src="{p.name}">' for p in gallery),
        encoding='utf-8')
    audit=dict(completed_batches=checked,path_runs=len(labels)*checked,exact_budgets=True,
        measured_in_ablation=integrated,
        all_final_scores_and_positions_match_corrected_ablation=True,
        finite_nonnegative_diversity=True,matched_initial_populations=True,
        metrics_do_not_consume_rng_or_objective_evaluations=True)
    (output/'audit.json').write_text(json.dumps(audit,indent=2))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/configs/trait_study_zero_velocity13.json')
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/results/diversity_zero_velocity13')
    parser.add_argument('--reference',type=Path,default=ROOT/'experiments/results/trait_study_zero_velocity13')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--integrated',action='store_true',help='Report two-path diversity recorded during the ablation')
    args=parser.parse_args()
    if args.integrated:report(args.output,integrated=True);return
    if args.report:report(args.output);return
    cfg=load_config(args.config)
    if cfg['budget']%cfg['population']:raise ValueError('Diversity study requires complete final generations')
    if args.workers<1:raise ValueError('workers must be positive')
    reference_plan=json.loads((args.reference/'plan.json').read_text())
    if reference_plan['config']!=cfg:raise ValueError('Reference and diversity study scientific configurations differ')
    digest=hashlib.sha256((fingerprint(cfg)+Path(__file__).read_text().replace('\r\n','\n')).encode()).hexdigest()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    plan=dict(config=cfg,fingerprint=digest,reference=str(args.reference.resolve()),labels=LABELS,
        metric_names=METRICS,format_version=1)
    # JSON normalizes tuples to lists before comparison.
    plan=json.loads(json.dumps(plan))
    if (output/'plan.json').exists():
        if json.loads((output/'plan.json').read_text())!=plan:raise ValueError('Source/config changed; choose fresh output')
    else:(output/'plan.json').write_text(json.dumps(plan,indent=2))
    with shard_lease(output,'diversity'):
        (output/'problem_definitions.json').write_bytes((args.reference/'problem_definitions.json').read_bytes())
        (output/'reference_plan.json').write_text(json.dumps(reference_plan,indent=2))
        def stage(name,done=0):
            (output/'progress.json').write_text(json.dumps(dict(stage=name,completed=done,total=len(jobs(cfg)),
                updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))))
        pending=[]
        for job in jobs(cfg):
            path=result_path(output,job)
            if path.exists():
                saved=read_result(path)
                if saved['fingerprint']!=digest or saved['status']!='budget_reached':raise ValueError('Invalid saved result')
            else:pending.append(job)
        done=len(jobs(cfg))-len(pending);stage('optimizing',done)
        print(f'{done}/{len(jobs(cfg))} saved; {len(pending)} pending; four paths per batch',flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            iterator=iter(pending);active={}
            def fill():
                while len(active)<2*args.workers:
                    job=next(iterator,None)
                    if job is None:break
                    active[pool.submit(trial,job,cfg,str(output),digest,str(args.reference))]=job
            fill()
            while active:
                ready,_=wait(active,return_when=FIRST_COMPLETED)
                for future in ready:
                    job=active.pop(future);future.result();done+=1;stage('optimizing',done)
                    print(f'[{done}/{len(jobs(cfg))}] {job}: recorded; ablation match verified',flush=True)
                fill()
        stage('reporting',done);report(output)
        stage('packaging',done)
        archive=ROOT/'artifacts'/(output.name+'_results.zip')
        with ZipFile(archive,'w',ZIP_DEFLATED) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file() and not p.name.endswith(('.lock','.log')) and p.name!='progress.json':
                    z.write(p,output.name+'/'+p.relative_to(output).as_posix())
        with ZipFile(archive) as z:assert z.testzip() is None
        stage('complete',done);print('Complete: '+str(output),flush=True)


if __name__=='__main__':main()
