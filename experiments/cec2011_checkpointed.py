"""Resumable native CEC2011 benchmark with live progress and per-problem plots."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from functools import lru_cache,partial
import hashlib
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import threading
import time
import traceback
import numpy as np
from experiments.final_benchmark import StrictObjective,BudgetComplete,write_result as original_write,read_result,shard_lease
from experiments.trait_plan import ROOT
from hppso.experiments.runners import DEFAULT_ALGORITHMS
from experiments.optimizer_checkpoint import CheckpointObjective

ALGORITHMS=[a for a in DEFAULT_ALGORITHMS if a!='Sep-CMA-ES']


def write_result(path,data):
    for attempt in range(200):
        try:return original_write(path,data)
        except PermissionError:
            if attempt==199:raise
            time.sleep(.05)


def ga_budget_only(objective,dim,lo,hi,population,ignored_iterations,seed):
    import random
    from hppso.algorithms.ga_mpc import run_ga_mpc
    random.seed(seed);np.random.seed(seed)
    return run_ga_mpc(objective,dim,lo,hi,ps=population,max_gen=None)


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,allow_nan=False),encoding='utf-8')
    for _ in range(50):
        try:os.replace(tmp,path);return
        except PermissionError:time.sleep(.01)
    # A concurrent Windows HTTP reader may hold the previous snapshot open.
    # Keep that snapshot until the next refresh; telemetry must not stop a trial.
    tmp.unlink(missing_ok=True)


@lru_cache(maxsize=22)
def problem(number):
    from minionpy.cec import CEC2011Functions
    f=CEC2011Functions(number);bounds=np.asarray(f.get_bounds(),dtype=float)
    if number==3:bounds=np.array([[0.6,0.9]])  # Original CEC2011 report, pp. 7/33.
    if bounds.shape!=(f.dimension,2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:,0]>bounds[:,1]):
        raise ValueError('Invalid CEC2011 metadata')
    return f,bounds[:,0],bounds[:,1]


def path_for(output,job):
    return Path(output)/'trials'/f'T{job["problem"]}'/job['algorithm'].replace(' ','_')/f'seed_{job["seed"]:03d}.json.gz'



def trial(job,output,digest):
    f,lo,hi=problem(job['problem']);objective=CheckpointObjective(lambda x:float(f.evaluate(np.asarray(x).reshape(1,-1))[0]),job,
        Path(output)/'checkpoints'/f'T{job["problem"]}'/job['algorithm'].replace(' ','_')/f'seed_{job["seed"]:03d}',digest)
    objective.active=Path(output)/'active'/f'worker_{os.getpid()}.json'
    objective.active.parent.mkdir(exist_ok=True);objective.publish()
    started=time.perf_counter();status='failed';error=None
    try:
        args=(objective,job['dimension'],lo,hi,job['population'],2**31-1)
        if job['algorithm']=='GA-MPC':
            ga_budget_only(*args,job['seed'])
        elif job['algorithm']=='CMA-ES':
            DEFAULT_ALGORITHMS[job['algorithm']](*args,2**62,job['seed'],stopfitness=-np.inf)
        else:DEFAULT_ALGORITHMS[job['algorithm']](*args,job['seed'])
        raise RuntimeError('Optimizer returned before consuming its evaluation budget')
    except BudgetComplete:
        if np.isfinite(objective.best):status='budget_reached'
        else:error='No finite candidate found within the evaluation budget'
    except Exception:error=traceback.format_exc()
    if objective.count and (not objective.trace or objective.trace[-1][0]!=objective.count) and np.isfinite(objective.best):
        objective.trace.append([objective.count,objective.best])
    row=dict(job,status=status,error=error,evaluations=objective.count,
        best_score=objective.best if np.isfinite(objective.best) else None,
        best_position=None if objective.best_position is None else objective.best_position.tolist(),
        trace=objective.trace,nonfinite_candidates=objective.nonfinite_candidates,
        replayed_evaluations=min(objective.count,len(objective.cached)),new_objective_evaluations=objective.new_evaluations,
        checkpoint_directory=str(objective.folder),checkpoint_method='numerical snapshot plus verified fitness replay',
        elapsed_scope='this invocation, including replay and checkpoint IO; not cumulative across restarts',
        elapsed_seconds=time.perf_counter()-started,fingerprint=digest)
    objective.close()
    path=path_for(output,job)
    if path.exists():
        archive=path.parent/'attempts';archive.mkdir(exist_ok=True)
        shutil.copyfile(path,archive/(path.name+f'.{time.time_ns()}'))
    write_result(path,row);objective.publish(done=True)
    return {k:v for k,v in row.items() if k not in ('trace','best_position')}


def plot_problem(output,number,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(output)/'figures';folder.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    scores=[]
    for algo in ALGORITHMS:
        records=[read_result(path_for(output,r)) for r in rows if r['algorithm']==algo and r['status']=='budget_reached']
        if len(records)!=3:continue
        common=set(n for n,_ in records[0]['trace'])
        for r in records[1:]:common.intersection_update(n for n,_ in r['trace'])
        x=np.array(sorted(common))
        if len(x)==0:continue
        y=np.array([[dict(r['trace'])[n] for n in x] for r in records]);scores.append(y)
        line,=ax.plot(x,np.median(y,axis=0),label=algo,lw=1.5)
        ax.fill_between(x,y.min(axis=0),y.max(axis=0),color=line.get_color(),alpha=.1)
    ax.set_xscale('log');ax.set_xlabel('Objective evaluations (log scale)');ax.set_ylabel('Best objective value')
    if scores and all(np.all(y>0) for y in scores):ax.set_yscale('log')
    ax.grid(alpha=.25);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3,fontsize=9)
    complete=sum(r['status']=='budget_reached' for r in rows)
    ax.set_title(f'CEC2011 T{number}: D={rows[0]["dimension"]}, population 200, budget {rows[0]["budget"]:,}\n'
                 f'Median and min–max over seeds 0–2; {complete}/27 valid trials')
    fig.savefig(folder/f'T{number}.png',dpi=160);fig.savefig(folder/f'T{number}.pdf');plt.close(fig)


PAGE='''<!doctype html><meta charset="utf-8"><title>CEC2011 live experiment</title>
<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:12px}table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}progress{width:100%}</style>
<h1>CEC2011 live progress</h1><p>Nine algorithms; seeds 0–2; population 200; checkpointed equal-budget run; target evaluations shown in progress. Sep-CMA-ES excluded.</p>
<p id="status">Loading…</p><progress id="bar" max="594"></progress><h2>Active trials</h2><table><tbody id="active"></tbody></table>
<p><a href="exports/">Raw trial summary CSV</a> · <a href="tables/index.html">Final ranks and statistics</a></p><div id="plots"></div>
<script>let shown=''; async function update(){try{let p=await(await fetch('progress.json?t='+Date.now())).json();
document.querySelector('#status').textContent=p.stage+': '+p.completed+'/'+p.total+' trials; '+p.failed+' failed; '+p.finished_problems.length+'/22 problems finished';
document.querySelector('#bar').value=p.completed;
let a=await(await fetch('active.json?t='+Date.now())).json();document.querySelector('#active').innerHTML='<tr><th>Problem</th><th>Algorithm</th><th>Seed</th><th>Evaluations</th><th>Best value</th><th>Non-finite candidates</th><th>Replayed (cached)</th><th>New calls</th></tr>'+a.map(r=>`<tr><td>T${r.problem}</td><td>${r.algorithm}</td><td>${r.seed}</td><td>${r.evaluations}/${r.budget}</td><td>${r.best_score??'initializing'}</td><td>${r.nonfinite_candidates}</td><td>${r.replayed_evaluations??0}</td><td>${r.new_evaluations??0}</td></tr>`).join('');
let key=p.finished_problems.join(',');if(key!==shown){shown=key;document.querySelector('#plots').innerHTML=p.finished_problems.map(n=>`<h2>T${n}</h2><a href="figures/T${n}.pdf">PDF</a><img src="figures/T${n}.png">`).join('')}}catch(e){document.querySelector('#status').textContent='Refreshing saved progress…'}}update();setInterval(update,3000)</script>'''


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/results/cec2011_pop200_fe400k_checkpointed')
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--port',type=int,default=8774)
    parser.add_argument('--budget',type=int,default=400000)
    parser.add_argument('--schedule-horizon',type=int,default=None)
    args=parser.parse_args()
    if args.workers<1:raise ValueError('workers must be positive')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.budget<200:raise ValueError('Budget must include the population')
    previous=None
    if (out/'plan.json').exists():
        try:previous=json.loads((out/'plan.json').read_text())
        except (ValueError,OSError):previous=json.loads((out/'plan.backup.json').read_text())
    horizon=args.schedule_horizon if args.schedule_horizon is not None else previous['config']['schedule_horizon'] if previous else args.budget
    if horizon<1:raise ValueError('Schedule horizon must be positive')
    metadata=[]
    for i in range(1,23):
        f,lo,hi=problem(i);metadata.append(dict(problem=i,dimension=f.dimension,lower=lo.tolist(),upper=hi.tolist(),budget=args.budget))
    cfg=dict(problems=metadata,algorithms=ALGORITHMS,seeds=[0,1,2],population=200,evaluation_budget=args.budget,stopping_rule='objective evaluations only',schedule_horizon=horizon,checkpoint_method='numeric snapshots and verified evaluation replay',minionpy=importlib.metadata.version('minionpy'))
    immutable=json.loads(json.dumps(cfg));immutable.pop('evaluation_budget')
    for p in immutable['problems']:p.pop('budget')
    digest=hashlib.sha256(json.dumps(immutable,sort_keys=True).encode())
    sources=[Path(__file__),ROOT/'experiments/optimizer_checkpoint.py',ROOT/'experiments/final_benchmark.py',ROOT/'src/hppso/experiments/runners.py']+sorted((ROOT/'src/hppso/algorithms').glob('*.py'))
    import minionpy
    sources+=sorted(Path(minionpy.__file__).parent.glob('*.py'))+sorted(Path(minionpy.__file__).parent.glob('*.pyd'))
    for path in sources:digest.update(path.name.encode());digest.update(path.read_bytes())
    digest=digest.hexdigest();plan=dict(config=cfg,fingerprint=digest,python=platform.python_version(),numpy=np.__version__)
    if previous:
        if previous['fingerprint']!=digest or previous['python']!=plan['python'] or previous['numpy']!=plan['numpy']:
            raise ValueError('Checkpoint source/software/settings changed; choose a fresh output')
        if args.budget<previous['config']['evaluation_budget']:raise ValueError('Cannot reduce a checkpointed run budget')
        plan['budget_history']=previous.get('budget_history',[previous['config']['evaluation_budget']])
        if plan['budget_history'][-1]!=args.budget:plan['budget_history'].append(args.budget)
    else:plan['budget_history']=[args.budget]
    for plan_name in ['plan.backup.json','plan.json']:
        destination=out/plan_name;temp=out/(plan_name+'.tmp')
        with temp.open('w') as f:json.dump(plan,f);f.flush();os.fsync(f.fileno())
        from experiments.optimizer_checkpoint import replace_retry
        replace_retry(temp,destination)
    (out/'index.html').write_text(PAGE,encoding='utf-8')
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(SimpleHTTPRequestHandler,directory=str(out)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    print(f'Live progress: http://127.0.0.1:{args.port}/',flush=True)
    import pandas as pd
    rows=[];finished=[]
    def publish(stage):
        atomic(out/'progress.json',dict(stage=stage,completed=len(rows),total=594,
            failed=sum(r['status']!='budget_reached' for r in rows),finished_problems=finished,updated=time.time()))
        active=[]
        for path in (out/'active').glob('*.json'):
            try:
                r=json.loads(path.read_text())
                if not r['done'] and time.time()-r['updated']<120:active.append(r)
            except (OSError,ValueError):pass
        atomic(out/'active.json',active)
    with shard_lease(out,'cec2011'):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for meta in metadata:
                number=meta['problem'];pending=[];block=[]
                for algo in ALGORITHMS:
                    for seed in cfg['seeds']:
                        job=dict(problem=number,dimension=meta['dimension'],algorithm=algo,seed=seed,population=200,budget=meta['budget'],schedule_horizon=horizon)
                        path=path_for(out,job)
                        if path.exists():
                            r=read_result(path)
                            if r['fingerprint']!=digest:raise ValueError('Mixed trial fingerprint')
                            if r['status']=='budget_reached' and r['evaluations']==job['budget']:
                                rows.append(r);block.append(r)
                            else:pending.append(job)
                        else:pending.append(job)
                print(f'T{number}: D={meta["dimension"]}, budget={meta["budget"]}; {len(pending)} pending trials',flush=True)
                iterator=iter(pending);active={}
                def fill():
                    while len(active)<2*args.workers:
                        job=next(iterator,None)
                        if job is None:break
                        active[pool.submit(trial,job,str(out),digest)]=job
                fill();publish('running')
                while active:
                    ready,_=wait(active,timeout=2,return_when=FIRST_COMPLETED)
                    for future in ready:
                        job=active.pop(future);r=future.result();rows.append(r);block.append(r)
                        print(f'[{len(rows)}/594] T{number} {r["algorithm"]} seed={r["seed"]}: {r["status"]}, FE={r["evaluations"]}, best={r["best_score"]}',flush=True)
                        if r['error']:print(r['error'],flush=True)
                    fill();publish('running')
                from experiments.cec2011_log_curves import render as render_log
                render_log(out,number,plan,exclude_initialization=True);finished.append(number)
                df=pd.DataFrame([{k:v for k,v in r.items() if k not in ('trace','best_position')} for r in rows]);exports=out/'exports';exports.mkdir(exist_ok=True)
                version=str(time.time_ns());df.to_csv(exports/f'runs_T{number}_{version}.csv',index=False)
                df[df.status=='budget_reached'].groupby(['problem','algorithm']).best_score.agg(['count','mean','std','median','min','max']).to_csv(exports/f'summary_T{number}_{version}.csv')
                publish('running');print(f'T{number} finished; convergence plot saved.',flush=True)
        publish('reporting')
        from experiments.cec2011_pop200_equal_budget_tables import generate
        generate(out)
        publish('complete');print('All CEC2011 trials finished.',flush=True)
    print('Progress page remains available; Ctrl+C closes it.',flush=True)
    try:
        while True:time.sleep(30)
    except KeyboardInterrupt:server.shutdown()


if __name__=='__main__':main()
