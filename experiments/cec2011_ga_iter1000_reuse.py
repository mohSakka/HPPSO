"""Replay GA-MPC with fitness reuse and exactly 1000 generations."""
from experiments import cec2011_iterations as base
from hppso.algorithms.ga_mpc import run_ga_mpc
import random
from experiments.trait_plan import ROOT
from experiments.final_benchmark import read_result,write_result as original_write,shard_lease
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import numpy as np

def write_result(path,data):
    for attempt in range(200):
        try:return original_write(path,data)
        except PermissionError:
            if attempt==199:raise
            time.sleep(.05)

def trial(job,out,digest):
    base.write_result=write_result
    base.trial(job,out,digest);r=read_result(base.path_for(out,job))
    if r['status']=='iterations_completed':
        r['fitness_reuse']=True;r['termination']='1000 generations'
    write_result(base.path_for(out,job),r);return r

def plot(output,number,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained');positive=True;magnitudes=[]
    for algo in base.ALGORITHMS:
        rs=[r for r in rows if r['algorithm']==algo and r['status'] in ('iterations_completed','evaluation_budget_reached')]
        if len(rs)!=3:continue
        common=set(n for n,_ in rs[0]['trace'])
        for r in rs[1:]:common.intersection_update(n for n,_ in r['trace'])
        x=sorted(common);y=np.array([[dict(r['trace'])[n] for n in x] for r in rs])
        positive=positive and bool(np.all(y>0));magnitudes.extend(np.abs(y).ravel())
        line,=ax.plot(x,np.median(y,axis=0),label=algo)
        ax.fill_between(x,y.min(axis=0),y.max(axis=0),color=line.get_color(),alpha=.1)
    ax.set_xscale('log')
    if positive:ax.set_yscale('log')
    else:ax.set_yscale('symlog',linthresh=max(max(magnitudes,default=1)*1e-6,1e-12))
    ax.set_xlabel('Objective evaluations');ax.set_ylabel('Best objective value');ax.grid(alpha=.25)
    ax.set_title(f'CEC2011 T{number}: population 200; 1000 iterations for all algorithms\nGA-MPC fitness reuse; stabilized CMA-ES; three seeds, median and min–max')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,fontsize=9)
    folder=output/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/f'T{number}.png',dpi=160);fig.savefig(folder/f'T{number}.pdf');plt.close(fig)

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    source=ROOT/'experiments/results/cec2011_pop200_stable_cma'
    out=ROOT/'experiments/results/cec2011_pop200_iter1000_fitness_reuse';out.mkdir(exist_ok=True)
    original=json.loads((source/'plan.json').read_text())
    digest=hashlib.sha256(Path(__file__).read_bytes()+Path(base.__file__).read_bytes()+
        (ROOT/'src/hppso/algorithms/ga_mpc.py').read_bytes()+original['fingerprint'].encode()).hexdigest()
    plan=json.loads(json.dumps(original));plan.setdefault('problem_algorithm_fingerprints',{}).update({f'{i}:GA-MPC':digest for i in range(1,23)})
    plan.pop('ga_evaluation_cap',None)
    plan['ga_fitness_reuse']=dict(rerun='all 66 GA-MPC trials',stopping_rule='1000 generations; no evaluation cap',source_directory=str(source))
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:base.atomic(out/'plan.json',plan)
    for f in (source/'trials').rglob('*.json.gz'):
        record=read_result(f)
        if record['algorithm']=='GA-MPC':continue
        assert record['population']==200 and record['iterations']==1000 and record['status']=='iterations_completed'
        dest=out/f.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(f,dest)
    (out/'figures').mkdir(exist_ok=True)
    for f in (source/'figures').glob('*'):
        if f.is_file():shutil.copyfile(f,out/'figures'/f.name)
    page=base.PAGE.replace('max="594"','max="66"').replace('<h1>',
        '<p><a href="tables/index.html">Updated ranks</a> · <a href="tables/minimum_table.html">Updated minimum table</a></p><h1>',1)
    page=page.replace('population 200; 1000 native optimizer iterations',
        'population 200; 1000 iterations for all algorithms; GA-MPC fitness reuse; T3 [0.6,0.9]')
    (out/'index.html').write_text(page,encoding='utf-8')
    server=ThreadingHTTPServer(('127.0.0.1',8773),partial(SimpleHTTPRequestHandler,directory=str(out)))
    threading.Thread(target=server.serve_forever,daemon=True).start();finished=[]
    def progress(stage):
        files=list((out/'trials').glob('T*/GA-MPC/seed_*.json.gz'))
        base.atomic(out/'progress.json',dict(stage=stage,completed=len(files),total=66,
            failed=sum(read_result(f)['status']=='failed' for f in files),finished_problems=finished))
        active=[]
        for f in (out/'active').glob('*.json'):
            try:
                r=json.loads(f.read_text())
                if not r['done'] and time.time()-r['updated']<120:active.append(r)
            except (OSError,ValueError):pass
        base.atomic(out/'active.json',active)
    order=list(range(1,23))
    with shard_lease(out,'ga-1000-generations'):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for number in order:
                meta=next(p for p in plan['config']['problems'] if p['problem']==number);futures={}
                for seed in [0,1,2]:
                    job=dict(problem=number,dimension=meta['dimension'],algorithm='GA-MPC',seed=seed,population=200,budget=None,iterations=1000)
                    path=base.path_for(out,job)
                    if path.exists():assert read_result(path)['fingerprint']==digest
                    else:futures[pool.submit(trial,job,str(out),digest)]=job
                progress('running')
                while futures:
                    ready,_=base.wait(futures,timeout=2,return_when=base.FIRST_COMPLETED)
                    for future in ready:
                        futures.pop(future);r=future.result()
                        print(f'T{number} seed={r["seed"]}: {r["status"]}, FE={r["evaluations"]}, termination={r.get("termination")}',flush=True)
                        if r['error']:print(r['error'],flush=True)
                    progress('running')
                rows=[read_result(f) for f in (out/'trials'/f'T{number}').rglob('*.json.gz')]
                plot(out,number,rows);finished.append(number);progress('running')
        progress('reporting')
        from experiments.cec2011_iteration_rank_tables import main as ranks
        from experiments.cec2011_minimum_tables import main as minimum
        sys.argv=[sys.argv[0],'--output',str(out)];ranks();minimum()
        from experiments.cec2011_wilcoxon import main as wilcoxon
        wilcoxon();progress('complete')
    while True:time.sleep(30)

if __name__=='__main__':main()
