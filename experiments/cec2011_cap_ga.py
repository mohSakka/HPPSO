"""Replay GA-MPC with a strict 200000-evaluation cap; preserve other trials."""
from experiments import cec2011_iterations as base
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading
import time
import numpy as np
import pandas as pd
from experiments.final_benchmark import read_result,write_result,shard_lease
from experiments.trait_plan import ROOT

class CappedObjective(base.LiveObjective):
    def __init__(self,f,job,output):
        super().__init__(f,job,output)
        self.budget=200000

def trial(job,output,digest):
    base.LiveObjective=CappedObjective
    result=base.trial(job,output,digest)
    row=read_result(base.path_for(output,job))
    if row['status']=='iterations_completed':
        assert row['evaluations']==200000
        source=ROOT/'experiments/results/cec2011_pop200_iter1000'
        previous=read_result(base.path_for(source,job))
        prefix=dict(previous['trace'])
        np.testing.assert_allclose(row['best_score'],prefix[200000],rtol=1e-12,atol=1e-10)
        row['status']='evaluation_budget_reached'
        row['termination']='200000 evaluation cap; native iteration maximum 1000'
        write_result(base.path_for(output,job),row)
    return row

def plot(output,number,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained');positive=True
    for algo in base.ALGORITHMS:
        rs=[r for r in rows if r['algorithm']==algo and r['status'] in ('iterations_completed','evaluation_budget_reached')]
        if len(rs)!=3:continue
        common=set(n for n,_ in rs[0]['trace'])
        for r in rs[1:]:common.intersection_update(n for n,_ in r['trace'])
        x=sorted(common);y=np.array([[dict(r['trace'])[n] for n in x] for r in rs])
        positive=positive and bool(np.all(y>0))
        line,=ax.plot(x,np.median(y,axis=0),label=algo)
        ax.fill_between(x,y.min(axis=0),y.max(axis=0),color=line.get_color(),alpha=.1)
    ax.set_xscale('log')
    if positive:ax.set_yscale('log')
    ax.set_xlabel('Objective evaluations');ax.set_ylabel('Best objective value');ax.grid(alpha=.25)
    ax.set_title(f'CEC2011 T{number}: population 200; GA-MPC capped at 200,000 evaluations\nOther algorithms: original 1000-iteration results; three seeds, median and min–max')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,fontsize=9)
    folder=output/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/f'T{number}.png',dpi=160);fig.savefig(folder/f'T{number}.pdf');plt.close(fig)

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'experiments/results/cec2011_pop200_iter1000')
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/results/cec2011_pop200_ga_cap200k')
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--port',type=int,default=8768)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(exist_ok=True,parents=True)
    original=json.loads((args.source/'plan.json').read_text());cfg=original['config']
    digest=hashlib.sha256((original['fingerprint']+Path(__file__).read_text()).encode()).hexdigest()
    plan=dict(original);plan['source_directory']=str(args.source.resolve())
    plan['algorithm_fingerprints']={a:(digest if a=='GA-MPC' else original['fingerprint']) for a in base.ALGORITHMS}
    plan['ga_evaluation_cap']=200000
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:base.atomic(out/'plan.json',plan)
    for f in (args.source/'trials').rglob('*.json.gz'):
        r=read_result(f)
        if r['algorithm']=='GA-MPC':continue
        dest=out/f.relative_to(args.source);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(f,dest)
    (out/'index.html').write_text(base.PAGE.replace('max="594"','max="66"').replace(
        'population 200; 1000 native optimizer iterations','population 200; GA-MPC capped at 200,000 evaluations; other results preserved'),encoding='utf-8')
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(SimpleHTTPRequestHandler,directory=str(out)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    finished=[]
    def progress(stage):
        done=len(list((out/'trials').glob('T*/GA-MPC/seed_*.json.gz')))
        base.atomic(out/'progress.json',dict(stage=stage,completed=done,total=66,failed=0,finished_problems=finished))
        active=[]
        for f in (out/'active').glob('*.json'):
            try:
                r=json.loads(f.read_text())
                if not r['done'] and time.time()-r['updated']<120:active.append(r)
            except (OSError,ValueError):pass
        base.atomic(out/'active.json',active)
    with shard_lease(out,'ga-cap'):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for meta in cfg['problems']:
                number=meta['problem'];pending=[]
                for seed in cfg['seeds']:
                    job=dict(problem=number,dimension=meta['dimension'],algorithm='GA-MPC',seed=seed,population=200,budget=200000,iterations=1000)
                    path=base.path_for(out,job)
                    if path.exists():
                        saved=read_result(path);assert saved['fingerprint']==digest and saved['evaluations']==200000
                    else:pending.append(job)
                futures={pool.submit(trial,j,str(out),digest):j for j in pending};progress('running')
                while futures:
                    ready,_=base.wait(futures,timeout=2,return_when=base.FIRST_COMPLETED)
                    for future in ready:
                        futures.pop(future);r=future.result()
                        if r['status']!='evaluation_budget_reached':raise RuntimeError(r['error'])
                        print(f'T{number} GA-MPC seed={r["seed"]}: {r["evaluations"]} evaluations, best={r["best_score"]}',flush=True)
                    progress('running')
                rows=[read_result(f) for f in (out/'trials'/f'T{number}').rglob('*.json.gz')]
                plot(out,number,rows);finished.append(number);progress('running')
        progress('ranking')
        from experiments.cec2011_iteration_rank_tables import main as rank
        import sys
        sys.argv=[sys.argv[0],'--output',str(out)];rank()
        progress('complete');print('Completed capped GA-MPC comparison.',flush=True)
    while True:time.sleep(30)

if __name__=='__main__':main()
