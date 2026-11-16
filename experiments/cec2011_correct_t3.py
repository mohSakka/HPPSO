"""Correct T3 to the original [0.6,0.9] domain; replay only its 27 trials."""
from experiments import cec2011_iterations as base
from experiments.cec2011_cap_ga import plot
from experiments.trait_plan import ROOT
from experiments.final_benchmark import read_result,write_result,shard_lease
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

class CorrectedObjective(base.LiveObjective):
    def __init__(self,f,job,output):
        super().__init__(f,job,output)
        if job['algorithm']=='GA-MPC':self.budget=200000
    def __call__(self,x):
        value=float(np.asarray(x).reshape(-1)[0])
        if not .6-1e-12<=value<=.9+1e-12:raise ValueError('T3 evaluated outside corrected bounds')
        return super().__call__(x)

def trial(job,out,digest):
    base.LiveObjective=CorrectedObjective
    base.trial(job,out,digest)
    r=read_result(base.path_for(out,job))
    r['search_bounds']=[[.6,.9]]
    if r['status']=='iterations_completed':
        assert .6<=r['best_position'][0]<=.9
        if job['algorithm']=='GA-MPC':
            assert r['evaluations']==200000;r['status']='evaluation_budget_reached'
    write_result(base.path_for(out,job),r)
    return r

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    source=ROOT/'experiments/results/cec2011_pop200_ga_cap200k'
    out=ROOT/'experiments/results/cec2011_pop200_ga_cap200k_t3corrected';out.mkdir(exist_ok=True)
    original=json.loads((source/'plan.json').read_text())
    digest=hashlib.sha256(Path(__file__).read_bytes()+Path(base.__file__).read_bytes()+original['fingerprint'].encode()).hexdigest()
    plan=json.loads(json.dumps(original));plan['problem_fingerprints']={'3':digest}
    plan['bounds_correction']=dict(problem=3,previous=[-.6,.9],correct=[.6,.9],
        source='https://al-roomi.org/multimedia/CEC_Database/CEC2011/CEC2011_TechnicalReport.pdf',pages=[7,33])
    for p in plan['config']['problems']:
        if p['problem']==3:p['lower']=[.6];p['upper']=[.9]
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:base.atomic(out/'plan.json',plan)
    for f in (source/'trials').rglob('*.json.gz'):
        if read_result(f)['problem']==3:continue
        dest=out/f.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(f,dest)
    (out/'figures').mkdir(exist_ok=True)
    for f in (source/'figures').glob('*'):
        if f.is_file() and f.stem!='T3':shutil.copyfile(f,out/'figures'/f.name)
    page=base.PAGE.replace('max="594"','max="27"').replace(
        'population 200; 1000 native optimizer iterations',
        'population 200; T3 corrected to [0.6,0.9]; GA-MPC capped at 200,000 evaluations')
    page=page.replace('<h1>','<p><a href="tables/index.html">Updated ranks</a> · <a href="tables/minimum_table.html">Updated minimum table</a></p><h1>',1)
    (out/'index.html').write_text(page,encoding='utf-8')
    server=ThreadingHTTPServer(('127.0.0.1',8769),partial(SimpleHTTPRequestHandler,directory=str(out)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    finished=[]
    def progress(stage):
        files=list((out/'trials'/'T3').rglob('*.json.gz'))
        base.atomic(out/'progress.json',dict(stage=stage,completed=len(files),total=27,
            failed=sum(read_result(f)['status']=='failed' for f in files),finished_problems=finished))
        active=[]
        for f in (out/'active').glob('*.json'):
            try:
                r=json.loads(f.read_text())
                if not r['done'] and time.time()-r['updated']<120:active.append(r)
            except (OSError,ValueError):pass
        base.atomic(out/'active.json',active)
    with shard_lease(out,'t3-correction'):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={}
            for algo in base.ALGORITHMS:
                for seed in [0,1,2]:
                    job=dict(problem=3,dimension=1,algorithm=algo,seed=seed,population=200,
                        budget=200000 if algo=='GA-MPC' else None,iterations=1000)
                    if base.path_for(out,job).exists():
                        assert read_result(base.path_for(out,job))['fingerprint']==digest
                    else:futures[pool.submit(trial,job,str(out),digest)]=job
            progress('running')
            while futures:
                ready,_=base.wait(futures,timeout=2,return_when=base.FIRST_COMPLETED)
                for future in ready:
                    futures.pop(future);r=future.result()
                    print(r['algorithm'],r['seed'],r['status'],r['best_score'],flush=True)
                progress('running')
        rows=[read_result(f) for f in (out/'trials'/'T3').rglob('*.json.gz')]
        plot(out,3,rows);finished.extend(range(1,23));progress('reporting')
        from experiments.cec2011_iteration_rank_tables import main as ranks
        from experiments.cec2011_minimum_tables import main as minimum
        sys.argv=[sys.argv[0],'--output',str(out)];ranks();minimum();progress('complete')
    while True:time.sleep(30)

if __name__=='__main__':main()
