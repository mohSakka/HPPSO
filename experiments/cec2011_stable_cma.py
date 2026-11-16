"""Replay all CMA-ES paths after numerical stabilization; preserve other paths."""
from experiments import cec2011_iterations as base
from experiments.cec2011_cap_ga import plot
from experiments.trait_plan import ROOT
from experiments.final_benchmark import read_result,write_result,shard_lease
from hppso.algorithms.cmaes import CMAES
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import random
import shutil
import sys
import threading
import time
import numpy as np

DIAGNOSTICS={}
def monitored(objective,dim,lo,hi,population,iterations,maxevals,seed,*,stopfitness=None):
    random.seed(seed);np.random.seed(seed)
    width=np.asarray(hi)-np.asarray(lo);sigma=.25*float(np.mean(width[width>0]))
    opt=CMAES(objective,dim,lo,hi,sigma,iterations,population,maxevals);opt.stopfitness=-np.inf
    try:
        with np.errstate(invalid='raise',divide='raise',over='raise'):return opt.optimize()[1:]
    finally:
        DIAGNOSTICS.clear();DIAGNOSTICS.update(covariance_repairs=opt.covariance_repairs,
            sigma_clips=opt.sigma_clips,final_sigma=opt.sigma,
            covariance_finite=bool(np.isfinite(opt.C).all()))

def trial(job,out,digest):
    base.DEFAULT_ALGORITHMS['CMA-ES']=monitored
    base.trial(job,out,digest);r=read_result(base.path_for(out,job));r['stability_diagnostics']=dict(DIAGNOSTICS)
    if r['status']=='iterations_completed':assert r['evaluations']==200000
    write_result(base.path_for(out,job),r);return r

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    source=ROOT/'experiments/results/cec2011_pop200_ga_cap200k_t3corrected'
    out=ROOT/'experiments/results/cec2011_pop200_stable_cma';out.mkdir(exist_ok=True)
    original=json.loads((source/'plan.json').read_text())
    digest=hashlib.sha256(Path(__file__).read_bytes()+Path(base.__file__).read_bytes()+
        (ROOT/'src/hppso/algorithms/cmaes.py').read_bytes()+original['fingerprint'].encode()).hexdigest()
    plan=json.loads(json.dumps(original));plan['problem_algorithm_fingerprints']={f'{i}:CMA-ES':digest for i in range(1,23)}
    plan['cma_stabilization']=dict(condition_limit=1e14,sigma_log_bounds_relative_to_box=[-32,14],
        rerun='all 66 CMA-ES trials',source_directory=str(source))
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:base.atomic(out/'plan.json',plan)
    for f in (source/'trials').rglob('*.json.gz'):
        if read_result(f)['algorithm']=='CMA-ES':continue
        dest=out/f.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(f,dest)
    (out/'figures').mkdir(exist_ok=True)
    for f in (source/'figures').glob('*'):
        if f.is_file():shutil.copyfile(f,out/'figures'/f.name)
    page=base.PAGE.replace('max="594"','max="66"').replace('<h1>',
        '<p><a href="tables/index.html">Updated ranks</a> · <a href="tables/minimum_table.html">Updated minimum table</a></p><h1>',1)
    page=page.replace('population 200; 1000 native optimizer iterations',
        'population 200; stabilized CMA-ES replay; T3 [0.6,0.9]; GA-MPC cap 200,000')
    (out/'index.html').write_text(page,encoding='utf-8')
    server=ThreadingHTTPServer(('127.0.0.1',8770),partial(SimpleHTTPRequestHandler,directory=str(out)))
    threading.Thread(target=server.serve_forever,daemon=True).start();finished=[]
    def progress(stage):
        files=list((out/'trials').glob('T*/CMA-ES/seed_*.json.gz'))
        base.atomic(out/'progress.json',dict(stage=stage,completed=len(files),total=66,
            failed=sum(read_result(f)['status']=='failed' for f in files),finished_problems=finished))
        active=[]
        for f in (out/'active').glob('*.json'):
            try:
                r=json.loads(f.read_text())
                if not r['done'] and time.time()-r['updated']<120:active.append(r)
            except (OSError,ValueError):pass
        base.atomic(out/'active.json',active)
    order=[8,13]+[n for n in range(1,23) if n not in (8,13)]
    with shard_lease(out,'stable-cma'):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for number in order:
                meta=next(p for p in plan['config']['problems'] if p['problem']==number);futures={}
                for seed in [0,1,2]:
                    job=dict(problem=number,dimension=meta['dimension'],algorithm='CMA-ES',seed=seed,population=200,budget=None,iterations=1000)
                    path=base.path_for(out,job)
                    if path.exists():assert read_result(path)['fingerprint']==digest
                    else:futures[pool.submit(trial,job,str(out),digest)]=job
                progress('running')
                while futures:
                    ready,_=base.wait(futures,timeout=2,return_when=base.FIRST_COMPLETED)
                    for future in ready:
                        futures.pop(future);r=future.result()
                        print(f'T{number} seed={r["seed"]}: {r["status"]}, FE={r["evaluations"]}, diagnostics={r["stability_diagnostics"]}',flush=True)
                        if r['error']:print(r['error'],flush=True)
                    progress('running')
                rows=[read_result(f) for f in (out/'trials'/f'T{number}').rglob('*.json.gz')]
                plot(out,number,rows);finished.append(number);progress('running')
        progress('reporting')
        from experiments.cec2011_iteration_rank_tables import main as ranks
        from experiments.cec2011_minimum_tables import main as minimum
        sys.argv=[sys.argv[0],'--output',str(out)];ranks();minimum();progress('complete')
    while True:time.sleep(30)

if __name__=='__main__':main()
