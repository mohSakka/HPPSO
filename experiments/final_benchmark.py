"""Portable, resumable, strictly evaluation-budgeted final benchmark."""
import os
# Set before importing NumPy; one BLAS thread per independent worker.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
import scipy

from hppso.benchmarks.suite import BENCHMARK_NAMES, SUITE_VERSION, build_benchmark_suite
from hppso.benchmarks.classical import problem_definitions
from hppso.experiments.runners import DEFAULT_ALGORITHMS

ROOT = Path(__file__).resolve().parents[1]


class BudgetComplete(BaseException):
    """Control signal deliberately outside ordinary algorithm exceptions."""


class StrictObjective:
    def __init__(self, function, budget, points=1000):
        self.function, self.budget = function, int(budget)
        self.count, self.best = 0, float("inf")
        self.best_position = None
        self.trace = []
        # Log-spaced early checkpoints plus uniform late checkpoints.
        n=min(points,self.budget)
        self.checkpoints=set(np.unique(np.r_[np.geomspace(1,self.budget,n).astype(int),
                                             np.linspace(1,self.budget,n).astype(int)]).tolist())

    def evaluation_progress(self):
        return self.count/self.budget

    def __call__(self, x):
        if self.count >= self.budget:
            raise BudgetComplete
        self.count += 1
        value=float(self.function(x))
        if not np.isfinite(value):
            raise FloatingPointError(f"Non-finite objective at evaluation {self.count}")
        if value < self.best:
            self.best=value
            self.best_position=np.asarray(x,dtype=float).copy()
        if self.count in self.checkpoints:
            self.trace.append([self.count,self.best])
        if self.count == self.budget:
            # Save the final candidate, then interrupt before post-budget updates.
            raise BudgetComplete
        return value


@lru_cache(maxsize=8)
def suite(dim, seed):
    state=np.random.get_state()
    try:
        np.random.seed(seed+dim)
        return {p["name"]:p for p in build_benchmark_suite(
            dim,michalewicz_dimension=None,additional_seed=seed)}
    finally:
        np.random.set_state(state)


def algorithms(dim, cfg=None):
    if cfg and "algorithms" in cfg:
        return cfg["algorithms"]
    return [a for a in DEFAULT_ALGORITHMS if a != ("Sep-CMA-ES" if dim==30 else "CMA-ES")]


def load_config(path):
    cfg=json.loads(Path(path).read_text())
    if "algorithms" in cfg and (not cfg["algorithms"] or len(set(cfg["algorithms"])) != len(cfg["algorithms"]) or set(cfg["algorithms"])-set(DEFAULT_ALGORITHMS)):
        raise ValueError("Algorithms must be distinct registered names")
    if not cfg["dimensions"] or len(set(cfg["dimensions"]))!=len(cfg["dimensions"]):
        raise ValueError("Dimensions must be unique and nonempty")
    if len(set(cfg["seeds"]))!=len(cfg["seeds"]) or not cfg["seeds"]:
        raise ValueError("Seeds must be unique and nonempty")
    for d in cfg["dimensions"]:
        p=cfg["populations"][str(d)]
        if not isinstance(d,int) or d<2 or not isinstance(p,int) or p<5 or cfg["evaluations_per_dimension"]*d<p:
            raise ValueError("Invalid dimension/population/budget")
    if int(cfg["evaluations_per_dimension"])!=cfg["evaluations_per_dimension"] or cfg["evaluations_per_dimension"]<=0:
        raise ValueError("Evaluation multiplier must be a positive integer")
    if cfg["trace_points"]<1:
        raise ValueError("trace_points must be positive")
    return cfg


def jobs(cfg):
    for dim in cfg["dimensions"]:
        for name in BENCHMARK_NAMES:
            for algo in algorithms(dim,cfg):
                for seed in cfg["seeds"]:
                    yield dict(dimension=dim,function=name,algorithm=algo,seed=seed,
                               population=cfg["populations"][str(dim)],
                               budget=int(cfg["evaluations_per_dimension"]*dim))


def result_path(output, job):
    return Path(output)/"trials"/str(job["dimension"])/job["function"]/job["algorithm"].replace(" ","_")/f"seed_{job['seed']:03d}.json.gz"


def read_result(path):
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        return json.load(stream)


def write_result(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f".{os.getpid()}.tmp")
    with gzip.open(temp,"wt",encoding="utf-8") as stream:
        json.dump(data,stream,allow_nan=False)
    os.replace(temp,path)


def trial(job,cfg,output,fingerprint):
    problem=suite(job["dimension"],cfg["instance_seed"])[job["function"]]
    objective=StrictObjective(problem["function"],job["budget"],cfg["trace_points"])
    started=time.perf_counter()
    status,error="failed",None
    try:
        # Each iteration makes at least one evaluation; this guard is unreachable
        # before the objective's budget interrupt. It is not a stopping criterion.
        guard=job["budget"]+1
        args=(objective,job["dimension"],*problem["bounds"],job["population"],guard)
        runner=DEFAULT_ALGORITHMS[job["algorithm"]]
        if job["algorithm"] in ("CMA-ES","Sep-CMA-ES"):
            runner(*args,job["budget"]+1,job["seed"],stopfitness=-np.inf)
        else:
            runner(*args,job["seed"])
        raise RuntimeError("Algorithm returned before consuming its evaluation budget")
    except BudgetComplete:
        status="budget_reached"
    except Exception:
        error=traceback.format_exc()
    elapsed=time.perf_counter()-started
    if objective.count and (not objective.trace or objective.trace[-1][0]!=objective.count):
        if np.isfinite(objective.best):
            objective.trace.append([objective.count,objective.best])
    data=dict(job,status=status,error=error,evaluations=objective.count,
              best_score=objective.best if np.isfinite(objective.best) else None,
              best_position=objective.best_position.tolist() if objective.best_position is not None else None,
              elapsed_seconds=elapsed,trace=objective.trace,fingerprint=fingerprint,
              objective_scale=problem["objective_scale"],suite_version=SUITE_VERSION,host=platform.node(),finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    path=result_path(output,job)
    if path.exists():
        old=read_result(path)
        archive=path.parent/"attempts"/(path.name+f".{time.time_ns()}")
        archive.parent.mkdir(exist_ok=True)
        shutil.copyfile(path,archive)
        data["attempt"]=old.get("attempt",1)+1
    else:
        data["attempt"]=1
    write_result(path,data)
    return {k:v for k,v in data.items() if k not in ("trace","best_position")}


def source_hash(cfg):
    paths=sorted((ROOT/"src/hppso/algorithms").glob("*.py"))+sorted((ROOT/"src/hppso/benchmarks").glob("*.py"))
    paths += [ROOT/"src/hppso/experiments/runners.py",Path(__file__)]
    h=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode())
    for path in paths:
        h.update(path.name.encode());h.update(path.read_bytes().replace(b"\r\n",b"\n"))
    return h.hexdigest()


@contextmanager
def shard_lease(output, index):
    path=output/f".shard-{index}.lock"
    try:
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Shard already locked: {path}. After a killed job, verify no old worker remains before removing this lock.")
    try:
        with os.fdopen(fd,"w") as stream:
            json.dump(dict(host=platform.node(),pid=os.getpid(),started=time.time()),stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def run(args):
    cfg=load_config(args.config)
    if args.shards<1 or not 0<=args.shard<args.shards or args.workers<1:
        raise ValueError("Invalid workers or shard selection")
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    fingerprint=source_hash(cfg)
    plan=dict(config=cfg,suite_version=SUITE_VERSION,fingerprint=fingerprint,
              shards=args.shards,functions=list(BENCHMARK_NAMES),algorithms={str(d):algorithms(d,cfg) for d in cfg["dimensions"]})
    plan_path=output/"plan.json"
    # Unique temp + hard link publishes a complete plan exactly once, even when
    # independent SLURM shards start simultaneously on the same shared directory.
    temp=output/f".plan-{os.getpid()}-{time.time_ns()}.tmp"
    temp.write_text(json.dumps(plan,indent=2),encoding="utf-8")
    try:
        try: os.link(temp,plan_path)
        except FileExistsError: pass
    finally: temp.unlink(missing_ok=True)
    if json.loads(plan_path.read_text())!=plan:
        raise ValueError("Config/source mismatch with existing run; use a fresh output directory")
    with shard_lease(output,args.shard):
        env=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),host=platform.node())
        (output/f"environment-shard-{args.shard}.json").write_text(json.dumps(env,indent=2))
        definitions=output/"definitions";definitions.mkdir(exist_ok=True)
        for dim in cfg["dimensions"]:
            data=problem_definitions(list(suite(dim,cfg["instance_seed"]).values()))
            file=definitions/f"{dim}d.json"
            tmp=file.with_name(file.name+f".{os.getpid()}-{time.time_ns()}.tmp")
            tmp.write_text(json.dumps(data),encoding="utf-8");os.replace(tmp,file)
        selected=[j for i,j in enumerate(jobs(cfg)) if i%args.shards==args.shard]
        pending=[];skipped=0
        for j in selected:
            file=result_path(output,j)
            if file.exists():
                r=read_result(file)
                if r["fingerprint"]!=fingerprint: raise ValueError("Trial fingerprint mismatch")
                if not args.retry_failed or r["status"]=="budget_reached":
                    skipped+=1;continue
            pending.append(j)
        print(f"Shard {args.shard}/{args.shards}: {len(selected)} assigned; {skipped} saved; {len(pending)} pending",flush=True)
        progress=output/f"progress-shard-{args.shard}.json"
        done=skipped;failed=0
        def publish():
            progress.write_text(json.dumps(dict(assigned=len(selected),completed=done,failed_this_invocation=failed,updated=time.time())))
        publish()
        iterator=iter(pending)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            active={}
            def fill():
                while len(active)<2*args.workers:
                    j=next(iterator,None)
                    if j is None:break
                    active[pool.submit(trial,j,cfg,str(output),fingerprint)]=j
            fill()
            while active:
                ready,_=wait(active,return_when=FIRST_COMPLETED)
                for future in ready:
                    job=active.pop(future)
                    # Worker infrastructure crashes propagate: saved trials survive;
                    # restart the shard to resume. Do not invent scientific outcomes.
                    r=future.result();done+=1;failed+=r["status"]!="budget_reached"
                    print(f"[{done}/{len(selected)}] {job['dimension']}D {job['function']} {job['algorithm']} seed={job['seed']} {r['status']} FE={r['evaluations']}/{job['budget']} best={r['best_score']} time={r['elapsed_seconds']:.2f}s",flush=True)
                publish();fill()
        print("Shard finished. Run --report when all shards are finished.",flush=True)


def report(output):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    output=Path(output)
    plan=json.loads((output/"plan.json").read_text());cfg=plan["config"]
    rows=[];traces={}
    for j in jobs(cfg):
        file=result_path(output,j)
        if not file.exists():continue
        r=read_result(file)
        if r["fingerprint"]!=plan["fingerprint"]:raise ValueError("Mixed trial fingerprints")
        rows.append({k:v for k,v in r.items() if k not in ("trace","best_position")})
        traces[(r["dimension"],r["function"],r["algorithm"],r["seed"])]=r
    if not rows:raise ValueError("No saved trials")
    df=pd.DataFrame(rows);df.to_csv(output/"runs.csv",index=False)
    summaries=[];ranks=[]
    plotdir=output/"convergence";plotdir.mkdir(exist_ok=True)
    for dim in cfg["dimensions"]:
        rankrows=[]
        with PdfPages(plotdir/f"{dim}d.pdf") as pdf:
            for name in BENCHMARK_NAMES:
                groups={}
                for algo in algorithms(dim,cfg):
                    group=[traces.get((dim,name,algo,s)) for s in cfg["seeds"]]
                    valid=[r for r in group if r is not None and r["status"]=="budget_reached"]
                    complete=len(valid)==len(cfg["seeds"])
                    scores=[r["best_score"] for r in valid]
                    summaries.append(dict(dimension=dim,function=name,algorithm=algo,n_valid=len(valid),n_expected=len(group),
                        complete=complete,mean_score=float(np.mean(scores)) if complete else None,
                        std_score=float(np.std(scores,ddof=1)) if complete and len(scores)>1 else None,
                        mean_seconds=float(np.mean([r["elapsed_seconds"] for r in valid])) if complete else None))
                    if complete:groups[algo]=valid
                if len(groups)!=len(algorithms(dim,cfg)):continue  # Fair complete blocks only.
                rankrows.extend(summaries[-len(algorithms(dim,cfg)):])
                fig,ax=plt.subplots(figsize=(10,6))
                negative=False;extents=[]
                for algo,rs in groups.items():
                    grid=np.unique(np.concatenate([np.asarray(r["trace"])[:,0] for r in rs]))
                    curves=[]
                    for r in rs:
                        a=np.asarray(r["trace"])
                        curves.append(a[np.searchsorted(a[:,0],grid,side="right")-1,1])
                    lo,med,hi=np.quantile(curves,[.25,.5,.75],axis=0)
                    ax.step(grid,med,where="post",label=algo)
                    ax.fill_between(grid,lo,hi,step="post",alpha=.1)
                    negative|=bool(np.min(curves)<0);extents.append((np.min(curves),np.max(curves)))
                ax.set_xscale("log")
                if not negative:
                    low=min(x[0] for x in extents);high=max(x[1] for x in extents)
                    ax.set_yscale("log" if low>0 else "symlog",**({} if low>0 else {"linthresh":1e-12}))
                    ax.set_ylim(low*.8 if low>0 else 0, max(high*1.2,1e-12))
                ax.set(xlabel="Actual objective evaluations",ylabel="Best ln(1+f)" if name=="f2_schwefel_2_22" else "Best objective value",
                       title=f"{dim}D {name}: {len(cfg['seeds'])}-seed median and IQR")
                ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.25);fig.tight_layout()
                fig.savefig(plotdir/f"{dim}d_{name}.png",dpi=150);pdf.savefig(fig);plt.close(fig)
        if rankrows:
            block=pd.DataFrame(rankrows);block["rank"]=block.groupby("function").mean_score.rank(method="average")
            for algo,g in block.groupby("algorithm"):
                ranks.append(dict(dimension=dim,algorithm=algo,average_rank=g['rank'].mean(),n_functions=g.function.nunique()))
    pd.DataFrame(summaries).to_csv(output/"summary.csv",index=False)
    pd.DataFrame(ranks).to_csv(output/"average_ranks.csv",index=False)
    expected=sum(1 for _ in jobs(cfg))
    status=dict(expected=expected,saved=len(rows),missing=expected-len(rows),failed=int((df.status!="budget_reached").sum()),
                note="Incomplete/failed function blocks excluded from ranks and convergence comparison plots; see summary.csv.")
    (output/"report_status.json").write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--config",default=str(ROOT/"experiments/configs/final_benchmark.json"))
    parser.add_argument("--output",required=True)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--shards",type=int,default=1)
    parser.add_argument("--retry-failed",action="store_true")
    parser.add_argument("--report",action="store_true")
    args=parser.parse_args()
    if args.report:report(args.output)
    else:run(args)


if __name__=="__main__":main()
