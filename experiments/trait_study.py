"""Resumable vectorized trait experiments: one independent batch per seed/function."""
import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import traceback
import numpy as np
import scipy
from experiments.trait_plan import ROOT,load_config,build_plan
from experiments.trait_batch import BatchFunction,BatchBudget,BatchedTraitPSO
from experiments.final_benchmark import write_result,read_result,shard_lease
from experiments.trait_functions import trait_suite as suite
from hppso.benchmarks.classical import problem_definitions


def jobs(cfg):
    return [dict(function=f,seed=s) for f in cfg["functions"] for s in cfg["seeds"]]


def result_path(output,job):
    return Path(output)/"trials"/job["function"]/f"seed_{job['seed']:03d}.json.gz"


def fingerprint(cfg):
    paths=[Path(__file__),ROOT/"experiments/trait_plan.py",ROOT/"experiments/trait_batch.py",
           ROOT/"experiments/trait_functions.py",
           ROOT/"src/hppso/algorithms/trait_study.py",ROOT/"src/hppso/algorithms/_pso_bounds.py",
           ROOT/"experiments/final_benchmark.py"]
    paths+=sorted((ROOT/"src/hppso/benchmarks").glob("*.py"))
    if cfg.get("reuse_pso_controls"):
        paths.append(ROOT/"experiments/diversity_study.py")
    h=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode())
    for p in paths:
        h.update(p.relative_to(ROOT).as_posix().encode());h.update(p.read_bytes().replace(b"\r\n",b"\n"))
    return h.hexdigest()


def trial(job,cfg,protocol,output,digest,retry=False):
    problem=suite(cfg["dimension"],cfg["instance_seed"])[job["function"]]
    identifiers=list(protocol["variants"])
    reuse=cfg.get("reuse_pso_controls")
    if reuse:
        return trial_with_reused_controls(job,cfg,protocol,output,digest,problem,identifiers)
    saved_controls = {}
    if cfg.get("reuse_baseline"):
        baseline = protocol["baseline"]
        saved_controls[baseline] = load_saved_baseline(job, cfg, protocol, baseline)
    for identifier in list(identifiers):
        label = protocol["variants"][identifier]["label"]
        if label in cfg.get("reuse_comparators", {}):
            saved_controls[identifier] = load_saved_baseline(job, cfg, protocol, identifier,
                                                             cfg["reuse_comparators"][label])
    identifiers = [key for key in identifiers if key not in saved_controls]
    progress = None
    if cfg.get("live_seed_progress"):
        progress_file = result_path(output, job).with_suffix(".progress.json")
        progress_file.parent.mkdir(parents=True, exist_ok=True)

        def progress(count):
            temp = progress_file.with_suffix(f".{os.getpid()}.tmp")
            temp.write_text(json.dumps(dict(job, evaluations=count, pid=os.getpid())))
            # A dashboard read can briefly hold the destination open on Windows.
            try:
                os.replace(temp, progress_file)
            except PermissionError:
                temp.unlink(missing_ok=True)

    tracker=BatchBudget(BatchFunction(problem),cfg["budget"],len(identifiers),cfg["dimension"],cfg["trace_points"],progress)
    engine=None;status="failed";error=None
    started=time.perf_counter()
    try:
        engine=BatchedTraitPSO(tracker,problem["bounds"],cfg["population"],job["seed"],
            [protocol["variants"][key]["spec"] for key in identifiers],cfg)
        engine.optimize();status="budget_reached"
    except Exception:
        error=traceback.format_exc()
    elapsed=time.perf_counter()-started
    result=dict(job,status=status,error=error,fingerprint=digest,variant_ids=identifiers,
        evaluations_per_path=tracker.count,logical_evaluations=tracker.count*len(identifiers),
        objective_batch_calls=int(np.ceil(tracker.count/cfg["population"])),
        best_scores=[float(v) if np.isfinite(v) else None for v in tracker.best],
        best_positions=tracker.best_positions.tolist(),
        initial_best=None if engine is None else engine.initial_best.tolist(),
        reset_events=None if engine is None else engine.reset_events.tolist(),
        trace_evaluations=[n for n,_ in tracker.trace],
        trace_best=[values.tolist() for _,values in tracker.trace],
        batch_elapsed_seconds=elapsed,host=platform.node(),finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    if saved_controls and status == "budget_reached":
        for position, identifier in enumerate(protocol["variants"]):
            if identifier not in saved_controls:
                continue
            saved = saved_controls[identifier]
            column = saved["variant_ids"].index(identifier)
            if result["trace_evaluations"] != saved["trace_evaluations"]:
                raise ValueError("Saved control evaluation checkpoints differ")
            if not np.allclose(result["initial_best"], saved["initial_best"][column], rtol=1e-12, atol=1e-12):
                raise ValueError("Saved control initial objective differs")
            for field in ("best_scores", "best_positions", "initial_best", "reset_events"):
                result[field].insert(position, saved[field][column])
            for trace_row, saved_row in zip(result["trace_best"], saved["trace_best"], strict=True):
                trace_row.insert(position, saved_row[column])
        result.update(variant_ids=list(protocol["variants"]),
                      executed_paths=len(identifiers), executed_evaluations=result["logical_evaluations"],
                      logical_evaluations=tracker.count * len(protocol["variants"]),
                      reused_control_fingerprints={key: saved["fingerprint"] for key, saved in saved_controls.items()})
        if cfg.get("reuse_baseline"):
            result["reused_baseline_fingerprint"] = saved_controls[protocol["baseline"]]["fingerprint"]
    elif saved_controls:
        for position, identifier in enumerate(protocol["variants"]):
            if identifier in saved_controls:
                for field in ("best_scores", "best_positions", "initial_best", "reset_events"):
                    if result[field] is not None:
                        result[field].insert(position, None)
        result["variant_ids"] = list(protocol["variants"])
    path=result_path(output,job)
    if retry and path.exists():
        archive=path.parent/"attempts"/(path.name+f".{time.time_ns()}")
        archive.parent.mkdir(exist_ok=True);shutil.copyfile(path,archive)
    write_result(path,result)
    return dict(job,status=status,evaluations_per_path=tracker.count,paths=len(identifiers),elapsed=elapsed,error=error)


def load_saved_baseline(job, cfg, protocol, baseline, reference=None):
    reference = cfg["reuse_baseline"] if reference is None else reference
    source = ROOT / reference["directory"]
    plan = json.loads((source / "plan.json").read_text())
    if plan["fingerprint"] != reference["fingerprint"]:
        raise ValueError("Saved PSO source fingerprint differs")
    settings = ("dimension", "population", "budget", "instance_seed", "w", "c1", "c2", "trace_points",
                "velocity_initialization", "coordinatewise_hppso_draws", "pso_m_rate", "pso_m_strength")
    if any(cfg.get(key) != plan["config"].get(key) for key in settings):
        raise ValueError("Saved PSO uses different run settings")
    if plan["protocol"]["variants"][baseline]["spec"] != protocol["variants"][baseline]["spec"]:
        raise ValueError("Saved PSO uses a different variant specification")
    saved = read_result(result_path(source, job))
    if (saved["fingerprint"] != plan["fingerprint"] or saved["status"] != "budget_reached"
            or saved["evaluations_per_path"] != cfg["budget"] or baseline not in saved["variant_ids"]
            or any(saved[key] != value for key, value in job.items())):
        raise ValueError("Invalid saved PSO trial")
    return saved


def trial_with_reused_controls(job,cfg,protocol,output,digest,problem,identifiers):
    from experiments.diversity_study import METRICS, measure

    source=ROOT/cfg["reuse_pso_controls"]
    old_plan=json.loads((source/"plan.json").read_text())
    expected={k:v for k,v in cfg.items() if k not in ("reuse_pso_controls","coordinatewise_hppso_draws")}
    if old_plan["config"]!=expected or old_plan["protocol"]!=protocol:
        raise ValueError("Saved controls use different settings or variants")
    old=read_result(result_path(source,job))
    if (old["fingerprint"]!=old_plan["fingerprint"] or old["variant_ids"]!=identifiers
            or old["status"]!="budget_reached" or old["evaluations_per_path"]!=cfg["budget"]):
        raise ValueError("Invalid saved control batch")
    if cfg["budget"] % cfg["population"]:
        raise ValueError("Integrated diversity requires complete generations")

    baseline=protocol["baseline"]
    full=protocol["factorial"]["15"]
    specs={key:protocol["variants"][key]["spec"] for key in identifiers}
    active=[key for key in identifiers if key==baseline or specs[key]["backbone"]=="hp"]
    tracker=BatchBudget(BatchFunction(problem),cfg["budget"],len(active),cfg["dimension"],cfg["trace_points"])
    started=time.perf_counter()
    engine=BatchedTraitPSO(tracker,problem["bounds"],cfg["population"],job["seed"],[specs[key] for key in active],cfg)
    pair=[active.index(full),active.index(baseline)]
    generations=cfg["budget"]//cfg["population"]
    targets=set((np.unique(np.r_[1,np.geomspace(1,generations,100).astype(int),
                               np.linspace(1,generations,201).astype(int)])*cfg["population"]).tolist())
    counts=[tracker.count]
    diversity=[measure(engine.positions[pair],engine.width).tolist()]
    errors=[tracker.best[pair].tolist()]
    while tracker.count<tracker.budget:
        sample=min(tracker.count+cfg["population"],tracker.budget) in targets
        if sample:
            previous=engine.positions[pair]
        engine.step()
        if sample:
            counts.append(tracker.count)
            diversity.append(measure(engine.positions[pair],engine.width,previous).tolist())
            errors.append(tracker.best[pair].tolist())
    assert counts[-1]==cfg["budget"]
    assert np.isfinite(diversity).all() and np.min(diversity)>=0
    assert [n for n,_ in tracker.trace]==old["trace_evaluations"]
    columns=[identifiers.index(key) for key in active]
    base_col=identifiers.index(baseline)
    np.testing.assert_allclose(np.asarray([v[pair[1]] for _,v in tracker.trace]),
                               np.asarray(old["trace_best"])[:,base_col],rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(tracker.best[pair[1]],old["best_scores"][base_col],rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(tracker.best_positions[pair[1]],old["best_positions"][base_col],rtol=1e-12,atol=1e-10)
    for field,values in (
        ("best_scores",tracker.best),("best_positions",tracker.best_positions),
        ("initial_best",engine.initial_best),("reset_events",engine.reset_events),
    ):
        merged=np.asarray(old[field])
        merged[columns]=values
        old[field]=merged.tolist()
    trace=np.asarray(old["trace_best"])
    trace[:,columns]=np.asarray([values for _,values in tracker.trace])
    elapsed=time.perf_counter()-started
    old.update(fingerprint=digest,trace_best=trace.tolist(),batch_elapsed_seconds=elapsed,
               executed_paths=len(active),executed_evaluations=len(active)*tracker.count,
               reused_control_fingerprint=old_plan["fingerprint"],host=platform.node(),
               finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
               diversity_record=dict(labels=["HPPSO","PSO"],metrics=METRICS,evaluations=counts,
                                     diversity=diversity,best_errors=errors))
    write_result(result_path(output,job),old)
    return dict(job,status="budget_reached",evaluations_per_path=tracker.count,
                paths=len(active),elapsed=elapsed,error=None)


def run(args):
    cfg=load_config(args.config)
    if args.groups:cfg["groups"]=args.groups
    protocol=build_plan(cfg);digest=fingerprint(cfg)
    if args.workers<1 or args.shards<1 or not 0<=args.shard<args.shards:raise ValueError("Invalid workers/shards")
    if args.dry_run:
        print(json.dumps(dict(batch_jobs=len(jobs(cfg)),unique_paths=len(protocol["variants"]),
            comparisons=len(jobs(cfg))*len(protocol["variants"]),
            logical_evaluations=len(jobs(cfg))*len(protocol["variants"])*cfg["budget"],
            groups={k:len(v) for k,v in protocol["groups"].items()}),indent=2));return
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    plan=dict(config=cfg,protocol=protocol,fingerprint=digest,shards=args.shards,
              evaluator="vectorized-nine-v1",format_version=1)
    tmp=output/f".plan-{os.getpid()}-{time.time_ns()}.tmp"
    tmp.write_text(json.dumps(plan,indent=2),encoding="utf-8")
    try:
        try:os.link(tmp,output/"plan.json")
        except FileExistsError:pass
    finally:tmp.unlink(missing_ok=True)
    if json.loads((output/"plan.json").read_text())!=plan:
        raise ValueError("Source/config/shard count changed; choose a fresh output directory")
    with shard_lease(output,f"traits-{args.shard}"):
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                         host=platform.node(),platform=platform.platform())
        (output/f"environment-{args.shard}.json").write_text(json.dumps(environment,indent=2))
        definitions=problem_definitions([suite(cfg["dimension"],cfg["instance_seed"])[f] for f in cfg["functions"]])
        tmp=output/f".definitions-{os.getpid()}-{time.time_ns()}.tmp"
        tmp.write_text(json.dumps(definitions,indent=2));os.replace(tmp,output/"problem_definitions.json")
        selected=[j for i,j in enumerate(jobs(cfg)) if i%args.shards==args.shard]
        pending=[]
        for job in selected:
            path=result_path(output,job)
            if path.exists():
                saved=read_result(path)
                if saved["fingerprint"]!=digest:raise ValueError("Mixed trial fingerprint")
                if not args.retry_failed or saved["status"]=="budget_reached":continue
            pending.append(job)
        done=len(selected)-len(pending);failed=0
        print(f"Shard {args.shard}: {len(protocol['variants'])} paths per batch; {done}/{len(selected)} saved; {len(pending)} pending",flush=True)
        def progress():
            data=dict(completed=done,total=len(selected),failed_this_invocation=failed,
                      paths_per_batch=len(protocol["variants"]),budget_per_path=cfg["budget"],updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
            file=output/f"progress-{args.shard}.json";tmp=file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data));os.replace(tmp,file)
        progress();iterator=iter(pending)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            active={}
            def fill():
                while len(active)<2*args.workers:
                    job=next(iterator,None)
                    if job is None:break
                    active[pool.submit(trial,job,cfg,protocol,str(output),digest,args.retry_failed)]=job
            fill()
            while active:
                ready,_=wait(active,return_when=FIRST_COMPLETED)
                for future in ready:
                    active.pop(future);r=future.result();done+=1;failed+=r["status"]!="budget_reached"
                    print(f"[{done}/{len(selected)}] {r['function']} seed={r['seed']} {r['status']} "
                          f"{r['paths']} paths x {r['evaluations_per_path']} FE; batch={r['elapsed']:.2f}s",flush=True)
                    if r["error"]:print(r["error"],flush=True)
                progress();fill()
    print("Batches finished. Generate reports with: python -m experiments.trait_report --output "+str(output),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"experiments/configs/trait_study.json")
    parser.add_argument("--output",type=Path,default=ROOT/"experiments/results/trait_study")
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--groups",nargs="+")
    parser.add_argument("--shards",type=int,default=1);parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--retry-failed",action="store_true");parser.add_argument("--dry-run",action="store_true")
    run(parser.parse_args())


if __name__=="__main__":main()
