"""Audit completed path solutions and independently reconstruct pair interactions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.trait_study import jobs,result_path,fingerprint
from experiments.final_benchmark import read_result
from experiments.trait_functions import trait_suite as suite
from experiments.trait_provenance import snapshot_fingerprint


def audit(output,source_archive=None):
    output=Path(output);plan=json.loads((output/"plan.json").read_text())
    cfg,p=plan["config"],plan["protocol"];ids=list(p["variants"]);lookup={v:i for i,v in enumerate(ids)}
    sources=plan.get("analysis_sources",[dict(config=cfg,fingerprint=plan["fingerprint"])])
    expected_fingerprints={}
    for source in sources:
        if fingerprint(source["config"])!=source["fingerprint"]:
            archive=Path(source_archive) if source_archive is not None and len(sources)==1 else output/source.get("source_archive","")
            if not archive.is_file() or snapshot_fingerprint(source["config"],archive)!=source["fingerprint"]:
                raise ValueError("Scientific source/configuration changed and no verified source archive is available")
        for name in source["config"]["functions"]:expected_fingerprints[name]=source["fingerprint"]
    expected_jobs=jobs(cfg);checked=0;logical=0;pair_data={};zero_reset=lookup[p["no_reinitialization"]]
    legacy_matches=0;all_reset_counts=[];executed=0
    for function in cfg["functions"]:
        responses=[]
        for seed in cfg["seeds"]:
            job=dict(function=function,seed=seed);r=read_result(result_path(output,job))
            assert r["variant_ids"]==ids and r["fingerprint"]==expected_fingerprints[function]
            assert r["status"]=="budget_reached" and r["error"] is None
            assert r["evaluations_per_path"]==cfg["budget"]
            assert r["logical_evaluations"]==len(ids)*cfg["budget"]
            assert r["objective_batch_calls"]==int(np.ceil(cfg["budget"]/cfg["population"]))
            trace=np.asarray(r["trace_best"]);grid=np.asarray(r["trace_evaluations"])
            assert np.isfinite(trace).all() and np.all(np.diff(grid)>0)
            assert np.all(np.diff(trace,axis=0)<=0) and grid[-1]==cfg["budget"]
            np.testing.assert_array_equal(trace[-1],r["best_scores"])
            np.testing.assert_allclose(r["initial_best"],np.full(len(ids),r["initial_best"][0]),rtol=1e-12,atol=1e-10)
            prefix=trace[grid<=cfg["population"]]
            np.testing.assert_allclose(prefix,np.broadcast_to(prefix[:,:1],prefix.shape),rtol=1e-12,atol=1e-10)
            problem=suite(cfg["dimension"],cfg["instance_seed"])[function]
            positions=np.asarray(r["best_positions"]);lo,hi=problem["bounds"]
            assert positions.shape==(len(ids),cfg["dimension"])
            assert np.all(positions>=lo-1e-10) and np.all(positions<=hi+1e-10)
            # Evaluate the ORIGINAL scalar benchmark, not the vectorized adapter.
            scalar=np.array([problem["function"](x) for x in positions])
            np.testing.assert_allclose(scalar,r["best_scores"],rtol=5e-12,atol=1e-9)
            for i,v in enumerate(ids):
                if not p["variants"][v]["spec"].get("reinitialize",False):assert r["reset_events"][i]==0
            assert r["reset_events"][zero_reset]==0
            all_reset_counts.append(r["reset_events"])
            order=[lookup[p["factorial"][str(mask)]] for mask in range(16)]
            final=np.asarray(r["best_scores"])[order];initial=np.asarray(r["initial_best"])[order]
            assert np.all(final>=-1e-10) and np.all(final<=initial+1e-8)
            response=np.divide(np.maximum(final,0),initial,out=np.zeros_like(final),where=initial>0)
            responses.append(response)
            # An independent explicit difference-in-differences construction.
            for a in range(4):
                for b in range(a+1,4):
                    subset=(1<<a)|(1<<b)
                    backgrounds=[m for m in range(16) if not m&subset]
                    effect=np.mean([response[m|subset]-response[m|(1<<a)]-response[m|(1<<b)]+response[m] for m in backgrounds])
                    pair_data[function,seed,subset]=float(effect)
            if cfg.get("velocity_initialization") != "zeros" and function=="f1_sphere" and cfg["budget"]==100000 and cfg["dimension"]==100 and cfg["population"]==50:
                legacy=Path(__file__).resolve().parent/"results/openness_ablation_100d/trials/f1_sphere/PSO"/f"seed_{seed:03d}.json.gz"
                if legacy.exists():
                    old=read_result(legacy);i=lookup[p["baseline"]]
                    np.testing.assert_allclose(r["best_scores"][i],old["best_score"],rtol=1e-12,atol=1e-9)
                    legacy_matches+=1
            checked+=1;logical+=r["logical_evaluations"]
            executed+=r.get("executed_evaluations",r["logical_evaluations"])
    assert checked==len(expected_jobs)
    runs=pd.read_csv(output/"runs.csv");summary=pd.read_csv(output/"summary.csv")
    assert len(runs)==checked*len(ids) and runs.status.eq("budget_reached").all()
    assert len(summary)==len(cfg["functions"])*len(ids) and summary.n.eq(len(cfg["seeds"])).all()
    seed_effects=pd.read_csv(output/"factorial_seed_contrasts.csv")
    effects=pd.read_csv(output/"factorial_effects.csv")
    for (function,seed,subset),value in pair_data.items():
        row=seed_effects[seed_effects.function.eq(function)&seed_effects.seed.eq(seed)&seed_effects.subset.eq(subset)]
        assert len(row)==1
        np.testing.assert_allclose(row.contrast.iloc[0],value,rtol=1e-11,atol=1e-12)
    for subset in (3,5,6,9,10,12):
        per_seed=[np.mean([pair_data[f,s,subset] for f in cfg["functions"]]) for s in cfg["seeds"]]
        row=effects[effects.function.eq("aggregate_fixed_functions")&effects.subset.eq(subset)]
        np.testing.assert_allclose(row.mean_contrast.iloc[0],np.mean(per_seed),rtol=1e-11,atol=1e-12)
    for i,v in enumerate(ids):
        recorded=summary[summary.variant.eq(v)]
        assert np.isfinite(recorded[["mean","std","median","median_ci95_low","median_ci95_high"]].to_numpy()).all()
    result=dict(completed_batches=checked,independent_paths=len(ids),completed_path_runs=checked*len(ids),
        optimization_evaluations=logical,exact_budgets=True,all_solutions_feasible=True,
        original_scalar_benchmarks_recompute_scores=True,matched_initial_populations=True,
        monotone_finite_traces=True,no_reset_controls_verified=True,
        pairwise_factorial_contrasts_independently_reconstructed=True,
        aggregate_seed_blocks_verified=True,legacy_sphere_pso_seeds_matched=legacy_matches)
    if cfg.get("reuse_pso_controls"):
        result.update(new_optimization_evaluations=executed,reused_evaluations=logical-executed)
    (output/"audit.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-archive",type=Path)
    args=parser.parse_args();audit(args.output,args.source_archive)
