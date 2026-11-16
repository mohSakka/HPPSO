"""Combine compatible, disjoint trait batches while preserving source fingerprints."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from experiments.trait_study import fingerprint,jobs,result_path
from experiments.final_benchmark import read_result
from experiments.trait_provenance import snapshot_fingerprint


def combine(sources,output,archives=()):
    sources=[Path(p).resolve() for p in sources];output=Path(output).resolve()
    if output.exists():raise ValueError("Choose a fresh combined output directory")
    archived={}
    for archive in archives:
        archive=Path(archive).resolve()
        for source in sources:
            plan=json.loads((source/"plan.json").read_text())
            if snapshot_fingerprint(plan["config"],archive)==plan["fingerprint"]:
                archived[plan["fingerprint"]]=archive
    plans=[json.loads((source/"plan.json").read_text()) for source in sources]
    first=plans[0];cfg=dict(first["config"]);functions=[];manifest=[]
    shared=lambda c:{k:v for k,v in c.items() if k!="functions"}
    for source,plan in zip(sources,plans):
        if shared(plan["config"])!=shared(cfg) or plan["protocol"]!=first["protocol"]:
            raise ValueError("Scientific settings and paths must match across sources")
        if set(functions)&set(plan["config"]["functions"]):raise ValueError("Duplicate functions across sources")
        if fingerprint(plan["config"])!=plan["fingerprint"] and plan["fingerprint"] not in archived:
            raise ValueError(f"Source version not verified: {source}; supply its source archive")
        for job in jobs(plan["config"]):
            r=read_result(result_path(source,job))
            if r["fingerprint"]!=plan["fingerprint"] or r["status"]!="budget_reached" or r["evaluations_per_path"]!=cfg["budget"]:
                raise ValueError(f"Incomplete source trial: {source}, {job}")
        functions.extend(plan["config"]["functions"])
        manifest.append(dict(directory=str(source),config=plan["config"],fingerprint=plan["fingerprint"]))
    cfg["functions"]=functions
    output.mkdir(parents=True);provenance=output/"provenance";provenance.mkdir()
    definitions={}
    for source,plan in zip(sources,plans):
        for job in jobs(plan["config"]):
            dest=result_path(output,job);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(result_path(source,job),dest)
        definitions.update(json.loads((source/"problem_definitions.json").read_text()))
        entry=next(s for s in manifest if s["fingerprint"]==plan["fingerprint"])
        subfolder=provenance/plan["fingerprint"][:16];subfolder.mkdir(exist_ok=True)
        shutil.copyfile(source/"plan.json",subfolder/"execution_plan.json")
        entry["execution_plan"]=(subfolder/"execution_plan.json").relative_to(output).as_posix()
        entry["environment_files"]=[]
        for environment in source.glob("environment-*.json"):
            shutil.copyfile(environment,subfolder/environment.name)
            entry["environment_files"].append((subfolder/environment.name).relative_to(output).as_posix())
        if plan["fingerprint"] in archived:
            archive=archived[plan["fingerprint"]];filename=plan["fingerprint"][:16]+"_source.zip"
            shutil.copyfile(archive,provenance/filename)
            next(s for s in manifest if s["fingerprint"]==plan["fingerprint"])["source_archive"]="provenance/"+filename
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    combined=dict(first,config=cfg,fingerprint=digest,analysis_sources=manifest,
                  note="Combined analysis manifest. Each raw batch retains its original execution fingerprint.")
    (output/"plan.json").write_text(json.dumps(combined,indent=2))
    (output/"problem_definitions.json").write_text(json.dumps(definitions,indent=2))
    print(f"Combined {len(functions)} functions, {len(jobs(cfg))} batches without changing raw results.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--sources",nargs="+",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-archives",nargs="*",type=Path,default=[])
    args=parser.parse_args();combine(args.sources,args.output,args.source_archives)
