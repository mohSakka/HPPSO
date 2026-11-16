"""Run the ten-function extension, then combine, plot, audit and package all 13."""
import argparse
import json
from pathlib import Path
import time
from zipfile import ZipFile,ZIP_DEFLATED
from experiments.trait_study import run
from experiments.trait_plan import ROOT


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--output",type=Path,default=ROOT/"experiments/results/trait_study_added10")
    parser.add_argument("--combined-output",type=Path,default=ROOT/"experiments/results/trait_study_expanded13")
    parser.add_argument("--original",type=Path,default=ROOT/"experiments/results/trait_study_local3_final")
    parser.add_argument("--source-archive",type=Path,default=ROOT/"artifacts/trait_study_local3_source_snapshot.zip")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    def stage(name):
        data=dict(stage=name,updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
        (args.output/"pipeline_status.json").write_text(json.dumps(data,indent=2))
        print("Pipeline stage: "+name,flush=True)
    stage("optimizing")
    run(argparse.Namespace(config=ROOT/"experiments/configs/trait_study_added10.json",output=args.output,
        workers=args.workers,groups=None,shards=1,shard=0,retry_failed=False,dry_run=False))
    from experiments.trait_report import report
    from experiments.audit_trait_study import audit
    from experiments.combine_trait_studies import combine
    stage("reporting_additions");report(args.output,sweep_curves=True)
    stage("auditing_additions");audit(args.output)
    stage("combining")
    if not args.combined_output.exists():
        combine([args.original,args.output],args.combined_output,[args.source_archive])
    else:
        saved=json.loads((args.combined_output/"plan.json").read_text())
        expected=[str(args.original.resolve()),str(args.output.resolve())]
        if [s["directory"] for s in saved.get("analysis_sources",[])]!=expected:
            raise ValueError("Existing combined output has incompatible sources")
    stage("reporting_expanded");report(args.combined_output,sweep_curves=True)
    stage("auditing_expanded");audit(args.combined_output)
    stage("packaging")
    archive=ROOT/"artifacts/trait_study_expanded13_results.zip"
    with ZipFile(archive,"w",ZIP_DEFLATED) as z:
        for path in sorted(args.combined_output.rglob("*")):
            if path.is_file() and not path.name.endswith((".lock",".tmp")):
                z.write(path,Path("trait_study_expanded13")/path.relative_to(args.combined_output))
    with ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError("Result archive validation failed")
    stage("complete")
    print("Completed expanded study: "+str(args.combined_output),flush=True)
    print("Results: "+str(archive),flush=True)


if __name__=="__main__":main()
