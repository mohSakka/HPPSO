"""Resume the complete corrected 13-function study, report, audit and package."""
import argparse
import json
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED
from experiments.trait_plan import ROOT
from experiments.trait_study import run


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    output = ROOT/'experiments/results/trait_study_zero_velocity13'
    output.mkdir(parents=True, exist_ok=True)
    def stage(name):
        (output/'pipeline_status.json').write_text(json.dumps(dict(
            stage=name, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2))
        print('Pipeline stage: '+name,flush=True)
    stage('optimizing')
    run(argparse.Namespace(config=ROOT/'experiments/configs/trait_study_zero_velocity13.json',
        output=output,workers=args.workers,groups=None,shards=1,shard=0,
        retry_failed=False,dry_run=False))
    from experiments.trait_report import report
    from experiments.audit_trait_study import audit
    stage('reporting'); report(output,sweep_curves=True)
    stage('auditing'); audit(output)
    stage('packaging')
    archive=ROOT/'artifacts/trait_study_zero_velocity13_results.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.name!='pipeline_status.json':
                z.write(p,'trait_study_zero_velocity13/'+p.relative_to(output).as_posix())
    with ZipFile(archive) as z:
        assert z.testzip() is None
    stage('complete')


if __name__=='__main__': main()
