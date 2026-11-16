"""Resume the unchanged CEC2011 run with lock-resistant versioned CSV exports."""
# Import the frozen experiment first so its numerical thread limits take effect.
from experiments import cec2011_budgeted as benchmark
import json
from pathlib import Path
import time
import pandas as pd


def main():
    original=pd.DataFrame.to_csv
    def export(frame,path_or_buf=None,*args,**kwargs):
        if isinstance(path_or_buf,(str,Path)) and Path(path_or_buf).name in ('runs.csv','summary.csv'):
            path=Path(path_or_buf);folder=path.parent/'exports';folder.mkdir(exist_ok=True)
            target=folder/f'{path.stem}_{time.time_ns()}.csv'
            result=original(frame,target,*args,**kwargs)
            manifest=path.parent/'csv_latest.json'
            previous=json.loads(manifest.read_text()) if manifest.exists() else {}
            previous[path.name]=target.relative_to(path.parent).as_posix()
            benchmark.atomic(manifest,previous)
            benchmark.atomic(path.parent/'recovery_metadata.json',dict(
                reason='Windows denied overwriting summary.csv; export CSVs under unique names.',
                optimizer_source_unchanged=True,raw_trial_fingerprints_unchanged=True,
                export_manifest='csv_latest.json'))
            return result
        return original(frame,path_or_buf,*args,**kwargs)
    pd.DataFrame.to_csv=export
    benchmark.PAGE=benchmark.PAGE.replace('href="runs.csv"','id="runs-link" href="runs.csv"').replace(
        'href="summary.csv"','id="summary-link" href="summary.csv"')
    benchmark.PAGE=benchmark.PAGE.replace("document.querySelector('#bar').value=p.completed;", """
document.querySelector('#bar').value=p.completed;
try{let exports=await(await fetch('csv_latest.json?t='+Date.now())).json();
for(let [file,id] of [['runs.csv','runs-link'],['summary.csv','summary-link']]){
if(exports[file])document.getElementById(id).href=exports[file]}}catch(e){}
""")
    benchmark.main()


if __name__=='__main__':main()
