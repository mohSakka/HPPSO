"""Minimum final objective over successful seeds, with explicit seed coverage."""
import argparse
import gzip
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results/cec2011_pop200_ga_cap200k')
    args=parser.parse_args();out=args.output;plan=json.loads((out/'plan.json').read_text())
    algorithms=plan['config']['algorithms'];rows=[]
    for path in (out/'trials').rglob('*.json.gz'):
        with gzip.open(path,'rt') as f:r=json.load(f)
        assert r['fingerprint']==plan.get('problem_algorithm_fingerprints',{}).get(str(r['problem'])+':'+r['algorithm'],plan.get('problem_fingerprints',{}).get(str(r['problem']),plan.get('algorithm_fingerprints',{}).get(r['algorithm'],plan['fingerprint'])))
        rows.append(r)
    assert len(rows)==594
    df=pd.DataFrame(rows);assert not df.duplicated(['problem','algorithm','seed']).any()
    valid=df[df.status.isin(['iterations_completed','evaluation_budget_reached'])]
    assert np.isfinite(valid.best_score).all()
    grouped=valid.groupby(['problem','algorithm']).best_score.agg(['min','count'])
    values=grouped['min'].unstack().reindex(index=range(1,23),columns=algorithms)
    counts=grouped['count'].unstack().reindex(index=range(1,23),columns=algorithms).fillna(0).astype(int)
    shown=values.copy().astype(object)
    for p in values.index:
        for a in algorithms:
            n=counts.loc[p,a]
            shown.loc[p,a]='Failed' if n==0 else f'{values.loc[p,a]:.6g}'+(' †' if n<3 else '')
    for table in (values,counts,shown):table.index=['T'+str(n) for n in table.index];table.index.name='Problem'
    shown=shown.rename(columns={'Original PSO':'PSO'})
    folder=out/'tables';folder.mkdir(exist_ok=True)
    values.to_csv(folder/'minimum_by_problem.csv',float_format='%.17g')
    counts.to_csv(folder/'minimum_seed_counts.csv')
    note=('Minimum final objective across seeds 0–2; lower is better. Only successful, finite final outcomes are included. '
          'Population 200; GA-MPC has a strict 200,000-evaluation cap, while the other algorithms retain their 1000-iteration results. '
          'Failed means no successful seed. † marks fewer than three successful seeds. '
          'Partial best-so-far values from failed runs are excluded. '
          'These best-seed values differ from mean-based rankings. Values display six significant digits; CSV retains full precision.')
    if not plan.get('ga_evaluation_cap'):
        note=note.replace('GA-MPC has a strict 200,000-evaluation cap, while the other algorithms retain their 1000-iteration results.', 'All algorithms use 1000 iterations, with no evaluation cap; actual evaluation counts are recorded.')
    page='<!doctype html><meta charset="utf-8"><title>CEC2011 minimum objective table</title>'
    page+='<style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse;white-space:nowrap}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf3f8}tbody tr:nth-child(even){background:#f7f9fb}</style>'
    page+='<h1>Minimum final objective per algorithm and problem</h1><p>'+html.escape(note)+'</p>'
    page+='<p><a href="minimum_by_problem.csv">Full-precision CSV</a> · <a href="minimum_seed_counts.csv">Successful seed counts</a></p>'
    page+='<div style="overflow-x:auto">'+shown.to_html()+'</div>'
    (folder/'minimum_table.html').write_text(page,encoding='utf-8')
    (folder/'minimum_table.md').write_text(note+'\n\n'+shown.to_markdown(),encoding='utf-8')
    for file,link in [(out/'index.html','tables/minimum_table.html'),(folder/'index.html','minimum_table.html')]:
        s=file.read_text(encoding='utf-8')
        if link not in s:s=s.replace('<h1>',f'<p><a href="{link}">Minimum objective table</a></p><h1>',1)
        file.write_text(s,encoding='utf-8')
    print(shown.to_markdown())


if __name__=='__main__':main()
