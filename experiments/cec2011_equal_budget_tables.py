"""Audited tables for the population-90, equal-150000-evaluation experiment."""
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd


def generate(output):
    output=Path(output);plan=json.loads((output/'plan.json').read_text());rows=[]
    algorithms=plan['config']['algorithms']
    for f in (output/'trials').rglob('*.json.gz'):
        with gzip.open(f,'rt') as stream:r=json.load(stream)
        assert r['fingerprint']==plan['fingerprint'] and r['population']==90
        if r['status']=='budget_reached':assert r['evaluations']==150000 and np.isfinite(r['best_score'])
        rows.append({k:v for k,v in r.items() if k not in ('trace','best_position')})
    df=pd.DataFrame(rows);assert len(df)==594 and not df.duplicated(['problem','algorithm','seed']).any()
    valid=df[df.status=='budget_reached']
    complete=[i for i,b in df.groupby('problem') if len(b)==27 and b.status.eq('budget_reached').all()]
    stats=valid.groupby(['problem','algorithm']).best_score.agg(['count','mean','std','median','min','max'])
    stats['rank']=stats.groupby(level='problem')['mean'].rank(method='average')
    stats.loc[~stats.index.get_level_values('problem').isin(complete),'rank']=np.nan
    ranks=stats['rank'].unstack().reindex(columns=algorithms)
    assert np.allclose(ranks.loc[complete].sum(axis=1),45)
    overall=ranks.mean().sort_values().to_frame('average_rank');overall.insert(0,'position',overall.average_rank.rank(method='min'))
    counts=stats['count'].unstack().reindex(index=range(1,23),columns=algorithms).fillna(0).astype(int)
    minimum=stats['min'].unstack().reindex(index=range(1,23),columns=algorithms)
    performance=pd.DataFrame(index=range(1,23),columns=algorithms)
    for i in performance.index:
        for a in algorithms:
            if counts.loc[i,a]!=3:performance.loc[i,a]=f'Incomplete ({counts.loc[i,a]}/3)';continue
            r=stats.loc[(i,a)];performance.loc[i,a]=f'{r["mean"]:.7g} ± {r["std"]:.3g} ({r["rank"]:g})'
    folder=output/'tables';folder.mkdir(exist_ok=True)
    stats.to_csv(folder/'performance_and_ranks.csv',float_format='%.17g');ranks.to_csv(folder/'ranks_by_problem.csv')
    overall.to_csv(folder/'overall_ranks.csv');minimum.to_csv(folder/'minimum_by_problem.csv',float_format='%.17g')
    counts.to_csv(folder/'minimum_seed_counts.csv');df.to_csv(folder/'audited_trial_results.csv',index=False)
    df[df.status!='budget_reached'].to_csv(folder/'failed_trials.csv',index=False)
    note=(f'Population 90; exactly 150,000 objective evaluations for every algorithm; seeds 0–2. '
        f'Mean-based ranks, lower is better; exact ties receive average ranks. {len(complete)} complete problem blocks enter overall ranks. '
        'SD uses the sample denominator. Minima use successful final outcomes only. Three-seed results are preliminary. '
        'T3 bounds are [0.6,0.9]; CMA-ES uses covariance safeguards; GA-MPC reuses known fitness. '
        'No generation limit stops GA-MPC. All objective calls, including non-finite candidates, count toward the budget.')
    style='<style>body{font:15px system-ui;margin:25px}table{border-collapse:collapse;white-space:nowrap}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf3f8}.wide{overflow:auto}</style>'
    page='<!doctype html><meta charset="utf-8">'+style+'<h1>CEC2011: equal evaluation budgets</h1><p>'+note+'</p>'
    for title,table in [('Overall ranks',overall),('Per-problem ranks',ranks),('Mean ± SD (rank)',performance)]:
        page+='<h2>'+title+'</h2><div class="wide">'+table.to_html(float_format=lambda x:f'{x:.6g}')+'</div>'
    page+='<p><a href="minimum_table.html">Minimum values</a> · <a href="performance_and_ranks.csv">Numeric results CSV</a></p>'
    (folder/'index.html').write_text(page,encoding='utf-8')
    (folder/'minimum_table.html').write_text('<!doctype html><meta charset="utf-8">'+style+'<h1>Minimum across successful seeds</h1><p>'+note+'</p><div class="wide">'+minimum.to_html(float_format=lambda x:f'{x:.7g}')+'</div><h2>Successful seed counts</h2>'+counts.to_html(),encoding='utf-8')
    (folder/'audit.json').write_text(json.dumps(dict(trials=len(df),successful=len(valid),failed=len(df)-len(valid),
        complete_problems=complete,exact_evaluations=150000,population=90,unique_trial_keys=True),indent=2))
    print(overall.to_string(),flush=True)
    if len(valid)==594:
        import sys
        from experiments.cec2011_wilcoxon import main as wilcoxon_tables
        previous=sys.argv
        try:
            sys.argv=[sys.argv[0],'--output',str(output)];wilcoxon_tables()
        finally:sys.argv=previous
