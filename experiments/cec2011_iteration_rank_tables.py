"""Build CEC2011 per-problem ranking tables directly from audited raw trials."""
import argparse
import gzip
import json
from pathlib import Path
import html
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results/cec2011_pop200_iter1000')
    args=parser.parse_args();output=args.output
    plan=json.loads((output/'plan.json').read_text());cfg=plan['config'];algorithms=cfg['algorithms']
    rows=[]
    for path in (output/'trials').rglob('*.json.gz'):
        with gzip.open(path,'rt') as stream:r=json.load(stream)
        assert r['fingerprint']==plan.get('problem_algorithm_fingerprints',{}).get(str(r['problem'])+':'+r['algorithm'],plan.get('problem_fingerprints',{}).get(str(r['problem']),plan.get('algorithm_fingerprints',{}).get(r['algorithm'],plan['fingerprint']))),path
        assert r['population']==200,path
        assert r['iterations']==1000 or (r['algorithm']=='GA-MPC' and plan.get('ga_fitness_reuse') and r['iterations'] is None),path
        if r['status'] in ('iterations_completed','evaluation_budget_reached'):assert np.isfinite(r['best_score']),path
        if plan.get('ga_evaluation_cap') and r['algorithm']=='GA-MPC':assert r['evaluations']<=plan['ga_evaluation_cap'],path
        rows.append({k:v for k,v in r.items() if k not in ('trace','best_position')})
    df=pd.DataFrame(rows)
    assert len(df)==594 and not df.duplicated(['problem','algorithm','seed']).any()
    assert df.groupby(['problem','algorithm']).size().eq(3).all()
    assert set(df.problem)==set(range(1,23)) and set(df.algorithm)==set(algorithms)
    success=('iterations_completed','evaluation_budget_reached')
    valid=df[df.status.isin(success)]
    complete=[p for p,b in df.groupby('problem') if len(b)==27 and b.status.isin(success).all()]
    stats=valid.groupby(['problem','algorithm']).best_score.agg(['count','mean','std','median','min','max'])
    stats['rank']=stats.groupby(level='problem')['mean'].rank(method='average',ascending=True)
    stats.loc[~stats.index.get_level_values('problem').isin(complete),'rank']=np.nan
    means=stats['mean'].unstack().reindex(columns=algorithms)
    ranks=stats['rank'].unstack().reindex(columns=algorithms)
    assert np.allclose(ranks.loc[complete].sum(axis=1),45)
    winners=means.loc[complete].eq(means.loc[complete].min(axis=1),axis=0)
    overall=pd.DataFrame(dict(average_rank=ranks.mean(),best_mean_count_including_ties=winners.sum(),
        outright_best_mean_count=(winners & winners.sum(axis=1).eq(1).to_numpy()[:,None]).sum()))
    overall=overall.sort_values('average_rank');overall.insert(0,'overall_rank',overall.average_rank.rank(method='average'))
    folder=output/'tables';folder.mkdir(exist_ok=True)
    stats.to_csv(folder/'performance_and_ranks.csv',float_format='%.17g')
    ranks.to_csv(folder/'ranks_by_problem.csv')
    overall.to_csv(folder/'overall_ranks.csv')
    df.to_csv(folder/'audited_trial_results.csv',index=False)
    df[~df.status.isin(success)].to_csv(folder/'failed_trials.csv',index=False)
    formatted=pd.DataFrame(index=means.index,columns=algorithms)
    for number in means.index:
        for algo in algorithms:
            if (number,algo) not in stats.index:
                formatted.loc[number,algo]='FAILED (0/3 complete)';continue
            r=stats.loc[(number,algo)]
            if r['count']<3:
                formatted.loc[number,algo]=f'INCOMPLETE ({int(r["count"])}/3 complete)';continue
            formatted.loc[number,algo]=f'{r["mean"]:.7g} ± {r["std"]:.3g} ({r["rank"]:g})'
    formatted.index=['T'+str(i) for i in formatted.index];formatted.index.name='Problem'
    shown=ranks.copy();shown.index=['T'+str(i) for i in ranks.index];shown.index.name='Problem'
    dims=pd.DataFrame([dict(Problem='T'+str(p['problem']),D=p['dimension'],Population=200,Iterations=1000,
        Evaluation_budget='not used',Seeds='0,1,2') for p in cfg['problems']]).set_index('Problem')
    dims.to_csv(folder/'problem_budgets.csv')
    methodology=('Ranks compare the mean final objective value over seeds 0–2; lower is better. '
        'Exact ties at stored numerical precision receive average ranks. Ranks are computed before display rounding. '
        f'Problems excluded because of incomplete outcomes: {sorted(set(range(1,23))-set(complete))}. '
        f'Overall scores average the {len(complete)} complete within-problem ranks with equal problem weights. '
        'Standard deviation uses the sample denominator n−1. Three seeds provide preliminary descriptive results; '
        f'ranks do not establish statistical significance. Population 200 and 1000 iterations: {len(valid)} successful trials and {len(df)-len(valid)} failed trials. Failed partial results are not ranked. '
        f'Non-finite candidate evaluations ({int(df.nonfinite_candidates.sum()):,} total) counted in actual evaluation totals '
        'and were rejected; their counts are retained in audited_trial_results.csv.')
    if plan.get('cma_stabilization'):methodology+=' All 66 CMA-ES trials were rerun with the same covariance safeguards.'
    if plan.get('ga_fitness_reuse'):methodology+=' GA-MPC uses fitness reuse; stopping rule: '+plan['ga_fitness_reuse']['stopping_rule']+'.'
    if plan.get('ga_evaluation_cap'):
        methodology+=' GA-MPC was rerun with a maximum of 200,000 evaluations; the other algorithms use iteration-stopped results, whose actual evaluation counts can differ.'
    page='''<!doctype html><meta charset="utf-8"><title>CEC2011 algorithm ranking tables</title>
<style>body{font:15px system-ui;margin:28px;color:#172434}h1,h2{color:#183b59}.wide{overflow-x:auto}
table{border-collapse:collapse;white-space:nowrap;margin:16px 0}td,th{padding:9px 12px;border-bottom:1px solid #ddd;text-align:right}
th{background:#edf3f8}tbody tr:nth-child(even){background:#f7f9fb}p{max-width:1100px;line-height:1.5}</style>
<h1>CEC2011: algorithm ranks by problem</h1>'''
    page+='<p>'+html.escape(methodology)+'</p><p><a href="ranks_by_problem.csv">Rank matrix CSV</a> · '
    page+='<a href="performance_and_ranks.csv">Numeric performance and ranks CSV</a> · <a href="overall_ranks.csv">Overall ranks CSV</a></p>'
    for title,table in [('Overall ranking',overall),('Ranks for every problem',shown),
                        ('Mean ± sample standard deviation (rank)',formatted),('Problem configurations',dims)]:
        page+='<h2>'+title+'</h2><div class="wide">'+table.to_html(float_format=lambda x:f'{x:.4f}'.rstrip('0').rstrip('.'))+'</div>'
    (folder/'index.html').write_text(page,encoding='utf-8')
    lines=['# CEC2011 algorithm ranking tables','',methodology,'','## Overall ranks','',overall.to_markdown(floatfmt='.4f'),
           '','## Ranks by problem','',shown.to_markdown(floatfmt='g'),'','## Mean ± standard deviation (rank)','',formatted.to_markdown()]
    (folder/'ranking_tables.md').write_text('\n'.join(lines),encoding='utf-8')
    (folder/'audit.json').write_text(json.dumps(dict(trials=594,problems=22,algorithms=9,seeds=3,
        complete_ranked_problems=complete,excluded_problems=sorted(set(range(1,23))-set(complete)),failed_trials=len(df)-len(valid),finite_successful_final_scores=True,stopping_rule="1000 iterations",unique_trial_keys=True,
        rank_sums_verified=True,ranking_statistic='arithmetic mean final objective',ties='exact average ranks'),indent=2))
    index=output/'index.html';s=index.read_text(encoding='utf-8')
    if 'tables/index.html' not in s:s=s.replace('<h1>','<p><a href="tables/index.html">Algorithm ranking tables (complete problem blocks)</a></p><h1>',1)
    index.write_text(s,encoding='utf-8')
    print(overall.to_string());print('\n'+shown.to_markdown(floatfmt='g'))


if __name__=='__main__':main()
