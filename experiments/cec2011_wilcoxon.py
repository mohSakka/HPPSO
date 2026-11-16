"""Paired Wilcoxon signed-rank comparisons over CEC2011 problem means."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata,wilcoxon


def analyze(means):
    rows=[]
    for a,b in combinations(means.columns,2):
        x=means[a].to_numpy();y=means[b].to_numpy();d=x-y
        d[np.isclose(x,y,rtol=1e-12,atol=1e-12)]=0
        d=np.array([float(f'{v:.12g}') for v in d]);nonzero=d[d!=0]
        ranks=rankdata(abs(nonzero));plus=ranks[nonzero>0].sum();minus=ranks[nonzero<0].sum()
        test=wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto') if len(nonzero) else None
        rows.append(dict(algorithm_a=a,algorithm_b=b,n_problems=len(d),a_better=int((d<0).sum()),
            a_worse=int((d>0).sum()),ties=int((d==0).sum()),W_plus=plus,W_minus=minus,
            statistic=0 if test is None else test.statistic,p_raw=1 if test is None else test.pvalue))
    frame=pd.DataFrame(rows);order=np.argsort(frame.p_raw.to_numpy(),kind='stable')
    adjusted=np.empty(len(frame));last=0
    for j,i in enumerate(order):last=max(last,min(1,frame.iloc[i].p_raw*(len(frame)-j)));adjusted[i]=last
    frame['p_holm']=adjusted;frame['significant']=frame.p_holm<.05
    frame['winner']=[r.algorithm_a if r.W_minus>r.W_plus else r.algorithm_b if r.W_minus<r.W_plus else 'none' for r in frame.itertuples()]
    frame.loc[~frame.significant,'winner']='not significant'
    scores=[]
    for a in means.columns:
        involved=frame[(frame.algorithm_a==a)|(frame.algorithm_b==a)]
        wins=int((involved.winner==a).sum());losses=int((involved.significant&(involved.winner!=a)).sum())
        scores.append(dict(algorithm=a,significant_wins=wins,not_significant=8-wins-losses,
            significant_losses=losses,net_wins=wins-losses))
    summary=pd.DataFrame(scores).set_index('algorithm').sort_values('net_wins',ascending=False,kind='stable')
    summary.insert(0,'summary_position',summary.net_wins.rank(method='min',ascending=False).astype(int))
    return frame,summary


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results/cec2011_pop200_ga_reuse_budget200k')
    args=parser.parse_args();folder=args.output/'tables'
    stats=pd.read_csv(folder/'performance_and_ranks.csv')
    assert stats['count'].eq(3).all() and len(stats)==198
    means=stats.pivot(index='problem',columns='algorithm',values='mean')
    assert means.shape==(22,9) and np.isfinite(means.to_numpy()).all()
    pairs,summary=analyze(means);pairs.to_csv(folder/'wilcoxon_pairs.csv',index=False)
    summary.to_csv(folder/'wilcoxon_summary.csv')
    matrix=pd.DataFrame(1.,index=means.columns,columns=means.columns)
    for r in pairs.itertuples():matrix.loc[r.algorithm_a,r.algorithm_b]=matrix.loc[r.algorithm_b,r.algorithm_a]=r.p_holm
    matrix.to_csv(folder/'wilcoxon_holm_matrix.csv')
    hp=[]
    for r in pairs.itertuples():
        if 'HPPSO' not in (r.algorithm_a,r.algorithm_b):continue
        hp.append(dict(comparator=r.algorithm_b if r.algorithm_a=='HPPSO' else r.algorithm_a,
            p_raw=r.p_raw,p_holm=r.p_holm,result='not significant' if not r.significant else 'HPPSO better' if r.winner=='HPPSO' else 'HPPSO worse'))
    hp=pd.DataFrame(hp);hp.to_csv(folder/'wilcoxon_hppso.csv',index=False)
    identical=[(a,b) for a,b in combinations(means.index,2) if np.array_equal(means.loc[a],means.loc[b])]
    notes=('Two-sided paired Wilcoxon signed-rank tests on 22 per-problem means, each based on three seeds. '
        'Holm familywise correction over all 36 pairs, alpha=0.05. Differences are A minus B; W− favors A. '
        'Numerical ties use rtol=atol=1e-12; remaining differences are rounded to 12 significant digits. '
        'Zero differences are dropped (wilcox); SciPy method=auto handles the remaining ties/zeros. '
        'The summary position is a descriptive ordering by significant wins minus losses, with shared positions for equal scores. '
        'It is not an inherent Wilcoxon ranking, and equal positions do not establish statistical equivalence. '
        'Raw objective differences have different scales across problems, which can affect their absolute-rank weights. '
        'The signed-rank location interpretation assumes a symmetric distribution of paired differences; '
        'these are preliminary results on selected benchmark instances, not proof of universal superiority.')
    sensitivity=''
    if identical:
        remove=sorted(set(b for a,b in identical));p2,s2=analyze(means.drop(index=remove))
        p2.to_csv(folder/'wilcoxon_pairs_distinct_profiles.csv',index=False);s2.to_csv(folder/'wilcoxon_summary_distinct_profiles.csv')
        sensitivity=(f'<p>Identical recorded mean profiles: {identical}. This does not by itself prove identical problem definitions. '
            f'Sensitivity analysis omits T{remove[0]} ({len(means)-len(remove)} remaining problems); verify evaluator definitions before publication.</p>'+s2.to_html())
    page='<!doctype html><meta charset="utf-8"><title>CEC2011 Wilcoxon comparisons</title>'
    page+='<style>body{font:15px system-ui;margin:25px;max-width:1500px}p{line-height:1.5}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf3f8}.wide{overflow:auto}</style>'
    page+='<h1>Wilcoxon signed-rank comparisons</h1><p>'+notes+'</p><p><a href="wilcoxon_pairs.csv">All 36 comparisons CSV</a> · <a href="wilcoxon_summary.csv">Summary CSV</a></p>'
    for title,table in [('Descriptive net-wins summary',summary),('HPPSO comparisons',hp),('Holm-adjusted p-value matrix',matrix),('All pairwise tests',pairs)]:
        page+='<h2>'+title+'</h2><div class="wide">'+table.to_html(index=table is summary or table is matrix,float_format=lambda x:f'{x:.6g}')+'</div>'
    page+='<h2>Sensitivity to identical recorded profiles</h2>'+sensitivity
    page+='<p>Method reference: <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html">SciPy Wilcoxon documentation</a>.</p>'
    (folder/'wilcoxon.html').write_text(page,encoding='utf-8')
    for f,link in [(args.output/'index.html','tables/wilcoxon.html'),(folder/'index.html','wilcoxon.html')]:
        s=f.read_text(encoding='utf-8')
        if link not in s:s=s.replace('<h1>',f'<p><a href="{link}">Wilcoxon comparisons</a></p><h1>',1)
        f.write_text(s,encoding='utf-8')
    (folder/'wilcoxon_method.json').write_text(json.dumps(dict(problem_pairs=22,seed_means=3,tests=36,
        correction='Holm',alpha=.05,tie_rtol=1e-12,tie_atol=1e-12,zero_method='wilcox',method='auto',identical_profiles=identical),indent=2))
    print(summary.to_string());print(hp.to_string(index=False))
    print('SHADE vs CMA-ES:',pairs[((pairs.algorithm_a=='CMA-ES')&(pairs.algorithm_b=='SHADE'))].to_string(index=False))


if __name__=='__main__':main()
