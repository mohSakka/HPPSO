"""Within-problem exact Wilcoxon tests and descriptive dimension/rank plots."""
import argparse
from itertools import combinations,product
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exact_signed_rank(d):
    d=np.asarray(d);d=d[d!=0]
    if not len(d):return 0.,0.,1.
    ranks=rankdata(np.abs(d));positive=float(ranks[d>0].sum());negative=float(ranks[d<0].sum())
    observed=abs(positive-negative)
    null=np.asarray(list(product((-1,1),repeat=len(d))))@ranks
    p=float(np.mean(np.abs(null)>=observed-1e-12))
    return positive,negative,p


def write_significance_ranks(folder, tests, summary):
    summary = summary.assign(net_wins=summary.significant_wins - summary.significant_losses)
    summary['shared_rank'] = summary.groupby('problem').net_wins.rank(method='average', ascending=False)
    ranks = summary.pivot(index='problem', columns='algorithm', values='shared_rank')
    stats = ranks.agg(['mean', 'median', 'min', 'max']).T
    stats['mode'] = [', '.join(f'{value:g}' for value in ranks[a].mode()) for a in stats.index]
    stats['problems'] = len(ranks)
    ranks.to_csv(folder / 'significance_ranks_by_problem.csv')
    stats.to_csv(folder / 'significance_rank_statistics.csv')
    summary.to_csv(folder / 'wilcoxon_summary.csv', index=False)
    count = int(tests.p_holm.lt(.05).sum())
    seeds = int(tests.n_seeds.iloc[0])
    minimum_p = 2 ** (1 - seeds)
    page = (
        '<!doctype html><meta charset="utf-8"><title>Per-problem Wilcoxon ranks</title>'
        '<h1>Per-problem Wilcoxon ranks</h1>'
        '<p>Exact two-sided paired signed-rank tests compare final best objectives within each problem '
        f'using {seeds} matched seeds. Zero differences are omitted and magnitude ties get average ranks. '
        'Holm correction applies separately to each problem’s 36 comparisons at the 5% level.</p>'
        f'<p>{count} of {len(tests)} comparisons are significant. The smallest raw p-value is '
        f'{tests.p_raw.min():g}. With {seeds} nonzero pairs, the two extreme sign patterns out of {2 ** seeds} '
        f'give a minimum possible two-sided p-value of {minimum_p:g}.</p>'
        '<p>Positions are based on significant wins minus losses. Equal net wins share the average '
        'of the occupied ranks. When no comparison is significant, all nine algorithms receive rank 5.</p>'
        '<p><a href="pairwise_tests.csv">All pairwise tests</a> · '
        '<a href="significance_ranks_by_problem.csv">Per-problem ranks</a> · '
        '<a href="significance_rank_statistics.csv">Rank statistics</a></p>'
        '<h2>Statistics across problems</h2>' + stats.to_html(float_format=lambda value: f'{value:g}')
        + '<h2>Shared ranks by problem</h2>' + ranks.to_html(float_format=lambda value: f'{value:g}')
    )
    (folder / 'index.html').write_text(page, encoding='utf-8')
    print(stats.to_string())
    print(f'Significant comparisons: {count}/{len(tests)}. Minimum raw p: {tests.p_raw.min():g}')


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results/cec2011_pop200_fe200k')
    parser.add_argument('--significance-only', action='store_true',
                        help='Use the validated seed score export and report significance-based positions only')
    parser.add_argument('--scores', type=Path, help='Use the combined audited score export from a seed extension')
    args=parser.parse_args(argv);out=args.output;plan=json.loads((out/'plan.json').read_text())
    seeds = [0, 1, 2]
    if args.significance_only:
        if args.scores:
            cfg = plan['config']
            seeds = cfg['seeds']
            df = pd.read_csv(args.scores)
            assert df.status.eq('budget_reached').all() and df.evaluations.eq(cfg['evaluation_budget']).all()
            dimensions = {p['problem']: p['dimension'] for p in cfg['problems']}
            assert set(df.algorithm) == set(cfg['algorithms'])
        else:
            df = pd.read_csv(out / 'tables/seed_ranks.csv', usecols=['problem', 'seed', 'algorithm', 'best_objective'])
            df = df.rename(columns={'best_objective': 'best_score'})
            dimensions = {job['problem']: job['dimension'] for job in plan['jobs']}
        algos = sorted(df.algorithm.unique())
        assert set(df.problem) == set(dimensions) and set(df.seed) == set(seeds)
        assert len(df) == 22 * 9 * len(seeds) and len(algos) == 9
    else:
        cfg = plan['config']
        df = pd.read_csv(out / 'tables/audited_trial_results.csv')
        assert len(df)==594 and df.status.eq('budget_reached').all() and df.evaluations.eq(200000).all()
        algos=cfg['algorithms'];dimensions={p['problem']:p['dimension'] for p in cfg['problems']}
    assert not df.duplicated(['problem','algorithm','seed']).any()
    pairs=[];summaries=[]
    for number,block in df.groupby('problem'):
        values=block.pivot(index='seed',columns='algorithm',values='best_score').reindex(index=seeds,columns=algos)
        assert values.shape==(len(seeds),9) and np.isfinite(values.to_numpy()).all()
        rows=[]
        for a,b in combinations(algos,2):
            x=values[a].to_numpy();y=values[b].to_numpy();d=x-y
            d[np.isclose(x,y,rtol=1e-12,atol=1e-12)]=0
            d=np.array([float(f'{v:.12g}') for v in d]);plus,minus,p=exact_signed_rank(d)
            rows.append(dict(problem=number,dimension=dimensions[number],algorithm_a=a,algorithm_b=b,
                n_seeds=len(seeds),nonzero_pairs=int(np.count_nonzero(d)),a_better=int((d<0).sum()),
                a_worse=int((d>0).sum()),ties=int((d==0).sum()),W_plus=plus,W_minus=minus,
                p_raw=p,p_holm=None))
        order=sorted(range(36),key=lambda i:rows[i]['p_raw']);previous=0
        for j,i in enumerate(order):
            previous=max(previous,min(1,rows[i]['p_raw']*(36-j)));rows[i]['p_holm']=previous
        for a in algos:
            involved=[r for r in rows if a in (r['algorithm_a'],r['algorithm_b'])]
            wins=losses=0
            for r in involved:
                if r['p_holm']>=.05:continue
                winner=r['algorithm_a'] if r['W_minus']>r['W_plus'] else r['algorithm_b']
                if winner==a:wins+=1
                else:losses+=1
            summaries.append(dict(problem=number,dimension=dimensions[number],algorithm=a,
                significant_wins=wins,significant_losses=losses,not_significant=8-wins-losses,
                conclusion='Not distinguished' if wins==losses==0 else 'See pairwise tests'))
        pairs.extend(rows)
    tests=pd.DataFrame(pairs);summary=pd.DataFrame(summaries)
    assert len(tests)==792 and tests.p_raw.min() >= 2 ** (1 - len(seeds))
    folder=out/'tables/per_problem_wilcoxon';folder.mkdir(parents=True,exist_ok=True)
    tests.to_csv(folder/'pairwise_tests.csv',index=False);summary.to_csv(folder/'wilcoxon_summary.csv',index=False)
    if args.significance_only:
        write_significance_ranks(folder, tests, summary)
        return
    stats = pd.read_csv(out / 'tables/performance_and_ranks.csv')
    ranks=stats.pivot(index='problem',columns='algorithm',values='rank').reindex(columns=algos)
    assert np.allclose(ranks.sum(axis=1),45)
    rank_table=ranks.copy();rank_table.insert(0,'dimension',[dimensions[i] for i in ranks.index])
    rank_table.to_csv(folder/'descriptive_ranks_by_problem.csv')
    associations=[]
    for a in algos:
        rho=float(spearmanr([dimensions[i] for i in ranks.index],ranks[a]).statistic)
        associations.append(dict(algorithm=a,spearman_rho_dimension_vs_descriptive_rank=rho))
    associations=pd.DataFrame(associations).set_index('algorithm');associations.to_csv(folder/'dimension_rank_associations.csv')
    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    for k,(a,ax) in enumerate(zip(algos,axes.flat)):
        x=np.array([dimensions[i] for i in ranks.index]);y=ranks[a].to_numpy()
        color=plt.get_cmap('tab10')(k)
        ax.scatter(x,y,color=color,alpha=.55,s=32,label='Individual problem rank')
        grouped=pd.DataFrame(dict(dimension=x,rank=y)).groupby('dimension')['rank'].mean()
        ax.plot(grouped.index,grouped.values,color=color,marker='o',markersize=3,lw=1.2,label='Mean at the same dimension')
        ax.set_xscale('log');ax.set_xlim(.8,300);ax.set_ylim(9.4,.6);ax.set_yticks([1,3,5,7,9]);ax.grid(alpha=.2)
        ax.set_xlabel('Native dimension (log scale)');ax.set_ylabel('Mean-objective rank (1 = best)')
        ax.set_title(f'{a}; Spearman rho={associations.loc[a].iloc[0]:+.2f}',fontsize=10)
    axes[0,0].legend(fontsize=7,loc='lower left')
    fig.suptitle('CEC2011: dimensionality versus descriptive algorithm rank\nDifferent functions at their native dimensions; population 200, equal 200,000 evaluations, three seeds')
    fig.savefig(folder/'dimension_vs_rank.png',dpi=180);fig.savefig(folder/'dimension_vs_rank.pdf');plt.close(fig)
    notes=('Tests compare paired seeds 0–2 within each problem, using exact two-sided sign enumeration '
        'of Wilcoxon signed ranks. Zero differences are omitted; magnitude ties receive average ranks. '
        'Numerical ties use rtol=atol=1e-12; remaining differences use 12 significant digits. '
        'Holm correction is applied separately to each problem’s 36 comparisons (alpha=0.05). '
        'With only three nonzero pairs, the minimum attainable two-sided p-value is 2/8=0.25. '
        'Therefore none of the 792 comparisons can be significant, even without correction. '
        'The Wilcoxon results provide no per-problem ordering: not distinguished does not prove equivalence. '
        'The numerical rank table and dimension plot instead use the saved mean-final-objective ranks, '
        'with average ranks for exact mean ties. They are descriptive, not Wilcoxon ranks. '
        'The dimension association is confounded by problem identity, constraints and landscape; it is not a controlled scaling experiment. '
        'The fixed 200,000-call budget also gives fewer evaluations per dimension at larger D. '
        'Points are individual problems; lines connect average ranks for problems sharing a dimension, not fitted scaling curves. '
        'Spearman rho describes this selected suite only (negative means better ranks at larger D). '
        'T18 and T20 have identical recorded mean profiles and both appear in the dataset; no independence claim is made for them.')
    style='<style>body{font:15px system-ui;margin:25px}p{line-height:1.5;max-width:1200px}table{border-collapse:collapse;white-space:nowrap}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf3f8}.wide{overflow:auto}img{max-width:1200px;width:100%}</style>'
    status=summary.pivot(index='problem',columns='algorithm',values='conclusion').reindex(columns=algos)
    page='<!doctype html><meta charset="utf-8">'+style+'<h1>Per-problem Wilcoxon tests and dimensionality</h1><p>'+notes+'</p>'
    page+='<p><a href="pairwise_tests.csv">All 792 tests CSV</a> · <a href="descriptive_ranks_by_problem.csv">Per-problem descriptive ranks CSV</a> · <a href="dimension_vs_rank.pdf">Plot PDF</a></p>'
    page+='<h2>Dimension versus descriptive rank</h2><img src="dimension_vs_rank.png">'
    for title,table in [('Per-problem descriptive ranks (mean objective)',rank_table),('Wilcoxon outcomes: all not distinguished',status),('Descriptive dimension associations',associations)]:
        page+='<h2>'+title+'</h2><div class="wide">'+table.to_html(float_format=lambda v:f'{v:.4g}')+'</div>'
    (folder/'index.html').write_text(page,encoding='utf-8')
    (folder/'method.json').write_text(json.dumps(dict(test='exact paired two-sided Wilcoxon per problem',
        seed_pairs=3,problems=22,tests=792,minimum_attainable_p=.25,minimum_observed_p=float(tests.p_raw.min()),
        correction='Holm within each problem',significant_comparisons=0,dimension_plot_ranks='descriptive mean-objective ranks'),indent=2))
    for file,link in [(out/'index.html','tables/per_problem_wilcoxon/index.html'),(out/'tables/index.html','per_problem_wilcoxon/index.html')]:
        text=file.read_text(encoding='utf-8')
        if link not in text:text=text.replace('<h1>',f'<p><a href="{link}">Per-problem Wilcoxon and dimension/rank plot</a></p><h1>',1)
        file.write_text(text,encoding='utf-8')
    print('Raw p-value range:',tests.p_raw.min(),tests.p_raw.max());print(associations.to_string())


if __name__=='__main__':main()
