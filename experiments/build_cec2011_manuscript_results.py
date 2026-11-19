from pathlib import Path
import gzip,json,hashlib,shutil,sys
from itertools import combinations
import numpy as np,pandas as pd
from scipy.stats import friedmanchisquare,studentized_range
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path.cwd()))
from experiments.final_benchmark_wilcoxon import exact_signed_rank
from experiments.cec2011_seed_ranks import ranks as numerical_ranks
from experiments.build_coordinatewise_manuscript_results import holm,COLORS,LABEL
root=Path('Springer_Nature_LaTeX_Template (6)');src=Path('experiments/results/cec2011_coordinate_refresh_10_seeds');out=root/'cec2011_results';out.mkdir(exist_ok=True)
plan=json.loads((src/'plan.json').read_text());cfg=plan['config'];alg=cfg['algorithms'];dims={x['problem']:x['dimension'] for x in cfg['problems']}
df=pd.read_csv(src/'tables/final_scores.csv',float_precision='round_trip')
assert len(df)==1980 and not df.duplicated(['problem','algorithm','seed']).any()
assert df.status.eq('budget_reached').all() and df.evaluations.eq(200000).all() and np.isfinite(df.best_score).all()
assert set(df.seed)==set(range(10)) and set(df.problem)==set(range(1,23)) and set(df.algorithm)==set(alg)
for row in df.itertuples():
 folder=src if row.seed>=3 else Path('experiments/results')/('cec2011_hppso_coordinate_refresh_200k' if row.algorithm=='HPPSO' else 'cec2011_pop200_fe200k')
 file=folder/'trials'/f'T{row.problem}'/row.algorithm.replace(' ','_')/f'seed_{row.seed:03d}.json.gz'
 with gzip.open(file,'rt') as f:t=json.load(f)
 assert t['best_score']==row.best_score and t['population']==200 and t['evaluations']==200000 and t['status']=='budget_reached'
 assert t['fingerprint']==row.fingerprint
print('Audited all 1,980 exported scores against saved trial files.',flush=True)
summary=df.groupby(['problem','algorithm']).best_score.agg(['mean','std','median','min','max']).reset_index();summary.to_csv(out/'numerical_results.csv',index=False)
pairs=[];details=[]
for p in range(1,23):
 wide=df[df.problem.eq(p)].pivot(index='seed',columns='algorithm',values='best_score').reindex(columns=alg)
 tests=[]
 for a,b in combinations(alg,2):
  plus,minus,pv=exact_signed_rank((wide[a]-wide[b]).to_numpy())
  tests.append(dict(problem=p,algorithm_a=a,algorithm_b=b,p_raw=pv,positive_rank_sum=plus,negative_rank_sum=minus))
 adj=holm([x['p_raw'] for x in tests]);wins=dict.fromkeys(alg,0);losses=dict.fromkeys(alg,0)
 for t,pv in zip(tests,adj):
  t['p_holm']=pv;t['winner']=''
  if pv<=.05:
   win,lose=(t['algorithm_a'],t['algorithm_b']) if t['negative_rank_sum']>t['positive_rank_sum'] else (t['algorithm_b'],t['algorithm_a'])
   wins[win]+=1;losses[lose]+=1;t['winner']=win
 net=pd.Series({a:wins[a]-losses[a] for a in alg});rr=net.rank(ascending=False,method='average')
 for a in alg:details.append(dict(problem=p,dimension=dims[p],algorithm=a,rank=rr[a],wins=wins[a],losses=losses[a],ties=8-wins[a]-losses[a]))
 pairs+=tests
pd.DataFrame(pairs).to_csv(out/'wilcoxon_pairs.csv',index=False);detail=pd.DataFrame(details);detail.to_csv(out/'significance_ranks.csv',index=False)
means=summary.pivot(index='problem',columns='algorithm',values='mean').reindex(columns=alg)
wide=df.pivot(index=['problem','seed'],columns='algorithm',values='best_score').reindex(columns=alg)
duplicates=[(a,b) for a,b in combinations(range(1,23),2) if np.array_equal(wide.loc[a].values,wide.loc[b].values)]
print('Identical score profiles',duplicates,flush=True)
fr=[];post=[];rank_summaries=[]
for name,mm in [('all_22',means),('omit_T20',means.drop(index=20))]:
 rr=np.array([numerical_ranks(x) for x in mm.values]);av=rr.mean(0);test=friedmanchisquare(*rr.T)
 fr.append(dict(analysis=name,n=len(mm),chi_square=float(test.statistic),p=float(test.pvalue)))
 for a,r in zip(alg,av):rank_summaries.append(dict(analysis=name,algorithm=a,mean_rank=r))
 for i,j in combinations(range(9),2):
  pv=studentized_range.sf(np.sqrt(2)*abs(av[i]-av[j])/np.sqrt(90/(6*len(mm))),9,np.inf)
  post.append(dict(analysis=name,algorithm_a=alg[i],algorithm_b=alg[j],p_nemenyi=pv,rank_a=av[i],rank_b=av[j]))
pd.DataFrame(fr).to_csv(out/'friedman_tests.csv',index=False);pd.DataFrame(post).to_csv(out/'nemenyi_pairs.csv',index=False);pd.DataFrame(rank_summaries).to_csv(out/'friedman_ranks.csv',index=False)
print(fr,flush=True)
rowend=r' \\'
lines=[r'\begin{table}[H]\centering\scriptsize',r'\caption{Within-problem Wilcoxon significance ranks for the ten-seed CEC2011 experiment. Ranks order significant wins minus losses against eight opponents, with averaged ties. The last column gives HPPSO wins/losses/nonsignificant comparisons. All 36 pairs per problem are Holm-adjusted at $\alpha=0.05$.}\label{tab:cec_significance}',r'\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.05}',r'\begin{tabular}{@{}lrrrrrrrrrrc@{}}\toprule','Problem & $d$ & '+' & '.join(LABEL[a] for a in alg)+' & W/L/T'+rowend,r'\midrule']
for p in range(1,23):lines.append(f'T{p} & {dims[p]} & '+' & '.join(['5']*9)+' & 0/0/8'+rowend)
lines +=[r'\midrule','Mean rank & --- & '+' & '.join(['5.00']*9)+' & ---'+rowend,'Median rank & --- & '+' & '.join(['5.00']*9)+' & ---'+rowend,r'\bottomrule\end{tabular}\end{table}']
(out/'significance_table.tex').write_text('\n'.join(lines))
av=pd.DataFrame(rank_summaries).query("analysis=='all_22'").set_index('algorithm').mean_rank
sensitivity=pd.DataFrame(rank_summaries).query("analysis=='omit_T20'").set_index('algorithm').mean_rank
lines=[r'\begin{table}[H]\centering\small',r'\caption{Across-problem Friedman mean ranks and Nemenyi comparisons with HPPSO. Ranks use the ten-seed mean cost within each problem; smaller is better. Post hoc probabilities account for all 36 algorithm pairs. The final column removes the duplicated T20 score profile.}\label{tab:cec2011_friedman_ranks}\label{tab:average_ranks2011}',r'\begin{tabular}{@{}lrrr@{}}\toprule',r'Algorithm & Mean rank (22) & $p_{\mathrm{Nemenyi}}$ vs HPPSO & Mean rank (21) \\\midrule']
for a in av.sort_values().index:
 pp=[t for t in post if t['analysis']=='all_22' and {t['algorithm_a'],t['algorithm_b']}=={a,'HPPSO'}]
 value='---' if not pp else (r'$<0.0001$' if pp[0]['p_nemenyi']<.0001 else f"{pp[0]['p_nemenyi']:.4f}")
 lines.append(f'{LABEL[a]} & {av[a]:.3f} & {value} & {sensitivity[a]:.3f}'+rowend)
lines +=[r'\bottomrule\end{tabular}\end{table}'];(out/'friedman_table.tex').write_text('\n'.join(lines))
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':7,'axes.titlesize':9,'xtick.labelsize':6,'ytick.labelsize':7,'pdf.fonttype':42})
for page,start in enumerate(range(1,23,6),1):
 problems=list(range(start,min(start+6,23)));nrows=(len(problems)+1)//2
 fig,axs=plt.subplots(nrows,2,figsize=(7.2,2.6*nrows),squeeze=False)
 for ax,p in zip(axs.flat,problems):
  arrays=[df[(df.problem==p)&(df.algorithm==a)].sort_values('seed').best_score.values for a in alg]
  boxes=ax.boxplot(arrays,patch_artist=True,showfliers=False,widths=.6,medianprops={'color':'black','linewidth':1})
  for i,(box,a,values) in enumerate(zip(boxes['boxes'],alg,arrays),1):
   box.set_facecolor(COLORS[a]);box.set_alpha(.55)
   ax.scatter(i+np.linspace(-.17,.17,10),values,s=5,c=COLORS[a],edgecolors='none',zorder=3)
  flat=np.concatenate(arrays)
  if flat.min()>0 and flat.max()/flat.min()>100:ax.set_yscale('log')
  else:ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4),useMathText=True)
  if flat.min()>0 and np.ptp(flat)<=1e-12*max(1,float(np.max(np.abs(flat)))):
   ax.set_ylim(0,1.15*flat.max());ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4),useMathText=True,useOffset=False)
  ax.set_title(f'T{p} ({dims[p]}D)');ax.set_xticks(range(1,10),[LABEL[a] for a in alg],rotation=55,ha='right');ax.set_ylabel('Final best cost');ax.grid(axis='y',alpha=.2)
  ax.spines[['top','right']].set_visible(False)
 for ax in list(axs.flat)[len(problems):]:ax.axis('off')
 fig.tight_layout(pad=1.2,h_pad=2,w_pad=1.5);fig.savefig(out/f'boxplots_{page}.pdf');fig.savefig(out/f'boxplots_{page}.png',dpi=180);plt.close(fig)
 print('Boxplots',page,flush=True)
lines=[r'\section{CEC2011 Numerical Results and Cost Distributions}\label{app:cec_results}',r'The numerical table reports the final objective values over ten seeds. Figures~\ref{fig:cec_boxes_1}--\ref{fig:cec_boxes_4} include all nine algorithms, including CSA. Each box spans the interquartile range, its line marks the median, and whiskers extend to the most extreme observations within 1.5 interquartile ranges. All ten observations are overlaid as points. Positive costs spanning more than two orders of magnitude use logarithmic axes; the remaining panels use linear axes. T3 is plotted from zero to avoid magnifying numerical roundoff. These values are the implemented benchmark objectives, including any built-in penalties, and do not separately establish constraint feasibility.',r'\begin{table}[htbp]\centering\scriptsize',r'\caption{Mean final objective and sample standard deviation (in parentheses) over ten seeds. No per-problem significance ordering is established.}\label{tab:cec2011_performance}',r'\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{1.08}',r'\begin{tabular}{@{}lrrrrrrrrr@{}}\toprule','Problem & '+' & '.join(LABEL[a] for a in alg)+rowend,r'\midrule']
for p in range(1,23):
 b=summary[summary.problem==p].set_index('algorithm')
 cells=[rf"\shortstack[r]{{{b.loc[a,'mean']:.2e}\\({b.loc[a,'std']:.2e})}}" for a in alg]
 lines.append(f'T{p} & '+' & '.join(cells)+rowend+'[3pt]')
lines +=[r'\bottomrule\end{tabular}\end{table}']
for i,start in enumerate(range(1,23,6),1):
 lines +=[r'\begin{figure}[p]\centering',rf'\includegraphics[width=\textwidth,height=0.86\textheight,keepaspectratio]{{cec2011_results/boxplots_{i}.pdf}}',rf'\caption{{Final-cost distributions for T{start}--T{min(start+5,22)} over ten seeds, with population 200 and 200,000 evaluations per run. All nine algorithms, including CSA, are shown.}}\label{{fig:cec_boxes_{i}}}',r'\end{figure}']
(out/'appendix.tex').write_text('\n'.join(lines))
manifest=dict(source=str(src),sha256=hashlib.sha256((src/'tables/final_scores.csv').read_bytes()).hexdigest(),audited_trials=1980,significant_wilcoxon_pairs=int(sum(t['p_holm']<=.05 for t in pairs)),paired_comparisons=792,duplicate_profiles=duplicates,friedman=fr,ranking_tolerance='Friedman only: rtol=atol=1e-12, anchored at smallest value in each tied group',hppso=cfg['hppso'])
(out/'audit.json').write_text(json.dumps(manifest,indent=2));shutil.copyfile(src/'plan.json',out/'source_plan.json')
