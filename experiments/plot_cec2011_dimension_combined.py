"""All nine algorithm rank/dimension profiles on one shared axis."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,ScalarFormatter


def main():
    folder=Path(__file__).resolve().parent/'results/cec2011_pop200_fe200k/tables/per_problem_wilcoxon'
    df=pd.read_csv(folder/'descriptive_ranks_by_problem.csv')
    algorithms=[c for c in df if c not in ('problem','dimension')]
    grouped=df.groupby('dimension')[algorithms].mean()
    grouped.to_csv(folder/'mean_rank_by_dimension.csv')
    fig,ax=plt.subplots(figsize=(12,7),layout='constrained')
    markers=['o','s','^','D','v','P','X','<','>']
    for i,a in enumerate(algorithms):
        ax.plot(grouped.index,grouped[a],label='PSO' if a=='Original PSO' else a,
                color=plt.get_cmap('tab10')(i),marker=markers[i],markersize=5,
                lw=2.2 if a=='HPPSO' else 1.4,alpha=.95)
    ax.set_xscale('log');ax.set_xlim(.9,270);ax.set_ylim(9.35,.65);ax.set_yticks(range(1,10))
    ticks=[1,6,10,20,40,96,140,240]
    ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('Native problem dimension (log scale)')
    ax.set_ylabel('Mean descriptive rank at each dimension (1 = best)')
    ax.set_title('CEC2011: algorithm rank versus dimensionality\nPopulation 200; 200,000 evaluations per algorithm; three seeds')
    ax.grid(alpha=.25);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=5,fontsize=10)
    fig.savefig(folder/'dimension_vs_rank_combined.png',dpi=180)
    fig.savefig(folder/'dimension_vs_rank_combined.pdf');plt.close(fig)
    page=folder/'index.html';s=page.read_text(encoding='utf-8')
    replacement='<h2>All algorithms on one axis</h2><p>Points average mean-objective ranks across problems sharing a native dimension. These are descriptive ranks, not statistically resolved within-problem Wilcoxon ranks. Different functions occur at different dimensions.</p><p><a href="dimension_vs_rank_combined.pdf">Combined plot PDF</a></p><img src="dimension_vs_rank_combined.png"><h2>Individual algorithm panels</h2>'
    if 'dimension_vs_rank_combined.png' not in s:s=s.replace('<h2>Dimension versus descriptive rank</h2>',replacement)
    page.write_text(s,encoding='utf-8')
    print(folder/'dimension_vs_rank_combined.png')


if __name__=='__main__':main()
