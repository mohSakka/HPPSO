"""Replay the 2D coverage experiment and connect peer geometry to visits."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from experiments.openness_mechanism.coverage_2d import RULES, BatchedTraitPSO, BatchBudget, BatchFunction, trait_suite
import numpy as np


ROOT = Path(__file__).with_name('full_run_coverage_2d')
PHASES = (10, 100, 1199)


def replay(function, seed, plan):
    settings, spec = plan['settings'], plan['hppso_spec']
    n = plan['population']
    problem = trait_suite(2, settings['instance_seed'])[function]
    tracker = BatchBudget(BatchFunction(problem), plan['budget'], 4, 2, settings['trace_points'])
    specs = [spec, dict(spec, peer_rule='better'), dict(spec, peer_rule='any'),
             dict(spec, traits=[t for t in spec['traits'] if t != 'openness'])]
    engine = BatchedTraitPSO(tracker, problem['bounds'], n, seed, specs, settings)
    peer_rng = np.random.default_rng(np.random.SeedSequence([seed, 902]))
    grid = np.zeros((4, 10000), dtype=int)
    lanes = np.arange(4)[:, None]
    candidates, observed, examples = [], [], []
    for epoch in range(1, plan['budget']//n):
        peer = peer_rng.integers(0,n-1,n)
        peer += peer >= np.arange(n)
        worse = engine.scores[:,peer] > engine.scores
        better = engine.scores[:,peer] < engine.scores
        open_ = engine.traits[:,:,3] > engine.theta[:,None]
        active = open_ & np.stack((worse[0],better[1],np.ones(n,bool),np.zeros(n,bool)))
        distance = np.linalg.norm((engine.pbests-engine.global_best[:,None,:])/engine.width,axis=-1)/np.sqrt(2)
        worse_set = engine.scores[0,None,:] > engine.scores[0,:,None]
        better_set = engine.scores[0,None,:] < engine.scores[0,:,None]
        paired = open_[0] & worse_set.any(axis=1) & better_set.any(axis=1)
        if paired.any():
            means = [(mask[paired] @ distance[0])/mask[paired].sum(axis=1) for mask in (worse_set,better_set)]
            candidates.append((epoch, *[a.mean() for a in means]))
        if seed == 0 and epoch == 5:
            choices = np.flatnonzero(paired & active[0])
            if choices.size:
                i = choices[0]
                j = peer[i]
                k = np.flatnonzero(better_set[i])[0]
                points = np.stack((engine.positions[0,i],engine.global_best[0],engine.pbests[0,j],engine.pbests[0,k]))
                examples.append((points-engine.lower)/engine.width)
        engine.step()
        unit = (engine.positions-engine.lower)/engine.width
        bins = np.clip((100*unit).astype(int),0,99)
        ids = bins[...,1]*100 + bins[...,0]
        old = grid[lanes,ids]
        for i, rule in enumerate(RULES):
            novel = old[i] == 0
            active_only = np.setdiff1d(ids[i,active[i] & novel],ids[i,~active[i]])
            selected = distance[i,peer[active[i]]]
            observed.append((epoch,rule,len(selected),selected.sum(),len(np.unique(ids[i,novel])),len(active_only)))
        grid[lanes,ids] = np.where(old==0,epoch,old)
    assert tracker.count == plan['budget']
    return function,seed,candidates,observed,grid,examples


def main():
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from experiments.openness_mechanism.analyze import COLORS

    plan = json.loads((ROOT/'plan.json').read_text())
    names = plan['settings']['function_display_names']
    jobs = [(f,s,plan) for f in plan['functions'] for s in plan['seeds']]
    candidate_rows, observations, examples = [], [], []
    with np.load(ROOT/'first_visit_maps.npz') as previous, ProcessPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(replay,*job) for job in jobs]
        for task in tasks:
            f,s,c,o,grid,e = task.result()
            np.testing.assert_array_equal(grid.reshape(4,100,100),previous[f'{f}_{s:03d}_100'])
            candidate_rows.extend((f,s,*row) for row in c)
            observations.extend((f,s,*row) for row in o)
            examples.extend((f,points) for points in e)
            print(f'{f}, seed {s}: coverage reproduced',flush=True)
    c = pd.DataFrame(candidate_rows,columns=['function','seed','epoch','worse_distance','better_distance'])
    o = pd.DataFrame(observations,columns=['function','seed','epoch','rule','interactions','distance_sum','new_cells','active_only_new_cells'])
    c.to_csv(ROOT/'peer_candidate_distances.csv',index=False)
    o.to_csv(ROOT/'peer_visits_by_epoch.csv',index=False)
    o['phase'] = pd.cut(o.epoch,[0,*PHASES],labels=['1–10','11–100','101–1199'])
    early = c[c.epoch.le(10)].groupby(['function','seed'])[['worse_distance','better_distance']].mean()
    phases = o.groupby(['function','seed','rule','phase'],observed=True)[['interactions','distance_sum','new_cells','active_only_new_cells']].sum()
    phases['selected_distance'] = phases.distance_sum / phases.interactions.replace(0,np.nan)
    phases.to_csv(ROOT/'peer_mechanism_by_seed.csv')
    early.to_csv(ROOT/'paired_candidate_distances_first10.csv')

    plot_peer_coverage(ROOT, plan, early, phases, o)
    if examples:
        plot_peer_geometry(names, examples, COLORS, plt)
    print(early.groupby('function').mean().to_string())
    print(phases.groupby(['rule','phase'],observed=True)[['new_cells','active_only_new_cells','selected_distance']].mean().to_string())


def plot_peer_coverage(output, plan, early, phases, visits, rules=RULES, *, show_candidates=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from experiments.openness_mechanism.analyze import COLORS

    names = plan['settings']['function_display_names']
    labels = ('Worse peer','Better peer','Random peer','Openness off')
    labels = dict(zip(RULES, labels, strict=True))
    interacting = [rule for rule in rules if rule != 'None']
    ncols = 3 if show_candidates else 2
    fig, axes = plt.subplots(len(plan['functions']), ncols, figsize=(12,12),
                             layout='constrained', squeeze=False)
    for i,f in enumerate(plan['functions']):
        if show_candidates:
            ax = axes[i,0]
            for row in early.loc[f].itertuples():
                ax.plot([0,1],[row.better_distance,row.worse_distance],'o-',ms=3,color='#71828c',alpha=.6)
            ax.set_xticks([0,1],['Better candidates','Worse candidates'])
        ax = axes[i,-2]
        ax.set_title(names[f], loc='left', fontweight='bold', fontsize=10)
        ax.set_ylabel('Normalized distance')
        for j,rule in enumerate(interacting):
            q = phases.xs((f,rule,'1–10'),level=('function','rule','phase')).selected_distance
            ax.scatter(np.full(len(q),j)+np.linspace(-.12,.12,len(q)),q,color=COLORS[rule],s=12)
            ax.plot(j,q.mean(),'_',color='black',ms=14)
        ax.set_xticks(range(len(interacting)),[labels[rule] for rule in interacting],fontsize=8)
        ax = axes[i,-1]
        ax.set_title(names[f], loc='left', fontweight='bold', fontsize=10)
        ax.set_ylabel('Grid cells visited (%)')
        for rule in rules:
            cells = visits[visits.function.eq(f)&visits.rule.eq(rule)].pivot(index='epoch',columns='seed',values='new_cells')
            ax.plot(cells.index,cells.cumsum().mean(axis=1)/100,label=labels[rule],color=COLORS[rule])
        ax.set_xscale('log')
        for ax in axes[i]: ax.grid(axis='y',alpha=.22)
    if show_candidates:
        axes[0,0].set_title('Candidate distances in the same swarm')
    axes[0,-2].set_title('(a) Peer distance to global best', pad=13)
    axes[0,-1].set_title('(b) Cumulative search coverage', pad=13)
    axes[-1,-1].set_xlabel('Swarm update (log scale)')
    axes[-1,-1].legend(fontsize=8)
    fig.savefig(output/'peer_distance_and_coverage.pdf')
    plt.close(fig)


def plot_peer_geometry(names, examples, COLORS, plt):
        fig, axes = plt.subplots(1,len(examples),figsize=(3.2*len(examples),3.7),layout='constrained',squeeze=False)
        for ax,(f,points) in zip(axes.flat,examples,strict=True):
            for j,label in ((2,'Selected worse-peer best'),(3,'Example better-peer best')):
                ax.plot(points[[1,j],0],points[[1,j],1],ls='--',color=COLORS['Worse' if j==2 else 'Better'])
                ax.scatter(*points[j],s=42,color=COLORS['Worse' if j==2 else 'Better'],label=label)
            ax.scatter(*points[1],marker='*',s=85,color='black',label='Global best')
            ax.scatter(*points[0],marker='x',color='#596270',label='Focal particle')
            ax.set_aspect('equal'); ax.set_title(names[f]); ax.grid(alpha=.2)
        handles, labels_ = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels_,loc='outside lower center',ncols=2,fontsize=8)
        fig.suptitle('Peer personal bests before update 5 · seed 0')
        fig.savefig(ROOT/'peer_geometry_example.pdf')
        plt.close(fig)


if __name__ == '__main__':
    main()
