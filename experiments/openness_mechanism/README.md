# Openness during global-best stagnation

The experiment separates two questions. Do worse-current-fitness peers carry more distinct personal bests? Does attracting a particle toward one of those personal bests improve short-term recovery when the number of interactions is held fixed?

## Results

The benchmark set was specified before the experiment. All 40 full HPPSO replays (four functions, seeds 0–9) matched the saved convergence checkpoints and final best positions exactly. Twenty function–seed runs contributed at least one eligible, nonoverlapping snapshot. The original run stayed on its original trajectory after each snapshot.

| Function | Recovery seeds / 10 | Snapshots | Mean normalized improvement per arm | Median normalized improvement per arm | Recovery per arm (seed mean) |
|---|---:|---:|---:|---:|---:|
| Shifted Lévy | 4 | 14 | 1.08e-7 | 1.10e-7 | 0% |
| Rotated Schaffer F7 | 4 | 9 | 0.0360 | 0.0333 | 72.5% |
| Shifted Ackley | 9 | 40 | 4.00e-6 | 8.33e-7 | 0% |
| Rotated Griewank | 3 | 9 | 0.00252 | 0.000623 | 50% |

**Historical memory.** During stagnation, worse peers had farther personal bests on average than better and random peers on each function. The worse-minus-better differences in normalized distance were about 6.9e-6 (Lévy), 0.01145 (Schaffer F7), 9.1e-6 (Ackley), and 1.1e-5 (Griewank). Both planned distance contrasts passed Holm correction on Ackley (adjusted p = 0.03125 each), where the absolute differences were small. The difference was much larger on Schaffer F7, where only four seeds contributed eligible stagnation states and both adjusted p-values were 0.50. Worse peers also retained larger current-minus-personal-best objective gaps on average on all four functions. [Memory-distance figure](results/01_memory_distance.pdf) · [Historical comparisons](results/historical_comparisons.csv)

**Direction.** The worse-peer direction had lower mean cosine alignment with the global-best direction than the better-peer direction on Lévy, Schaffer F7, and Ackley, and higher alignment on Griewank. Against the random-peer direction it was less aligned on Lévy and Schaffer F7 and more aligned on Ackley and Griewank. No alignment contrast passed Holm correction. [Supporting comparisons](results/supporting_comparisons.csv)

**Recovery.** At every snapshot the four branches reached the same final global-best objective and had the same first strict-improvement time or censoring status. In each arm, 44 of 72 snapshots registered a strict improvement and 8 reached the recovery threshold. The latter is a pooled snapshot count. The figure and table give equal weight to the contributing seeds within each function. The worse-minus-better, worse-minus-random, and worse-minus-no-attraction differences in normalized improvement were all zero, with exact and adjusted p = 1. The bootstrap intervals are [0, 0] because all observed paired differences were zero. These intervals do not establish equivalence for unsampled states. [Paired recovery figure](results/02_stagnation_recovery.pdf) · [Recovery probability figure](results/03_recovery_probability.pdf) · [Primary comparisons](results/primary_comparisons.csv) · [Branch summaries](results/branch_summary.csv)

**Interpretation.** The observations partially support the proposed motivation. Worse peers provided more spatially distinct historical memories in these stagnation states, with limited inferential support across functions. Directional distinctiveness varied by benchmark. A single forced interaction produced no measurable improvement in global-best recovery within ten updates compared with better peers, random peers, or no attraction. This conclusion is specific to the sampled states and intervention horizon. Longer horizons and repeated interactions were outside the design.

## Design and accounting

- All replays used 100 dimensions, 50 particles, 60,000 evaluations including initialization, the archived bounds and transformations, and the full HPPSO settings. Global stagnation means ten consecutive updates without a strict global-best improvement. Personality refreshing uses each particle's own stagnation counter and remains active in all branches.
- The original rule draws one peer uniformly from the other particles and applies a personal-best attraction only when the focal trait exceeds 0.5 and the peer has a worse current objective. The coefficient is 0.4. The batched engine draws peer indices for all particles and masks ineligible ones to preserve fixed random streams. The standalone HPPSO class draws indices only for eligible particles. Both use one draw per eligible particle with no retries.
- Experiment A uses the same open focal particles for all three rules. Each focal must have at least one better and one worse candidate. The distance and retained improvement are averaged exactly over the relevant candidate set, then over focals, then over states within each function, seed, and stagnation category. This estimates the conditional memory available from each rule, without conflating it with acceptance frequency.
- Experiment B takes the first eligible states with disjoint ten-update continuation windows, up to five per run, and draws the focal uniformly from its eligible set. Worse and better branches each draw uniformly from their relevant fitness set, and the random branch draws uniformly from all other particles. All three receive one forced attraction in the first update. The control receives zero. Further openness is disabled in every branch. Cognitive, social, curiosity, and personality-refresh random draws are matched by fixed-size streams restored from the snapshot. Each branch applies its own state-dependent masks. The snapshot-selection generator is separate from HPPSO's generators.
- The primary statistic is the mean paired difference of seed means for normalized improvement at update 10. The exact two-sided test enumerates all within-seed sign flips under the paired exchangeability null. Holm correction covers the 12 primary contrasts (three comparators × four functions). Historical comparisons use a separate Holm family for each outcome and stagnation category. Supporting outcomes each use their own Holm family. Intervals are pointwise percentile intervals from 10,000 resamples of seed blocks. Median selected distance is the median of within-seed medians. `time` records first strict improvement or the horizon for a right-censored branch. `event` distinguishes these cases. `restricted_time` is the seed mean of that capped time.
- The study consumed **2,544,000 new objective evaluations**. The 40 complete replays used 2,400,000, and the 72 snapshots × 4 branches × 10 updates × 50 particles used 144,000. Diagnostics and snapshot restoration use stored fitness values. Every intervention changed the first-step focal velocity after clipping in the archived snapshots.

## Reproduce

From the repository root in PowerShell, using the same Python environment as the full factorial ablation:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m experiments.openness_mechanism.run --config experiments/openness_mechanism/config.json --output experiments/openness_mechanism/results --workers 3
python -m experiments.openness_mechanism.analyze experiments/openness_mechanism/results
```

The replay command reuses complete batches in this output directory. For a clean rerun, pass a new output path to both commands. The reference is the archived `08_full_factorial` study, specified in `config.json`. The output manifest records its fingerprint and settings. [Coverage and evaluation counts](results/coverage.csv) · [Historical state measurements](results/historical_states.csv) · [Raw snapshot measurements](results/branch_snapshots.csv) · [Historical seed aggregates](results/historical_by_seed.csv) · [Recovery seed aggregates](results/recovery_by_seed.csv). The CSV rule `None` denotes the no-attraction branch. Complete swarm snapshots are stored beside the compressed trial files under `results/trials`.
