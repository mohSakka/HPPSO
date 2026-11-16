# Corrected PSO initialization and reruns

Every PSO variant now starts with zero velocity in every coordinate. This includes
PSO, PSO-m, PSO-RIW, HPPSO, the isolated trait variants and the openness study.
The previous results used different PSO/HPPSO velocity initialization; retain them
as historical results, not as the corrected comparison. Fresh output directories
and source fingerprints prevent mixing them with corrected runs.

## Full trait study

From the repository root, install and run:

```bash
python -m pip install -e .
python -u -m experiments.run_trait_zero_velocity --workers 2
```

This runs all 13 previously selected functions at 100D, population 50, seeds 0–9,
100,000 evaluations per path: 130 batches and 7,670 path runs. All previous trait
comparisons, one-at-a-time parameter sweeps and interactions remain. HP-backbone
is removed from the reported ablation comparison, leaving HPPSO, four leave-one-
trait-out variants and HPPSO without personality reinitialization. The empty
factorial cell remains an internal control required for interaction contrasts;
its label is `HPPSO all traits off (factorial control)`.

Results, progress, audit and plots are in
`experiments/results/trait_study_zero_velocity13`; the completed archive is
`artifacts/trait_study_zero_velocity13_results.zip`. The command resumes completed
corrected trials. Batch timing describes combined paths, not separate algorithm
execution times.

## Server final benchmark: only PSO, PSO-m, PSO-RIW

Extract the code bundle, enter `hppso-server`, then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -u -m experiments.final_benchmark --config experiments/configs/final_benchmark_pso_zero.json --output experiments/results/final_pso_zero --workers 48
python -m experiments.final_benchmark --output experiments/results/final_pso_zero --report
```

The configuration selects exactly these three algorithms, all 30 final benchmark
functions, dimensions 30/300/600/1000 and 20 seeds (0–19): 7,200 trials. Populations
are respectively 30/90/180/300; exact evaluation budgets are 30,000/300,000/600,000/
1,000,000, including initialization. Evaluation count is the sole stopping rule.
Velocity limits, reflection and c1=c2=2 are preserved. All three algorithms now
initialize velocity to zero. No other algorithm is rerun by this configuration.

The runner limits numerical library threads to one per worker. Choose workers
according to allocated CPU cores; use 24 on one 24-core EPYC 7443, or 48 when 48
physical cores are allocated. Results include per-trial wall time, raw best
solutions, sampled convergence traces, summary statistics and convergence PDFs.
See progress JSON files and terminal output for live completion counts. Rerun
the same command to resume; use a new output directory for changed settings.

Do not merge historical random-velocity results with this corrected dataset.
The source/config fingerprints are saved in the output plan and each trial.
