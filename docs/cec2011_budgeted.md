# Corrected CEC2011 experiments

Run all 22 fixed-dimension native CEC2011 problems with Original PSO, PSO-m,
PSO-RIW, HPPSO, GA-MPC, GWO, SHADE, CSA and full CMA-ES. Sep-CMA-ES is excluded.
Each algorithm uses population 300, seeds 0/1/2, and exactly 1000 times the native
problem dimension in objective evaluations, including initialization and every
additional objective call (including GA parent comparisons). The final generation
can be partial. An evaluation counter stops all algorithms at the exact budget;
an iteration guard cannot be reached first. CMA-ES fitness-target termination is
disabled to support problems with negative objective values.

PSO variants use corrected zero initial velocities, c1=c2=2, common velocity limits
and reflection. HPPSO uses eta=0.001 and the corrected single-peer openness rule.
CEC2011 objectives are neither shifted nor transformed. Bounds and dimensions
come from MinionPy 1.9.1 native metadata. In particular, this runner does not apply
the old repository loader's T3 [0.6,0.9] override: native T3 bounds are [-0.6,0.9].
The package's implementation defines this experiment; these are not classical
or custom composition functions from the earlier studies. No optimum is assumed
when plotting: curves show raw best objective values.

## Run or resume

```bash
python -m pip install -e . minionpy==1.9.1
python -u -m experiments.cec2011_budgeted --workers 2 --port 8766
```

On this Windows workspace, MinionPy is installed locally in .deps/cec2011:

```powershell
$env:PYTHONPATH='D:\hppso-git\.deps\cec2011;D:\hppso-git\src'
python -u -m experiments.cec2011_budgeted --workers 2 --port 8766
```

Open http://127.0.0.1:8766/ for live progress. Active evaluations and best scores
refresh every few seconds. Raw results are saved atomically after every trial.
After each problem's 27 trials finish, PNG/PDF comparison curves and summary
tables are updated. Curves show the median with min–max bands over three seeds;
these bands are not confidence intervals. X-axes are logarithmic evaluation
counts. Positive objectives use logarithmic y-axes; otherwise y is linear.
Only algorithms with all three valid seed outcomes enter a comparison curve.
Non-finite candidate evaluations count against the budget and return +infinity
as an ineligible candidate; no arbitrary finite penalty is used. Their counts
appear in active progress and raw results. A run without a finite best result
fails. Failed final outcomes remain explicit and are excluded from comparisons.

Results are under experiments/results/cec2011_corrected_3seeds_live. plan.json records
all bounds, dimensions, budgets, evaluator/package version and source fingerprint.
runs.csv contains per-trial objective values, evaluations, execution time and
failure status. summary.csv contains per-problem/algorithm descriptive statistics.
trials stores gzip records with best solutions and sampled objective traces.
Run the same command to resume completed trials. A source/config change requires
a fresh output directory. After a shutdown, verify that no old worker remains
before removing the stale .shard-cec2011.lock file.

If Windows denies overwriting a CSV (for example while it is open in another
application), resume the unchanged scientific run with:

```bash
python -u -m experiments.resume_cec2011_live --workers 2 --port 8766
```

This writes each updated table under a unique filename in exports. The live
page reads csv_latest.json to link to the newest tables. Reload the page once
after switching to this recovery command. Raw trial fingerprints and optimizer
source remain unchanged; recovery_metadata.json records the export-only change.

The HTTP page remains available after all trials finish, until the process is
stopped. It is bound to localhost only. Three seeds are preliminary evidence;
do not treat min–max bands as robust estimates of stochastic uncertainty.

The evaluator is documented at https://minion-py.readthedocs.io/en/latest/minionpy/pyapi.html.
