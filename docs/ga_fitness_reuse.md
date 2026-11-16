# GA-MPC fitness reuse and evaluation-only replay

GA-MPC now carries fitness values alongside the population and archive. Parent
ranking reuses the selected parents' known scores. New offspring are evaluated;
unchanged survivors and archive members reuse scores. Duplicate perturbation or
clipping invalidates scores only for coordinates that actually changed. Exact
repeated points within a generation reuse a bounded, generation-local cache.
The evaluator assumes deterministic objectives, as used by this CEC2011 study.

Best-solution selection and return use stored values without additional calls.
The returned coordinates match the best score seen anywhere in the evaluated
search, including candidates later discarded. This corrects the prior mismatch
between historical best tracking and returning the final population's best.

Regression tests compare the previous implementation with the revised one for
four seeds: identical sets of visited points and matching RNG state, with fewer
objective calls. Tests also cover no-crossover/duplicate cases, best-score/solution
consistency and exact evaluation stopping with no generation limit. The original
implementation used for regression is artifacts/ga_mpc_before_fitness_reuse.py.

The CEC2011 replay uses population 200 and seeds 0–2 for all 22 problems.
GA-MPC stops solely at exactly 200,000 actual objective calls: max_gen=None.
It can exceed 1000 generations. No objective evaluation is charged for a cache
lookup. The 1000-generation-limited launch was stopped and retained separately;
it is not mixed with this evaluation-only replay.

The other eight algorithms' 528 records are copied unchanged from the completed
stabilized-CMA comparison. T3 bounds remain [0.6,0.9]. Each problem/algorithm's
source fingerprint is preserved or updated explicitly in the new plan. All 66
GA-MPC runs are replayed, because fitness reuse changes how many generations fit
inside the evaluation budget. Updated curves and tables are generated afterward.

```powershell
$env:PYTHONPATH='D:\hppso-git\.deps\cec2011;D:\hppso-git\src'
python -u -m experiments.cec2011_ga_fitness_reuse --workers 2
```

Live progress: http://127.0.0.1:8771/.
Output: experiments/results/cec2011_pop200_ga_reuse_budget200k.
