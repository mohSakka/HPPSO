# GA-MPC evaluation cap

The updated comparison preserves the population-200, 1000-iteration results of
the other eight algorithms and reruns only GA-MPC for seeds 0–2 on all 22 problems.
GA-MPC has a hard maximum of 200,000 objective calls, including evaluations inside
parent sorting, selection and best-value tracking. Native iteration maximum
remains 1000. Final counter checks require exactly 200,000 calls in each replay.
The capped final score must match the recorded 200,000-evaluation checkpoint of
the original deterministic run. Saved solution coordinates come from the replay.

```powershell
$env:PYTHONPATH='D:\hppso-git\.deps\cec2011;D:\hppso-git\src'
python -u -m experiments.cec2011_cap_ga --workers 2 --port 8768
```

Output: experiments/results/cec2011_pop200_ga_cap200k. Live progress and new curves:
http://127.0.0.1:8768/. Completed trials are resumable. Original results remain
unchanged; the comparison plan records the fingerprint for each algorithm.
Ranks are regenerated after all 66 GA-MPC replays finish. Existing CMA-ES failures
on T8 and T13 remain explicit and those problem blocks stay excluded from overall
nine-algorithm rankings.

This change applies the new cap to GA-MPC. The other algorithms retain their
original stopping rules and actual evaluation counts (typically 200,200 for PSO
variants and 200,000 for CMA-ES). It is not a new equal-budget rerun of all nine.
