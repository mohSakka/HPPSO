# CEC2011: population 200 and 1000 iterations

This separate experiment retains all 22 native CEC2011 problems, the nine
corrected algorithms (excluding Sep-CMA-ES), and seeds 0–2. Every optimizer uses
population 200 and its native 1000-iteration loop. Initialization evaluations
are additional. Stopping is by iterations, not a common evaluation budget.
Actual evaluation counts and times are saved; GA-MPC can use additional objective
calls inside an iteration. Evaluation-based parameter schedules are disabled so
each optimizer uses its native iteration-based schedule. CMA-ES target-fitness
termination is disabled.

```powershell
$env:PYTHONPATH='D:\hppso-git\.deps\cec2011;D:\hppso-git\src'
python -u -m experiments.cec2011_iterations --workers 2 --port 8767
```

Results and resumable raw trials: experiments/results/cec2011_pop200_iter1000.
Live progress and per-problem convergence plots: http://127.0.0.1:8767/.
Plots use actual objective evaluations on their x-axis. Three-seed min–max bands
are descriptive ranges. Each problem exports its own runs_Tn.csv and
summary_Tn.csv. Earlier evaluation-budgeted experiments remain separate.
