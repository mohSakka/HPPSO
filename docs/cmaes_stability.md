# CMA-ES numerical stabilization and replay

The earlier CEC2011 population-200 run failed in covariance eigendecomposition
on all three T8 seeds and two T13 seeds. Reproducing T8 seed 0 with floating-point
exceptions enabled identified an invalid square root of a computed eigenvalue
at evaluation 112,800, followed by the reported decomposition failure at 113,000.

The implementation now symmetrizes covariance, floors eigenvalues at the larger
of lambda_max/1e14 and the smallest normal positive float, and reconstructs the
covariance and its square-root/inverse factors consistently when flooring occurs.
The condition-number safeguard intentionally regularizes near-singular states;
it is a documented algorithm change, not a mathematically equivalent rewrite.
Step size is updated in log space and limited to log(box scale)+[-32,14].
Non-finite covariance or log-step updates still raise explicit errors rather
than producing fabricated finite objective results. Objective formulas and
candidate eligibility rules are unchanged.

Tests cover an almost-singular covariance with a negative roundoff eigenvalue,
consistency of repaired matrix factors, explicit non-finite-state failure, and
finite multi-generation Sphere optimization. The CEC2011 replay runs with
NumPy invalid/division/overflow exceptions enabled. Raw trial records save
covariance repair counts, step-size clip counts and final covariance finiteness.

All 66 CMA-ES trials are rerun to avoid selecting the new version only for
previously failed cases. The other 528 trials are preserved byte-for-byte.
Population is 200, seeds are 0/1/2, native iteration maximum is 1000, and CMA-ES
fitness-target termination remains disabled. The corrected T3 interval [0.6,0.9]
and GA-MPC's 200,000-evaluation cap are retained. Table generation identifies
source versions per problem and algorithm. Complete problem blocks enter ranks;
failures, if any, remain explicit.

```powershell
$env:PYTHONPATH='D:\hppso-git\.deps\cec2011;D:\hppso-git\src'
python -u -m experiments.cec2011_stable_cma --workers 2
```

Output: experiments/results/cec2011_pop200_stable_cma.
Live progress, refreshed curves and final tables: http://127.0.0.1:8770/.
Each completed trial is resumable; T8/T13 are scheduled first. Previous results
are retained in their original folders.
