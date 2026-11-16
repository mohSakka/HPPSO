# PSO and HPPSO computational cost

## Complexity

For a swarm of $n$ particles with $d$ coordinates, let $m$ be the number whose openness exceeds the threshold and $k$ the number whose sampled peer is worse. For $n>1$, an open particle with zero-based index $i$ draws $u$ uniformly from the integers $0,\ldots,n-2$ and uses $j=u+\mathbf{1}[u\ge i]$. This maps each draw to one of the other particles uniformly. HPPSO compares the two cached current objective values. For a minimization problem, the velocity update applies if $f(x_j)>f(x_i)$, and the direction uses the peer's personal best $p_j-x_i$. When the comparison fails, the particle skips this update. Each open particle makes one draw and one comparison.

Checking openness, drawing peers, and testing their fitness takes $O(n+m)$ time in total. Applying the successful velocity updates takes $O(kd)$. This gives $O(n+m+kd)$ for the entire openness step. Peer selection uses $O(n)$ indices and comparisons and makes zero extra objective evaluations. The vectorized position and velocity updates, boundary handling, and curiosity mutation take $O(nd)$ time. Stagnation tracking and personality refresh add at most $O(n)$.

Both implementations therefore spend $\Theta(nd)$ per full swarm update on optimizer operations and make $n$ objective evaluations. If $C_f(x)$ is the cost of evaluating a particle, the full iteration takes

$$
\Theta(nd) + \sum_{i=1}^{n} C_f(x_i).
$$

With a common cost $C_f$, this is $\Theta(nd+nC_f)$. In the dense MLP implementation, a forward pass and loss evaluation over $B$ training examples costs $\Theta(Bd)$ for $d$ trainable weights and biases. The resulting iteration cost is $\Theta(nd+nBd)$.

Both algorithms use $\Theta(nd)$ working space for the swarm, plus the objective's data and workspace and any stored convergence history. HPPSO adds $O(n)$ trait values and counters and can allocate $O(nd)$ temporary arrays for peer updates and curiosity. These arrays can change the constant factors in elapsed time and peak memory.

## Archived server timings

The source is `hppso_final_results/hppso_final_results/final_results/runs.csv`. Recomputed mean times agree with `summary.csv` for all 240 PSO and HPPSO rows to within $6\times10^{-14}$ s. Only successful trials consuming their full evaluation budget were eligible. The analysis paired Original PSO and HPPSO by dimension, function, seed, population, budget, and benchmark fingerprint. Eight pairs ran on different hosts and were excluded, leaving 2,392 pairs across 120 problem–dimension combinations.

For each combination, a two-sided paired $t$-test used HPPSO time minus PSO time. Holm correction covered all 120 tests. The relative runtime column first takes the median of the paired HPPSO-to-PSO time ratios for each problem, then the median across the 30 problems at each dimension.

| Dimension | Population | Evaluations | Median runtime ratio | Significant HPPSO slowdowns |
|---:|---:|---:|---:|---:|
| 30 | 30 | 30,000 | 1.239 | 7/30 |
| 300 | 90 | 300,000 | 1.240 | 17/30 |
| 600 | 180 | 600,000 | 1.168 | 19/30 |
| 1,000 | 300 | 1,000,000 | 1.263 | 22/30 |

The elapsed times were collected under parallel server scheduling, so shared load can influence the paired differences and the tests should be treated as exploratory. The archive records a single benchmark fingerprint, which differs from the current checkout's fingerprint. The archived timings describe that run. The complexity analysis and Digits measurements describe the current implementation.

## Matched Digits experiment

The experiment uses the existing preprocessed Digits training split of 1,078 examples and the $64\to42\to28\to10$ MLP with 4,224 trainable parameters. Both methods use population 30, bounds $[-2,2]$, and 3,000 full-batch cross-entropy evaluations, including the initial population. For each of the 30 seeds, the methods start with identical particle positions and zero velocities. HPPSO uses the implementation defaults. Each trial runs in a fresh process, with one trial at a time and one thread per numerical library. PSO runs first for 15 seeds and HPPSO runs first for the other 15. The machine ran Windows 10 on an Intel Core i3-6006U at 2.00 GHz.

Wall time covers the training call, including initialization and evaluation accounting. Data loading, preprocessing, and a shared warm-up forward pass occur before timing. Memory is the Windows peak working set reported through `psutil`, which covers resident pages of the entire trial process, including the shared Python runtime, libraries and model. The raw measurements record the working set immediately before training and its previous peak so that startup memory can be checked. CPU time and the rise above the pretraining working set are supplementary measurements.

The two primary outcomes are elapsed training time and peak working set. The analysis uses the mean paired difference (HPPSO minus PSO), its marginal 95% confidence interval, and a paired two-sided $t$-test with Holm correction across the two primary outcomes.

| Metric | PSO mean ± SD | HPPSO mean ± SD | Paired difference [95% CI] | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Wall time (s) | 9.61 ± 1.53 | 11.33 ± 1.89 | +1.73 [0.87, 2.59] | 0.000303 |
| Peak process working set (MiB) | 176.62 ± 0.36 | 174.43 ± 0.34 | −2.20 [−2.37, −2.02] | 2.31 × 10⁻²¹ |

All 60 trials reached the evaluation budget. The peak rose during training in every trial. HPPSO also had a smaller mean peak rise over the pretraining working set (10.04 MiB versus 12.18 MiB), and higher mean CPU time (10.58 s versus 8.95 s). These measurements describe the total process and its optimizer implementation for this workload. Memory differences can reflect the size and lifetimes of temporary arrays, allocator behavior, and process residency.

The [raw trials](../experiments/results/pso_hppso_resources/nn_trials.csv), [environment](../experiments/results/pso_hppso_resources/nn_environment.json), [paired analysis](../experiments/results/pso_hppso_resources/nn_paired_resources.csv), and [paired plot](../experiments/results/pso_hppso_resources/nn_paired_resources.png) are saved alongside the [per-problem server tests](../experiments/results/pso_hppso_resources/server_paired_runtime_by_problem.csv).

## Reproduction

From the repository root on Windows, with the project dependencies and `psutil` installed, choose a fresh output directory:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -u -m experiments.pso_nn_resources --output experiments/results/pso_hppso_resources_repeat
python -m experiments.analyze_pso_resources --output experiments/results/pso_hppso_resources_repeat --nn --server-runs hppso_final_results/hppso_final_results/final_results/runs.csv
```
