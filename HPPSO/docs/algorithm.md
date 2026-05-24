# HPPSO Algorithm Description

Human Personality Based Particle Swarm Optimization (HPPSO) extends classical PSO by assigning each particle a **personality vector** modeled on human traits that continuously modulate search behavior.

## Personality Traits

| Trait | Symbol | Effect |
|-------|--------|--------|
| Curiosity | `curio` | Controls Gaussian exploration mutation probability |
| Confidence | `conf` | Balances cognitive (personal best) vs. social (global best) attraction |
| Aggressiveness | `agg` | Modulates inertia weight for velocity updates |
| Sociality/Openness | `soc` | Enables learning from better peers when above threshold (also called openness) |

## Velocity Update

For each particle `i` at iteration `t`:

```
V_i = inertia(agg_i) * V_i
    + (1 - conf_i) * c1 * r1 * (P_i - X_i)   [cognitive]
    + conf_i * c2 * r2 * (g_best - X_i)       [social]
```

When `soc_i > socialism_threshold` (sociality/openness above threshold), an additional socialism term is applied:

```
V_i += λ * (P_j - X_i)
```

where `j` is a randomly selected particle with better fitness than `i`.

## Curiosity Mutation

With probability `0.05 * curio_i` per dimension, Gaussian noise scaled by `η * range` is added to the position.

## Stagnation Recovery

If a particle's personal best does not improve for `stagnation_threshold` consecutive iterations, its personality vector is reinitialized uniformly in `[0, 1]`.

## Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_pop` | 30 | Population size |
| `c1`, `c2` | 1.5 | Cognitive and social coefficients |
| `eta` | 0.01 | Mutation scale factor |
| `lam` | 0.4 | Socialism learning rate |
| `socialism_threshold` | 0.5 | Minimum sociality/openness for peer learning |
| `stagnation_threshold` | 10 | Iterations before personality reset |

## Relationship to Baselines

HPPSO generalizes several PSO variants:

- **Standard PSO**: fixed inertia, no personality, no mutation
- **PSO-m**: Gaussian mutation (similar to curiosity component)
- **PSO-RIW**: random inertia weight (similar to aggressiveness modulation)

## Benchmark Competitors

This repository compares HPPSO against:

- PSO, PSO-m, PSO-RIW
- GA-MPC (Genetic Algorithm with Multi-parent Crossover)
- GWO (Grey Wolf Optimizer)
- SHADE (Success-History based Adaptive Differential Evolution)
- CMA-ES and Sep-CMA-ES
- CSA (Circle Search Algorithm)

## References

See the manuscript in [`docs/papers/HPPSO_Manuscript.pdf`](papers/HPPSO_Manuscript.pdf) for the full theoretical treatment and experimental results.
