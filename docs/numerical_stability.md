# Schwefel 2.22 numerical evaluation

For shifted coordinates z = x - o, the original objective is

    f(x) = S + P,  S = sum_i |z_i|,  P = product_i |z_i|.

The evaluator now returns g(x) = ln(1 + f(x)), computed as

    L_S = logsumexp(log(|z_i|))
    L_P = sum_i log(|z_i|), or -infinity if any coordinate is zero
    g = logaddexp(logaddexp(0, L_S), L_P).

NumPy logaddexp combines log-space terms without constructing the enormous
product or exponentiating it. There is no epsilon added to coordinates and no
exponent cap. At the optimum g = 0 exactly. For 1000 coordinates equal to 10,
g is approximately 2302.585, although raw f exceeds float64 capacity.

This is an equivalent minimization problem in exact arithmetic: ln(1 + t) is
strictly increasing for t >= 0, so minimizers and pointwise comparison ordering
are preserved. It is not equality of numerical scores. Floating-point rounding
can still produce ties for sufficiently close values. Algorithms using fitness
magnitudes or differences rather than ordering are not guaranteed identical
trajectories. Absolute stopping tolerances must be interpreted on the new scale.

The prior implementation clipped the log-product to [-700, 700], which produced
artificial plateaus and changed the objective; adding epsilon also changed zero
products. Its finite results are not evidence that original raw values fit in
float64. The historical manuscript infinity outcomes and the clipped repository
implementation must be distinguished explicitly.

## Reporting and statistical analysis

Rerun all algorithms on Schwefel 2.22 with this same evaluator, at both dimensions.
Label its means, standard deviations, and convergence curves as ln(1 + f).
Do not mix old raw/clipped results with new transformed results. Transforming
stored infinity cannot recover lost information. Inverting transformed means
with expm1 does not yield raw arithmetic means. Means, standard deviations,
rankings of means, and Wilcoxon signed-rank results need recalculation; monotone
transformations do not generally preserve them.

For legacy reproduction data, a function is excluded from average ranks and
pooled Wilcoxon comparisons whenever any algorithm has empty, missing, NaN, or
infinite run results. The long performance CSV records n_invalid and
analysis_included, and excluded ranks are blank. A finite rerun of Schwefel is
eligible for inclusion; it is no longer excluded by name or dimension.
No missing tests are counted as wins or ordinary ties.

The existing reproduction Wilcoxon procedure pools run scores across functions;
this patch retains that design. Pooling different objective scales needs a
separate methodological justification, especially after transforming one
benchmark. Consider per-function paired analyses with an explicitly justified
multiple-comparison procedure for the revised manuscript.

These code changes do not update published tables or establish revised numerical
findings. Regenerate Tables 3-7 and report included functions and exclusions
before making a response-to-reviewer claim about corrected results.

Reference: https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
