# Additional hard benchmark functions

The active unified suite is `classical20-additional-v3`: the original 20 classical
functions plus the following ten problems. The former extra Sphere/Rosenbrock
cases and every `scaling_*` function ID have been removed from the active code.
The IDs below are local registry IDs, not official CEC function numbers.

| ID | Function | Base domain per coordinate | Main property |
|---|---|---|---|
| f21_expanded_schaffer_f6 | Expanded Schaffer F6 | [-100,100] | Cyclic pair interactions, non-separable, multimodal |
| f22_rotated_ellipsoid | Rotated ellipsoid | [-100,100] | Non-separable, condition number 10^6 |
| f23_rotated_rastrigin | Rotated Rastrigin | [-5.12,5.12] | Non-separable, multimodal |
| f24_rotated_ackley | Rotated Ackley | [-32.768,32.768] | Coupled coordinates, multimodal |
| f25_composition | Three-center composition | [-5,5] | Spatially varying mixture of rotated landscapes |

No duplicate Sphere or Rosenbrock problems are added. There is no global
normalization, clipping, or rescaling of the ten objective values. Composition
weights and component biases are part of its explicit mathematical definition.

| ID | Function | Base domain per coordinate | Main property |
|---|---|---|---|
| f26_rotated_griewank | Rotated Griewank | [-600,600] | Rotated product coupling, multimodal |
| f27_rotated_discus | Rotated discus | [-100,100] | One stiff direction, non-separable |
| f28_schwefel_1_2 | Schwefel 1.2 | [-100,100] | Cumulative interactions, non-separable |
| f29_rotated_schaffer_f7 | Rotated Schaffer F7 | [-100,100] | Oscillatory pair coupling |
| f30_composition_multimodal | Multimodal three-center composition | [-5,5] | Different components and attraction widths |

## Shifts and rotations

Set z=x-o. The domain translates with the shift: x in [L+o,U+o], so z always
lies in [L,U]. Independently sampled coordinates of o place a known minimizer
at nonzero, varied coordinates. The optimum value remains zero; this is distinct
from the coordinates of the solution. All additions are nonnegative, with a
feasible global optimum at x=o. The composition also has two additional centers
inside this translated box, with positive component biases.

For a rotated component, define

    Q_k(v) = DCT-II(s_k * v[p_k]), norm='ortho'.

A permutation, sign changes and the orthonormal DCT preserve Euclidean norm.
They mix the variables with O(D log D) evaluation cost. This is a structured
orthogonal transformation, not a Haar-uniform random orthogonal matrix.
Instances are reproducible from the complete saved permutations and signs;
there is no need to regenerate dense matrices or infer them from a seed.
These local definitions are not claimed to be official CEC/BBOB instances.

## Exact objectives

For v of dimension D, let

    E(v) = sum_{i=0}^{D-1} (10^6)^(i/(D-1)) * v_i^2
    R(v) = sum_i [v_i^2 + 20*sin(pi*v_i)^2]
    A(v) = -20*expm1(-0.2*sqrt(mean(v_i^2)))
           - e*expm1(-2*mean(sin(pi*v_i)^2)).

R and A are algebraically equivalent to the usual Rastrigin and Ackley formulas.
The expm1/sine forms avoid subtractive cancellation at their zero optimum.

* f21: set r_i^2 = z_i^2 + z_{(i+1) mod D}^2 and sum
  0.5 + [sin(sqrt(r_i^2))^2 - 0.5]/(1+0.001*r_i^2)^2.
  The implementation uses an algebraically equivalent nonnegative numerator to
  avoid cancellation. The final-to-first pair is included.
* f22: E(Q_0(z)).
* f23: R(Q_0(z)).
* f24: A(Q_0(z)).
* f25: local centers c_0=0 and independently sampled c_1,c_2 in [-4,4]^D.
  Set d_i=||z-c_i||^2 and biases b=(0,100,200). For d_i>0,

      w_i = d_i^(-1/2)*exp(-d_i/(2D));  W_i=w_i/sum_j w_j
      F(x) = W_0*E(Q_0(z-c_0))
           + W_1*[R(Q_1(z-c_1))+100]
           + W_2*[A(Q_2(z-c_2))+200].

  If any d_i is zero, its component has weight one, so F(o+c_i)=b_i.
  Log-space normalization evaluates the weights without all-weight underflow.
  Each component has its own rotation. This is a custom, fully specified
  distance-weighted composition, not the retired single-center additive hybrid.
  Nonnegative components and biases establish F>=0, attained at x=o.

## Registry and saved definitions

Use `build_benchmark_suite(...)` for all 30 problems or
`build_additional_suite(...)` for these ten. The original
`build_problem_suite(...)` API continues to provide only the classical 20.
The two classical experiment entry points and the all-function runner now use
the unified suite. Dimension-trend experiments select the five new IDs by name.

`problem_definitions.json` records each added instance's shift, base/transformed
bounds, optimum, permutations, signs, centers, biases, weight convention, and
version. Explicit fixed shifts can be supplied without changing the generated
rotation for the same seed. The all-function resume plan includes the suite
version and exact function IDs: old result directories cannot be silently
resumed as this different suite. Use a fresh output directory for a full rerun.

Historical raw results, plots and tables are preserved under their original
names. Do not relabel old `scaling_*` scores as measurements of these new
functions: their domains, definitions and/or instances differ. They must be
excluded when summarizing the revised suite until replacement runs exist.
The prior 1000D full experiment remains stopped; validation smoke runs are not
new full-budget experiments.

## Definition references

Rotated ellipsoid, Rastrigin and Ackley definitions can be compared with the
primary CEC2010 large-scale benchmark report:
https://titan.csit.rmit.edu.au/~e46507/publications/lsgo-cec10.pdf .
Expanded Schaffer F6 and composition-function constructions are described in the
authors' CEC2014 definitions and source archive:
https://github.com/P-N-Suganthan/CEC2014 . Our rotation/instance choices and
three-center composition are explicitly local definitions above.

## Definitions added in additional-v3

For z=x-o and the saved orthogonal Q:

* f26: G(Qz), where G(v)=sum(v_i^2)/4000 + 1 - product(cos(v_i/sqrt(i))),
  i=1,...,D.
* f27: 10^6*(Qz)_1^2 + sum_{i=2}^D (Qz)_i^2.
* f28: sum_{i=1}^D (sum_{j=1}^i z_j)^2. No rotation is applied; cumulative
  sums intrinsically couple the variables.
* f29: let v=Qz and r_i=sqrt(v_i^2+v_{i+1}^2), i=1,...,D-1;
  F=[mean_i sqrt(r_i)*(1+sin(50*r_i^0.2)^2)]^2. This uses the basic Schaffer
  F7 formula with the saved rotation, without BBOB's additional transformations.
* f30 uses f25's three-center construction, biases (0,100,200), but components
  Rastrigin, Griewank and Expanded Schaffer F6. Its widths are sigma=(1,2,3):
  log(w_i)=-0.5*log(d_i)-d_i/(2*D*sigma_i^2), normalized with log-sum-exp.
  At an exact center only that component has weight one. Its minimum is zero
  at x=o. Components, widths, centers and rotations are included in the manifest.

All ten additional functions have known feasible zero-valued optima at the
nonzero coordinate shift, and have cross-coordinate interactions. The first
five instances retain their formulas and seeded instance generation; the suite
version changes to prevent accidental reuse of an older, smaller run plan.
Existing historical numerical results are not silently assigned to new cases.

Basic-function references:
https://www.sfu.ca/~ssurjano/griewank.html and
https://coco-platform.org/testsuites/bbob/functions/f17.html . These local rotated
instances and compositions are not presented as official BBOB/CEC implementations.
