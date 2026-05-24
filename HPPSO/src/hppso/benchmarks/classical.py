"""Classical benchmark functions with optional coordinate shifting."""

from __future__ import annotations

import numpy as np

FUNCTION_NAMES = [
    "f1_sphere",
    "f2_schwefel_2_22",
    "f3_zakharov",
    "f4_rosenbrock",
    "f5_powell_sum",
    "f6_rastrigin",
    "f7_ackley",
    "f8_griewank",
    "f9_weierstrass",
    "f10_levy",
    "f11_schwefel_2_26",
    "f12_styblinski_tang",
    "f13_michalewicz",
    "f14_alpine_1",
    "f15_happy_cat",
    "f16_hgbat",
    "f17_bent_cigar",
    "f18_discus",
    "f19_penalized_1",
    "f20_zakharov_2",
]

DEFAULT_BOUNDS = {
    "f1_sphere": (-100, 100),
    "f2_schwefel_2_22": (-10, 10),
    "f3_zakharov": (-10, 10),
    "f4_rosenbrock": (-30, 30),
    "f5_powell_sum": (-10, 10),
    "f6_rastrigin": (-5.12, 5.12),
    "f7_ackley": (-32.768, 32.768),
    "f8_griewank": (-600, 600),
    "f9_weierstrass": (-0.5, 0.5),
    "f10_levy": (-10, 10),
    "f11_schwefel_2_26": (-500, 500),
    "f12_styblinski_tang": (-5, 5),
    "f13_michalewicz": (0, np.pi),
    "f14_alpine_1": (-10, 10),
    "f15_happy_cat": (-100, 100),
    "f16_hgbat": (-100, 100),
    "f17_bent_cigar": (-100, 100),
    "f18_discus": (-100, 100),
    "f19_penalized_1": (-50, 50),
    "f20_zakharov_2": (-10, 10),
}


class BenchmarkFunctions:
    """Twenty shifted classical benchmark functions with overflow-safe evaluation."""

    def __init__(
        self,
        dim: int,
        lower_bound,
        upper_bound,
        random_shift: bool = True,
        fixed_shift_vector=None,
    ):
        self.dim = dim
        self.lower_bound = np.atleast_1d(lower_bound)
        self.upper_bound = np.atleast_1d(upper_bound)

        if random_shift:
            self.o = np.random.uniform(self.lower_bound, self.upper_bound, dim)
        elif fixed_shift_vector is not None:
            self.o = np.asarray(fixed_shift_vector, dtype=float)
        else:
            self.o = np.zeros(dim)

    def _prepare(self, x):
        return np.asarray(x, dtype=float) - self.o

    def f1_sphere(self, x):
        z = self._prepare(x)
        return float(np.sum(np.clip(z**2, 0, 1e300)))

    def f2_schwefel_2_22(self, x):
        z = self._prepare(x)
        abs_z = np.abs(z)
        log_prod = np.sum(np.log(abs_z + 1e-100))
        return float(np.sum(abs_z) + np.exp(np.clip(log_prod, -700, 700)))

    def f3_zakharov(self, x):
        z = self._prepare(x)
        sum1 = np.sum(np.clip(z**2, 0, 1e300))
        sum2 = np.sum(0.5 * np.arange(1, self.dim + 1) * z)
        sum2_safe = np.clip(sum2, -1e75, 1e75)
        return float(sum1 + sum2_safe**2 + sum2_safe**4)

    def f4_rosenbrock(self, x):
        z = self._prepare(x)
        z_sq = np.clip(z[:-1] ** 2, -1e150, 1e150)
        return float(np.sum(np.clip(100.0 * (z[1:] - z_sq) ** 2 + (1.0 - z[:-1]) ** 2, 0, 1e300)))

    def f5_powell_sum(self, x):
        z = self._prepare(x)
        abs_z = np.clip(np.abs(z), 0, 1e100)
        exp = np.arange(2, self.dim + 2) / self.dim
        return float(np.sum(np.power(abs_z, exp)))

    def f6_rastrigin(self, x):
        z = self._prepare(x)
        return float(10 * self.dim + np.sum(np.clip(z**2, 0, 1e300) - 10 * np.cos(2 * np.pi * z)))

    def f7_ackley(self, x):
        z = self._prepare(x)
        sum1 = np.sum(np.clip(z**2, 0, 1e300))
        sum2 = np.sum(np.cos(2 * np.pi * z))
        return float(-20 * np.exp(-0.2 * np.sqrt(sum1 / self.dim)) - np.exp(sum2 / self.dim) + 20 + np.e)

    def f8_griewank(self, x):
        z = self._prepare(x)
        sum_sq = np.sum(np.clip(z**2, 0, 1e300)) / 4000.0
        prod_cos = np.prod(np.cos(z / np.sqrt(np.arange(1, self.dim + 1))))
        return float(sum_sq - prod_cos + 1)

    def f9_weierstrass(self, x, a=0.5, b=3, k_max=20):
        z = self._prepare(x)
        k_values = np.arange(k_max + 1)
        term1 = a**k_values[:, np.newaxis]
        term2 = b**k_values[:, np.newaxis]
        cos_term = np.cos(2 * np.pi * term2 * (z + 0.5))
        sum_outer = np.sum(term1 * cos_term)
        cos_const = np.cos(2 * np.pi * b**k_values * 0.5)
        constant = self.dim * np.sum(a**k_values * cos_const)
        return float(sum_outer - constant)

    def f10_levy(self, x):
        z = self._prepare(x)
        w = 1 + (z - 1) / 4
        term1 = (np.sin(np.pi * w[0])) ** 2
        term_last = (w[-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[-1])) ** 2)
        sum_mid = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * w[:-1] + 1)) ** 2))
        return float(term1 + sum_mid + term_last)

    def f11_schwefel_2_26(self, x):
        z = self._prepare(x)
        return float(418.9829 * self.dim - np.sum(z * np.sin(np.sqrt(np.abs(z)))))

    def f12_styblinski_tang(self, x):
        z = self._prepare(x)
        z_safe = np.clip(z, -1e75, 1e75)
        return float(0.5 * np.sum(z_safe**4 - 16 * z_safe**2 + 5 * z_safe))

    def f13_michalewicz(self, x, m=10):
        z = self._prepare(x)
        return float(-np.sum(np.sin(z) * (np.sin(np.arange(1, self.dim + 1) * z**2 / np.pi)) ** (2 * m)))

    def f14_alpine_1(self, x):
        z = self._prepare(x)
        return float(np.sum(np.abs(z * np.sin(z) + 0.1 * z)))

    def f15_happy_cat(self, x, alpha=1 / 8):
        z = self._prepare(x)
        z_sq = np.sum(np.clip(z**2, 0, 1e300))
        inner = np.abs(z_sq - self.dim)
        return float((inner**2) ** alpha + (0.5 * z_sq + np.sum(z)) / self.dim + 0.5)

    def f16_hgbat(self, x, alpha=1 / 4):
        z = self._prepare(x)
        z_sq = np.sum(np.clip(z**2, 0, 1e300))
        z_sum = np.sum(z)
        inner = np.clip(np.abs(z_sq**2 - z_sum**2), 0, 1e150)
        return float((inner**2) ** alpha + (0.5 * z_sq + z_sum) / self.dim + 0.5)

    def f17_bent_cigar(self, x):
        z = self._prepare(x)
        return float(z[0] ** 2 + 1e6 * np.sum(np.clip(z[1:] ** 2, 0, 1e294)))

    def f18_discus(self, x):
        z = self._prepare(x)
        return float(1e6 * z[0] ** 2 + np.sum(np.clip(z[1:] ** 2, 0, 1e300)))

    def f19_penalized_1(self, x):
        z = self._prepare(x)

        def u(xi, a, k, m):
            res = np.zeros_like(xi)
            over = xi > a
            under = xi < -a
            res[over] = k * (np.clip(xi[over] - a, 0, 1e75) ** m)
            res[under] = k * (np.clip(-xi[under] - a, 0, 1e75) ** m)
            return res

        y = 1 + (z + 1) / 4
        term1 = 10 * (np.sin(np.pi * y[0])) ** 2
        term_last = (y[-1] - 1) ** 2
        sum_mid = np.sum((y[:-1] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * y[1:])) ** 2))
        penalty = np.sum(u(z, 10, 100, 4))
        return float((np.pi / self.dim) * (term1 + sum_mid + term_last) + penalty)

    def f20_zakharov_2(self, x):
        z = self._prepare(x)
        sum_abs = np.clip(np.sum(np.abs(z)), 0, 1e150)
        exp_term = np.exp(np.clip(np.sum(np.sin(z**2)), -700, 700))
        return float(sum_abs * exp_term)


RobustBenchmarkFunctions = BenchmarkFunctions


def make_shifted(func, dim: int, seed: int = 0):
    """Wrap a base function with a fixed random shift vector."""
    rng = np.random.RandomState(seed)
    shift = rng.uniform(-50, 50, dim)

    def shifted(x):
        return func(x - shift)

    return shifted


def build_problem_suite(
    dim: int = 30,
    random_shift: bool = False,
    fixed_shifts: dict[str, np.ndarray] | None = None,
):
    """Build the list of 20 benchmark problems for a given dimension."""
    problems = []
    for name in FUNCTION_NAMES:
        lb, ub = DEFAULT_BOUNDS[name]
        func_dim = 10 if name == "f13_michalewicz" else dim
        lb_arr = np.full(func_dim, lb)
        ub_arr = np.full(func_dim, ub)

        shift = None
        if fixed_shifts and name in fixed_shifts:
            shift = fixed_shifts[name]
        elif not random_shift:
            shift = np.random.uniform(lb_arr, ub_arr, func_dim)

        bench = BenchmarkFunctions(
            func_dim,
            lb_arr,
            ub_arr,
            random_shift=random_shift,
            fixed_shift_vector=shift,
        )
        problems.append(
            {
                "name": name,
                "function": getattr(bench, name),
                "dimension": func_dim,
                "bounds": (lb, ub),
                "shift_vector": bench.o.tolist(),
            }
        )
    return problems
