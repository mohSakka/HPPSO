import sys
import unittest
from pathlib import Path
from decimal import Decimal, localcontext

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hppso.benchmarks.classical import BenchmarkFunctions
from reproduction.aggregate import summarize_performance, average_ranks, wilcoxon_pairwise_scores
from reproduction.schema import MergedDataset, MergeReport


class StabilityTests(unittest.TestCase):
    def bench(self, n):
        return BenchmarkFunctions(n, -10, 10, random_shift=False)

    def test_matches_raw_transform_in_representable_range(self):
        rng = np.random.default_rng(7)
        b = self.bench(30)
        for x in rng.uniform(-2, 2, (20, 30)):
            expected = np.log1p(np.sum(abs(x)) + np.prod(abs(x)))
            self.assertAlmostEqual(b.f2_schwefel_2_22(x), expected, places=12)

    def test_large_product_matches_high_precision_reference(self):
        with localcontext() as ctx:
            ctx.prec = 80
            expected = float((Decimal(1) + 10000 + Decimal(10)**1000).ln())
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            actual = self.bench(1000).f2_schwefel_2_22(np.full(1000, 10.))
        self.assertAlmostEqual(actual, expected, places=10)

    def test_zero_factors_and_optimum_are_exact(self):
        b = self.bench(1000)
        self.assertEqual(b.f2_schwefel_2_22(np.zeros(1000)), 0)
        x = np.full(1000, 10.)
        x[0] = 0
        self.assertAlmostEqual(b.f2_schwefel_2_22(x), np.log1p(9990), places=12)

    def test_large_values_do_not_saturate(self):
        b = self.bench(1000)
        self.assertLess(b.f2_schwefel_2_22(np.full(1000, 5.)),
                        b.f2_schwefel_2_22(np.full(1000, 6.)))

    def test_nonfinite_function_excluded_from_all_comparisons(self):
        costs = {("f1_sphere", "HPPSO"): [1., 1.],
                 ("f1_sphere", "PSO"): [2., 2.],
                 ("f2_schwefel_2_22", "HPPSO"): [np.inf, np.inf],
                 ("f2_schwefel_2_22", "PSO"): [1., 1.]}
        data = MergedDataset(1000, costs, {}, {}, MergeReport(1000))
        perf = summarize_performance(data)
        excluded = perf[perf.function == "f2_schwefel_2_22"]
        self.assertTrue(excluded["rank"].isna().all())
        ranks = average_ranks(data).set_index("algorithm")["average_rank"]
        self.assertEqual(sorted(ranks.tolist()), [1., 2.])
        result = wilcoxon_pairwise_scores(data)
        self.assertTrue(np.isfinite(result.normalized_score).all())


if __name__ == "__main__":
    unittest.main()
