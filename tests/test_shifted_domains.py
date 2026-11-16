import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hppso.benchmarks.classical import (BenchmarkFunctions, DEFAULT_BOUNDS,
                                      build_problem_suite, problem_definitions)
from hppso.experiments.runners import DEFAULT_ALGORITHMS, run_benchmark


class ShiftedDomainTests(unittest.TestCase):
    def test_every_function_has_same_base_domain_and_values(self):
        np.random.seed(42)
        for problem in build_problem_suite(dim=5):
            with self.subTest(function=problem["name"]):
                n = problem["dimension"]
                shift = np.array(problem["shift_vector"])
                lo, hi = problem["bounds"]
                base_lo, base_hi = DEFAULT_BOUNDS[problem["name"]]
                np.testing.assert_allclose(lo-shift, base_lo, atol=1e-12)
                np.testing.assert_allclose(hi-shift, base_hi, atol=1e-12)
                np.testing.assert_allclose(hi-lo, base_hi-base_lo, atol=1e-12)
                self.assertEqual(len(np.unique(shift)), n)
                base = BenchmarkFunctions(n, base_lo, base_hi, random_shift=False)
                evaluate = getattr(base, problem["name"])
                for fraction in (.1, .5, .9):
                    z = np.full(n, base_lo + fraction*(base_hi-base_lo))
                    np.testing.assert_allclose(problem["function"](z+shift), evaluate(z),
                                               rtol=1e-9, atol=1e-9)

    def test_nonzero_base_minimizers_are_feasible_after_extreme_shifts(self):
        shifts = {name: np.array([-100., 0., 100.]) for name in
                  ("f4_rosenbrock", "f10_levy", "f11_schwefel_2_26")}
        suite = {p["name"]: p for p in build_problem_suite(3, fixed_shifts=shifts)}
        for name, shift in shifts.items():
            p = suite[name]
            zstar = np.full(3, 420.9687462275036 if "schwefel" in name else 1.)
            xstar = zstar+shift
            lo, hi = p["bounds"]
            self.assertTrue(np.all((lo <= xstar) & (xstar <= hi)))
            self.assertEqual(len(np.unique(xstar)), 3)
            # Schwefel's rounded 418.9829 constant gives a small nonzero value.
            self.assertLess(abs(p["function"](xstar)), 1e-3)

    def test_saved_definitions_reconstruct_exact_problem(self):
        suite = build_problem_suite(3, random_shift=True,
                                    fixed_shifts={"f1_sphere": [4., -5., 6.]})
        saved = json.loads(json.dumps(problem_definitions(suite)))
        shifts = {name: rec["shift_vector"] for name, rec in saved.items()}
        rebuilt = build_problem_suite(3, random_shift=True, fixed_shifts=shifts)
        self.assertEqual(saved, problem_definitions(rebuilt))
        self.assertEqual(saved["f1_sphere"]["shift_vector"], [4., -5., 6.])

    def test_all_runners_accept_translated_vector_bounds(self):
        p = build_problem_suite(3, fixed_shifts={"f1_sphere": [100., -200., 300.]})[0]
        original = p["function"]
        lo, hi = p["bounds"]
        def checked(x):
            self.assertTrue(np.all(x >= lo))
            self.assertTrue(np.all(x <= hi))
            return original(x)
        p["function"] = checked
        with contextlib.redirect_stdout(io.StringIO()):
            results = run_benchmark(DEFAULT_ALGORITHMS, [p], num_runs=1,
                                    pop_size=10, max_iters=2)
        for name, result in results.items():
            with self.subTest(algorithm=name):
                self.assertTrue(np.isfinite(result[p["name"]]["all_final_scores"]).all())

    def test_invalid_shift_rejected(self):
        for shift in ([1.], [1., np.nan], [1., np.inf]):
            with self.assertRaises(ValueError):
                BenchmarkFunctions(2, -10, 10, fixed_shift_vector=shift)


if __name__ == "__main__":
    unittest.main()
