import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hppso.benchmarks.additional import NAMES, build_additional_suite
from experiments.scaling import BudgetExhausted, EvaluationBudget, run_scaling
from experiments.plots import make_plots


class ScalingTests(unittest.TestCase):
    def test_optima_feasibility_and_reproducibility(self):
        for dim in (6, 100, 1000):
            a, b = build_additional_suite(dim), build_additional_suite(dim)
            for p, q in zip(a, b):
                self.assertEqual(p.definition(), q.definition())
                lo, hi = p.bounds
                self.assertTrue(np.all((lo < p.shift) & (p.shift < hi)))
                self.assertEqual(p(p.shift), 0)
                self.assertGreater(p(p.shift+np.linspace(.1, .5, dim)), 0)
                z = np.linspace(-1., 2., dim)
                if p.permutations.size:
                    np.testing.assert_allclose(p.rotate(z) @ p.rotate(z), z @ z, rtol=1e-12)

    def test_rotated_ellipsoid_has_cross_coordinate_coupling(self):
        p = build_additional_suite(6, names=["f22_rotated_ellipsoid"])[0]
        e1, e2 = np.eye(6)[:2]
        interaction = p(p.shift+e1+e2)-p(p.shift+e1)-p(p.shift+e2)
        self.assertGreater(abs(interaction), 1)

    def test_budget_never_exceeded(self):
        f = EvaluationBudget(lambda x: x, 2)
        self.assertEqual(f(4), 4)
        self.assertEqual(f(3), 3)
        with self.assertRaises(BudgetExhausted):
            f(1)
        self.assertEqual((f.count, f.best), (2, 3))

    def test_end_to_end_counts_metadata_and_plots(self):
        cfg = dict(experiment_type="scaling", dimensions=[6, 9], runs=2,
                   pop_size=5, budget_modes=["fixed", "proportional"],
                   fixed_evaluations=17, evaluations_per_dimension=4,
                   functions=["f21_expanded_schaffer_f6", "f25_composition"],
                   algorithms=["HPPSO", "Original PSO", "SHADE", "Sep-CMA-ES", "GA-MPC"])
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            out = Path(folder)
            results = run_scaling(cfg, out)
            json.dumps(results, allow_nan=False)
            for row in results["runs"]:
                self.assertNotEqual(row["status"], "failed", row["error"])
                self.assertLessEqual(row["evaluations"], row["budget"])
                self.assertGreater(row["evaluations"], 0)
            paths = make_plots(results, cfg, out)
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(p.stat().st_size > 0 for p in paths))
            definitions = json.loads((out/"problem_definitions.json").read_text())
            self.assertEqual(set(definitions), {"6", "9"})


if __name__ == "__main__":
    unittest.main()
