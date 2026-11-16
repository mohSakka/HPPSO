import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiments.smoke import settings, combined_suite, execute_trial, algorithms_for, DIMENSIONS, main


class SmokeTests(unittest.TestCase):
    def test_legacy_result_plan_is_rejected_before_starting(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"plan.json"
            old={"functions":["scaling_sphere"],"functions_per_dimension":25}
            path.write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError,"different plan"):
                main(["--output-dir",folder,"--dimensions","30","--phase","smoke"])
            self.assertEqual(json.loads(path.read_text()),old)
            self.assertFalse((Path(folder)/"process_id.txt").exists())

    def test_default_dimensions(self):
        self.assertEqual(DIMENSIONS, (30,))

    def test_cma_selection(self):
        self.assertIn("CMA-ES", algorithms_for(30))
        self.assertNotIn("Sep-CMA-ES", algorithms_for(30))
        self.assertEqual(len(algorithms_for(30)), 9)
        for dim in (250, 500, 750, 1000):
            self.assertNotIn("CMA-ES", algorithms_for(dim))
            self.assertIn("Sep-CMA-ES", algorithms_for(dim))
            self.assertEqual(len(algorithms_for(dim)), 9)

    def test_revised_budget_plan(self):
        self.assertEqual(settings(30, "full"), dict(population=30, iterations=3000, budget=90030))
        self.assertEqual(settings(1000, "full")["population"], 300)
        for dim in (250, 500, 750, 1000):
            cfg = settings(dim, "full")
            self.assertEqual(cfg["budget"], cfg["population"]*(cfg["iterations"]+1))
            self.assertGreaterEqual(cfg["budget"], 1000*dim)
            self.assertLess(cfg["budget"]-1000*dim, cfg["population"])

    def test_all_functions_use_requested_dimension(self):
        suite, definitions = combined_suite(30)
        self.assertEqual(len(suite), 30)
        self.assertEqual(len(definitions), 30)
        self.assertTrue(all(p["dimension"] == 30 for p in suite))
        self.assertEqual(next(p for p in suite if p["name"] == "f13_michalewicz")["function"].__self__.dim, 30)

    def test_worker_budget_and_negative_cma_targets(self):
        suite, _ = combined_suite(30)
        negative = next(p for p in suite if p["name"] == "f13_michalewicz")
        with tempfile.TemporaryDirectory() as folder:
            for algo in ("HPPSO", "CMA-ES", "Sep-CMA-ES"):
                identity = dict(phase="smoke", dimension=30, function=negative["name"],
                                algorithm=algo, seed=0, **settings(30, "smoke"))
                row = execute_trial((negative, identity), folder)
                self.assertNotEqual(row["status"], "failed", row["error"])
                self.assertGreaterEqual(row["evaluations"], 60)
                self.assertLessEqual(row["evaluations"], 90)
                self.assertTrue(np.isfinite(row["best_score"]))


if __name__ == "__main__":
    unittest.main()
