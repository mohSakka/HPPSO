import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hppso.algorithms.openness_ablation import OpennessPSO, VARIANTS, openness_delta
from hppso.algorithms.pso import PSO
from experiments.openness_ablation import DEFAULT_CONFIG, load_config, jobs, trial, result_path, report, fingerprint
from experiments.final_benchmark import read_result


class OpennessTests(unittest.TestCase):
    def test_four_gates_targets_equal_fitness_and_disabled_particles(self):
        positions = np.array([[1.], [2.], [-2.], [3.], [4.]])
        pbests = positions * 0.1
        scores = np.array([1., 4., 4., 9., 16.])
        peers = np.array([1, 0, 1, 4, 3])
        enabled = np.array([True, True, True, False, True])
        expected = {
            "PSO": [0, 0, 0, 0, 0],
            "PSO-open-worse-pbest": [-.32, 0, 0, 0, 0],
            "PSO-open-better-pbest": [0, -.76, 0, 0, -1.48],
            "PSO-open-any-pbest": [-.32, -.76, .88, 0, -1.48],
            "PSO-open-worse-current": [.4, 0, 0, 0, 0],
        }
        for name, (gate, target) in VARIANTS.items():
            np.testing.assert_allclose(openness_delta(positions, pbests, scores, peers,
                enabled, gate, target, .4)[:, 0], expected[name], atol=1e-14)

    def test_baseline_matches_existing_standard_pso_without_mutation(self):
        objective = lambda x: float(x @ x)
        bounds = np.array([[-10.] * 4, [10.] * 4])
        np.random.seed(17)
        old = PSO(objective, list(zip(*bounds)), 6, 12, mutation_rate=0)
        old_position, old_score, _ = old.optimize()
        new = OpennessPSO(objective, bounds, 6, "PSO", 17)
        for _ in range(12):
            new.step()
        np.testing.assert_allclose(new.positions, old.positions, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(new.global_best, old_position)
        self.assertAlmostEqual(new.global_score, old_score)

    def test_disabled_openness_preserves_matched_base_random_stream(self):
        opts = [OpennessPSO(lambda x: float(x @ x), ([-10.] * 3, [10.] * 3),
                           8, name, 13, threshold=1) for name in VARIANTS]
        for _ in range(7):
            for opt in opts:
                opt.step()
            for opt in opts[1:]:
                np.testing.assert_array_equal(opt.positions, opts[0].positions)
                np.testing.assert_array_equal(opt.velocities, opts[0].velocities)
                np.testing.assert_array_equal(opt.openness, opts[0].openness)

    def test_protocol_and_exact_budget_at_partial_generation(self):
        cfg = load_config(DEFAULT_CONFIG)
        self.assertEqual((cfg["dimension"], len(cfg["seeds"]), len(list(jobs(cfg)))), (100, 10, 200))
        cfg.update(dimension=4, population=4, budget=23, trace_points=8)
        with tempfile.TemporaryDirectory() as folder:
            for algorithm in VARIANTS:
                job = dict(function="f1_sphere", algorithm=algorithm, seed=0)
                record = trial(job, cfg, folder, "test")
                self.assertEqual((record["status"], record["evaluations"]), ("budget_reached", 23))
                saved = read_result(result_path(folder, job))
                self.assertEqual(saved["trace"][-1][0], 23)
                self.assertEqual(saved["trace"][-1][1], saved["best_score"])
                self.assertTrue(np.all(np.diff(np.array(saved["trace"])[:, 1]) <= 0))

    def test_challenging_protocol_includes_all_ten_functions(self):
        from hppso.benchmarks.additional import NAMES
        cfg = load_config(DEFAULT_CONFIG.with_name("openness_ablation_challenging.json"))
        self.assertEqual(tuple(cfg["functions"]), NAMES)
        self.assertEqual(len(list(jobs(cfg))), 500)
        self.assertEqual((cfg["dimension"], cfg["budget"], cfg["population"]), (100, 100000, 50))

    def test_final_selection_has_nine_functions_and_one_composition(self):
        cfg = load_config(DEFAULT_CONFIG.with_name("openness_ablation_final9.json"))
        self.assertEqual(len(cfg["functions"]), 9)
        self.assertEqual([f for f in cfg["functions"] if "composition" in f], ["f25_composition"])
        self.assertEqual(len(list(jobs(cfg))), 450)
        self.assertNotIn("f30_composition_multimodal", cfg["functions"])

    def test_more_than_four_functions_survive_both_report_types(self):
        import pandas as pd
        from experiments.plot_openness_detail import plot_details
        cfg = load_config(DEFAULT_CONFIG.with_name("openness_ablation_challenging.json"))
        cfg.update(dimension=4, population=4, budget=23, trace_points=8,
                   bootstrap_resamples=40, seeds=[0, 1], functions=[
                       "f21_expanded_schaffer_f6", "f22_rotated_ellipsoid", "f25_composition",
                       "f29_rotated_schaffer_f7", "f30_composition_multimodal"])
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "plan.json").write_text(json.dumps(dict(config=cfg, fingerprint="test", variants=VARIANTS)))
            for job in jobs(cfg):
                result = trial(job, cfg, folder, "test")
                self.assertEqual(result["status"], "budget_reached")
            report(output)
            plot_details(output)
            self.assertEqual(set(pd.read_csv(output / "summary.csv").function), set(cfg["functions"]))
            self.assertEqual(len(pd.read_csv(output / "summary.csv")), 25)
            self.assertEqual(json.loads((output / "report_status.json").read_text())["complete_functions"], cfg["functions"])
            for file in ("overview_02.png", "late_convergence_linear_02.png", "final_scores_linear_02.png",
                         "f30_composition_multimodal.png", "all_functions.pdf", "linear_detail.pdf"):
                self.assertTrue((output / "convergence" / file).is_file(), file)
            html = (output / "index.html").read_text()
            self.assertIn("overview_02.png", html)
            self.assertIn("final_scores_linear_02.png", html)

    def test_subset_reuse_preserves_execution_provenance_and_rejects_changed_budget(self):
        from experiments.reuse_openness_subset import reuse_subset
        from experiments.final_benchmark import suite
        from hppso.benchmarks.classical import problem_definitions
        cfg = load_config(DEFAULT_CONFIG)
        cfg.update(dimension=4, population=4, budget=23, trace_points=8, seeds=[0])
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source"
            source.mkdir()
            digest = fingerprint(cfg)
            (source / "plan.json").write_text(json.dumps(dict(config=cfg, fingerprint=digest,
                numpy_version=np.__version__, python_version="test", variants=VARIANTS)))
            problems = suite(4, cfg["instance_seed"])
            (source / "problem_definitions.json").write_text(json.dumps(problem_definitions(
                [problems[name] for name in cfg["functions"]])))
            job = dict(function="f1_sphere", algorithm="PSO", seed=0)
            trial(job, cfg, source, digest)
            raw = result_path(source, job).read_bytes()
            subset = dict(cfg, functions=["f1_sphere"])
            destination = Path(folder) / "subset"
            self.assertEqual(reuse_subset(source, destination, subset), 1)
            imported = read_result(result_path(destination, job))
            self.assertEqual(imported["fingerprint"], fingerprint(subset))
            self.assertEqual(imported["reused_from"]["execution_fingerprint"], digest)
            self.assertEqual(imported["trace"], read_result(result_path(source, job))["trace"])
            self.assertEqual(result_path(source, job).read_bytes(), raw)
            with self.assertRaisesRegex(ValueError, "Only the function selection"):
                reuse_subset(source, Path(folder) / "invalid", dict(subset, budget=24))


if __name__ == "__main__":
    unittest.main()
