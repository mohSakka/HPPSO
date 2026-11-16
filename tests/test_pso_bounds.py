"""Shared boundary-policy tests across all PSO variants."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hppso.algorithms import HPPSO, PSO
from hppso.algorithms._pso_bounds import reflect_bounds


class BoundsTests(unittest.TestCase):
    def test_repeated_reflections_and_fixed_dimensions(self):
        x = np.array([[-.2, 1.2, 2.2, 3.2, -1.2, 7.]])
        v = np.ones_like(x)
        reflect_bounds(x, v, np.zeros(6), np.array([1, 1, 1, 1, 1, 0]))
        np.testing.assert_allclose(x, [[.2, .8, .2, .8, .8, 0]])
        np.testing.assert_allclose(v, [[-1, -1, 1, -1, 1, 0]])

    def test_all_variants_share_defaults_and_reflect_velocity(self):
        objective = lambda x: float(x @ x)
        variants = [PSO(objective, [(0, 1)], 1, 1, mutation_rate=0),
                    PSO(objective, [(0, 1)], 1, 1, mutation_rate=1),
                    PSO(objective, [(0, 1)], 1, 1, mutation_rate=0,
                        w_random_range=(.4, .9)),
                    HPPSO(objective, n_pop=1, dimensions=1, max_it=1, bounds=(0, 1))]
        for opt in variants:
            with self.subTest(variant=type(opt).__name__, mutation=getattr(opt, 'mutation_rate', None)):
                self.assertEqual((opt.c1, opt.c2), (2, 2))
                np.testing.assert_allclose(opt.v_limit, [.2])
                if isinstance(opt, HPPSO):
                    opt.X[:] = .95
                    opt.P[:] = .95
                    opt.g_best[:] = .95
                    opt.V[:] = 10
                    opt.personality[:] = 0
                else:
                    opt.positions[:] = .95
                    opt.personal_best_positions[:] = .95
                    opt.global_best_position[:] = .95
                    opt.velocities[:] = 10
                with patch('numpy.random.normal', side_effect=lambda loc, scale, size: np.zeros(size)):
                    opt.optimize()
                x = opt.X if isinstance(opt, HPPSO) else opt.positions
                v = opt.V if isinstance(opt, HPPSO) else opt.velocities
                np.testing.assert_allclose(x, [[.85]])
                np.testing.assert_allclose(v, [[-.2]])

    def test_explicit_velocity_limits_match(self):
        objective = lambda x: float(x @ x)
        for limit in (.03, [.03, .07]):
            a = HPPSO(objective, dimensions=2, bounds=(0, 1), v_max=limit)
            b = PSO(objective, [(0, 1)] * 2, 3, 1, v_max=limit)
            np.testing.assert_array_equal(a.v_limit, b.v_limit)


if __name__ == "__main__":
    unittest.main()
