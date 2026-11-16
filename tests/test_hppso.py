"""Regression checks for the manuscript's HPPSO update rules."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hppso.algorithms.hppso import HPPSO


class HPPSOTests(unittest.TestCase):
    def swarm(self, positions, **kwargs):
        x = np.asarray(positions, dtype=float)
        opt = HPPSO(lambda row: float(row @ row), n_pop=len(x), dimensions=x.shape[1],
                    max_it=1, bounds=(-100, 100), c1=0, c2=0, **kwargs)
        opt.X = x.copy()
        opt.P = x.copy()
        opt.fitness = np.sum(x*x, axis=1)
        opt.p_best_fitness = opt.fitness.copy()
        best = opt.fitness.argmin()
        opt.g_best = x[best].copy()
        opt.g_best_fitness = float(opt.fitness[best])
        opt.personality[:] = 0
        return opt

    def test_one_peer_then_worse_gate_uses_peer_personal_best(self):
        opt = self.swarm([[1], [2], [-2], [3]])
        opt.personality[:, 3] = 1
        opt.P[1] = .5  # The gate uses current fitness, attraction uses P_j.
        # After self-exclusion: peers [1, 0, 1, 2]. Worse, better, equal, better.
        with patch('numpy.random.randint', return_value=np.array([0, 0, 1, 2])) as draw:
            opt.optimize()
        draw.assert_called_once_with(0, 3, size=4)
        np.testing.assert_allclose(opt.V[:, 0], [-.2, 0, 0, 0])

    def test_openness_threshold_is_strict(self):
        opt = self.swarm([[1], [2]])
        opt.personality[:, 3] = opt.socialism_threshold
        opt.optimize()
        np.testing.assert_array_equal(opt.V, 0)

    def test_velocity_random_coefficients_are_per_particle(self):
        opt = self.swarm([[1, 2], [3, 4]])
        opt.c1 = 1
        opt.P += 2
        with patch('numpy.random.rand', side_effect=[np.array([[.25], [.75]]),
                                                    np.zeros((2, 1)), np.ones((2, 2))]):
            opt.optimize()
        np.testing.assert_allclose(opt.V, [[.5, .5], [1.5, 1.5]])

    def test_clip_velocity_before_curiosity_and_reflect_position(self):
        opt = self.swarm([[99, 0]], v_max=[1, 2], alpha_curio=1)
        opt.V[:] = 100
        opt.personality[:, 0] = 1
        with patch('numpy.random.normal', return_value=np.array([[5., -5.]])):
            opt.optimize()
        np.testing.assert_allclose(opt.V, [[-1, 2]])
        np.testing.assert_allclose(opt.X, [[95, -3]])

    def test_stagnation_resets_traits_at_threshold(self):
        opt = self.swarm([[0]], stagnation_threshold=2)
        opt.max_it = 2
        with patch('numpy.random.rand', side_effect=lambda *shape: np.full(shape, .25)):
            score, history = opt.optimize()
        self.assertEqual(score, 0)
        self.assertEqual(history, [0, 0])
        np.testing.assert_array_equal(opt.stagnation_counters, [0])
        np.testing.assert_array_equal(opt.personality, .25)

    def test_improvement_updates_bests_and_clears_stagnation(self):
        opt = self.swarm([[2]])
        opt.V[:] = -2.5  # inertia .4 gives a step of -1
        opt.stagnation_counters[:] = 3
        score, _ = opt.optimize()
        self.assertEqual(score, 1)
        np.testing.assert_array_equal(opt.P, [[1]])
        np.testing.assert_array_equal(opt.g_best, [1])
        np.testing.assert_array_equal(opt.stagnation_counters, [0])


if __name__ == '__main__':
    unittest.main()
