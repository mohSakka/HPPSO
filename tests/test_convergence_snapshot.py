import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from experiments.convergence_snapshot import align_curves, step_curve, completed_functions


class ConvergenceTests(unittest.TestCase):
    def test_uses_stepwise_values_and_common_observed_range(self):
        rows=[dict(trace=[[1,10],[5,2]],evaluations=5,best_score=2),
              dict(trace=[[1,20],[3,10],[7,0]],evaluations=7,best_score=0)]
        x,median,lower,upper=align_curves(rows)
        np.testing.assert_array_equal(x,[1,3,5])
        np.testing.assert_array_equal(median,[15,10,6])
        np.testing.assert_array_equal(lower,[10,10,2])
        np.testing.assert_array_equal(upper,[20,10,10])

    def test_final_unsampled_evaluation_is_retained(self):
        x,y=step_curve(dict(trace=[[1,10],[3,5]],evaluations=5,best_score=1))
        np.testing.assert_array_equal(x,[1,3,5])
        np.testing.assert_array_equal(y,[10,5,1])


class CompletedFunctionTests(unittest.TestCase):
    def test_selected_algorithms_define_completion(self):
        rows=[dict(dimension=30,function="sphere",algorithm=a,seed=seed,status="budget_exhausted")
              for a in ("HPPSO","CMA-ES") for seed in (0,1)]
        self.assertEqual(completed_functions(rows,30,["HPPSO","CMA-ES"]),["sphere"])
        self.assertEqual(completed_functions(rows,30,["HPPSO","CMA-ES","Sep-CMA-ES"]),[])
