import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiments.diversity_study import measure,protocol
from experiments.trait_plan import ROOT,load_config
from experiments.trait_batch import BatchBudget,BatchedTraitPSO


class DiversityTests(unittest.TestCase):
    def test_known_geometry_and_movement(self):
        x=np.array([[[0.,0.],[2.,2.]]]);width=np.array([2.,2.])
        np.testing.assert_allclose(measure(x,width),[[.5,1.,0.]])
        np.testing.assert_allclose(measure(x,width,x-1),[[.5,1.,.5]])
        np.testing.assert_allclose(measure(np.ones((4,5,3)),np.ones(3)),0)

    def test_invariance_under_shift_and_coordinate_units(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(4,8,5));previous=x-.1
        width=np.arange(1,6.)
        expected=measure(x,width,previous)
        np.testing.assert_allclose(measure(x+100,width,previous+100),expected,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(measure(x*width,width*width,previous*width),expected)

    def test_recording_does_not_change_optimization(self):
        cfg=load_config(ROOT/'experiments/configs/trait_study_zero_velocity13.json')
        _,specs=protocol(cfg);bounds=(np.full(5,-4.),np.full(5,4.))
        def engine():
            tracker=BatchBudget(lambda x:np.sum(x*x,axis=-1),500,4,5,20)
            return BatchedTraitPSO(tracker,bounds,10,13,specs,cfg)
        observed,control=engine(),engine()
        while observed.tracker.count<500:
            previous=observed.positions.copy();observed.step()
            measure(observed.positions,observed.width,previous)
            control.step()
            np.testing.assert_array_equal(observed.positions,control.positions)
            np.testing.assert_array_equal(observed.tracker.best,control.tracker.best)
        self.assertEqual(observed.tracker.count,control.tracker.count)


if __name__=='__main__':unittest.main()
