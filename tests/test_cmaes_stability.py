import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hppso.algorithms.cmaes import CMAES

class StabilityTests(unittest.TestCase):
    def engine(self):return CMAES(lambda x:float(x@x),3,-5,5,1.,60,20,100000)
    def test_near_singular_covariance_remains_consistent(self):
        opt=self.engine();opt.C=np.diag([-1e-23,1e-20,1e-7]);opt._decompose_covariance()
        self.assertEqual(opt.covariance_repairs,1)
        self.assertTrue(np.isfinite(opt.invsqrtC).all());self.assertTrue((opt.D>0).all())
        self.assertLessEqual(float(opt.D.max()/opt.D.min()),1e7*(1+1e-12))
        np.testing.assert_allclose(opt.invsqrtC@opt.C@opt.invsqrtC,np.eye(3),atol=1e-12)
    def test_nonfinite_covariance_is_explicit_failure(self):
        opt=self.engine();opt.C[0,0]=np.nan
        with self.assertRaises(FloatingPointError):opt._decompose_covariance()
    def test_finite_sphere_optimization(self):
        np.random.seed(4);opt=self.engine();opt.stopfitness=-np.inf
        with np.errstate(invalid='raise',divide='raise',over='raise'):x,value,trace=opt.optimize()
        self.assertTrue(np.isfinite(value));self.assertLess(value,trace[0]);self.assertEqual(len(trace),60)

if __name__=='__main__':unittest.main()
