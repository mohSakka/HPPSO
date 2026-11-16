import sys,random,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hppso.algorithms.ga_mpc import run_ga_mpc

class ReuseTests(unittest.TestCase):
    def test_identical_search_points_and_rng_with_fewer_calls(self):
        scope={};exec((ROOT/'artifacts/ga_mpc_before_fitness_reuse.py').read_text(),scope)
        for seed in range(4):
            observations=[]
            for runner in [scope['run_ga_mpc'],run_ga_mpc]:
                points=[]
                def objective(x):points.append(np.array(x).copy());return float(x@x)
                random.seed(seed);np.random.seed(seed)
                result=runner(objective,4,-5,5,ps=12,max_gen=12)
                observations.append((points,result,random.getstate(),np.random.get_state()))
            old,new=observations
            self.assertEqual({p.tobytes() for p in old[0]},{p.tobytes() for p in new[0]})
            self.assertEqual(old[2],new[2]);np.testing.assert_array_equal(old[3][1],new[3][1])
            self.assertLess(len(new[0]),len(old[0])/2)
            self.assertAlmostEqual(new[1][1],min(float(x@x) for x in old[0]))
            self.assertAlmostEqual(new[1][1],float(new[1][0]@new[1][0]))
    def test_no_crossover_and_duplicate_perturbation(self):
        random.seed(0);np.random.seed(0);calls=[]
        def objective(x):calls.append(x.copy());return float(x@x)
        x,score,history=run_ga_mpc(objective,2,-1,1,ps=12,max_gen=10,cr=0)
        self.assertEqual(len(history),10);self.assertAlmostEqual(score,float(x@x))
        self.assertEqual(score,min(float(p@p) for p in calls))
    def test_evaluation_only_stopping_without_generation_limit(self):
        from experiments.final_benchmark import StrictObjective,BudgetComplete
        random.seed(9);np.random.seed(9)
        objective=StrictObjective(lambda x:float(x@x),51)
        with self.assertRaises(BudgetComplete):
            run_ga_mpc(objective,3,-1,1,ps=12,max_gen=None)
        self.assertEqual(objective.count,51)
        self.assertEqual(objective.trace[-1][0],51)

if __name__=='__main__':unittest.main()
