import sys,random,pickle,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from experiments.optimizer_checkpoint import CheckpointObjective
from experiments.final_benchmark import BudgetComplete
from hppso.experiments.runners import DEFAULT_ALGORITHMS
from hppso.algorithms.ga_mpc import run_ga_mpc

class CheckpointTests(unittest.TestCase):
    def execute(self,algorithm,budget,folder):
        calls=[0]
        def sphere(x):calls[0]+=1;return float(x@x)
        job=dict(problem=0,dimension=3,algorithm=algorithm,seed=3,population=8,budget=budget,schedule_horizon=181)
        objective=CheckpointObjective(sphere,job,folder,'test',interval=20)
        try:
            args=(objective,3,-2.,2.,8,2**31-1)
            with self.assertRaises(BudgetComplete):
                if algorithm=='GA-MPC':
                    random.seed(3);np.random.seed(3);run_ga_mpc(objective,3,-2,2,ps=8,max_gen=None)
                elif algorithm=='CMA-ES':DEFAULT_ALGORITHMS[algorithm](*args,2**62,3,stopfitness=-np.inf)
                else:DEFAULT_ALGORITHMS[algorithm](*args,3)
            return calls[0],objective.best,objective.best_position.copy()
        finally:objective.close()
    def test_all_nine_mid_generation_resume_and_extension(self):
        for algorithm in DEFAULT_ALGORITHMS:
            if algorithm=='Sep-CMA-ES':continue
            with self.subTest(algorithm=algorithm),tempfile.TemporaryDirectory() as folder:
                folder=Path(folder);full=folder/'full';split=folder/'split'
                reference=self.execute(algorithm,181,full)
                first=self.execute(algorithm,73,split);resumed=self.execute(algorithm,181,split)
                self.assertEqual(reference[0],181);self.assertEqual(first[0],73);self.assertEqual(resumed[0],108)
                self.assertEqual((full/'evaluations.bin').read_bytes(),(split/'evaluations.bin').read_bytes())
                self.assertEqual(reference[1],resumed[1]);np.testing.assert_array_equal(reference[2],resumed[2])
                again=self.execute(algorithm,181,split);self.assertEqual(again[0],0)
                with (split/'state.pkl').open('rb') as f:state=pickle.load(f)
                self.assertTrue(state['algorithm_frames']);self.assertEqual(state['evaluations'],181)
    def test_different_candidate_rejected_without_objective_call(self):
        with tempfile.TemporaryDirectory() as folder:
            job=dict(problem=0,dimension=1,algorithm='test',seed=0,population=1,budget=2,schedule_horizon=3)
            objective=CheckpointObjective(lambda x:1.,job,folder,'test');objective(np.array([0.]));objective.close()
            objective=CheckpointObjective(lambda x:self.fail('old objective called'),job,folder,'test')
            try:
                with self.assertRaises(RuntimeError):objective(np.array([1.]))
            finally:objective.close()

if __name__=='__main__':unittest.main()
