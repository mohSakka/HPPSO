import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiments.cec2011_budgeted import ALGORITHMS,problem,trial,path_for
from experiments.final_benchmark import read_result


class CECTests(unittest.TestCase):
    def test_all_native_problems_and_nine_algorithms_partial_budget(self):
        self.assertEqual(len(ALGORITHMS),9)
        self.assertNotIn('Sep-CMA-ES',ALGORITHMS)
        with tempfile.TemporaryDirectory() as folder:
            for number in range(1,23):
                f,lo,hi=problem(number)
                for algo in ALGORITHMS:
                    job=dict(problem=number,dimension=f.dimension,algorithm=algo,seed=0,population=300,budget=601)
                    r=trial(job,folder,'smoke')
                    self.assertEqual(r['status'],'budget_reached',str((number,algo,r['error'])))
                    saved=read_result(path_for(folder,job))
                    self.assertEqual(saved['evaluations'],601)
                    self.assertEqual(saved['trace'][-1][0],601)
                    self.assertTrue(np.isfinite(saved['best_score']))
                    x=np.array(saved['best_position'])
                    self.assertTrue(np.all(x>=lo-1e-9) and np.all(x<=hi+1e-9))
                    self.assertTrue(np.all(np.diff(np.array(saved['trace'])[:,1])<=0))
                print(f'Validated native T{number}: all nine algorithms, partial final population budget.',flush=True)


if __name__=='__main__':unittest.main()
