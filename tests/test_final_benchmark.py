import argparse
import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from experiments.final_benchmark import (StrictObjective,BudgetComplete,trial,read_result,
    result_path,load_config,jobs,algorithms,shard_lease,ROOT)
from hppso.experiments.runners import DEFAULT_ALGORITHMS


class FinalTests(unittest.TestCase):
    def test_final_protocol_counts_and_cma_split(self):
        cfg=load_config(ROOT/"experiments/configs/final_benchmark.json")
        self.assertEqual(cfg["dimensions"],[30,300,600,1000])
        self.assertEqual(cfg["seeds"],list(range(20)))
        js=list(jobs(cfg));self.assertEqual(len(js),21600)
        self.assertEqual(sum(j["budget"] for j in js),10422000000)
        self.assertIn("CMA-ES",algorithms(30));self.assertNotIn("Sep-CMA-ES",algorithms(30))
        for d in (300,600,1000):
            self.assertIn("Sep-CMA-ES",algorithms(d));self.assertNotIn("CMA-ES",algorithms(d))

    def test_exact_cap_includes_final_candidate(self):
        f=StrictObjective(lambda x:float(x[0]),3,5)
        f([9]);f([5])
        with self.assertRaises(BudgetComplete):f([1])
        with self.assertRaises(BudgetComplete):f([-100])
        self.assertEqual((f.count,f.best),(3,1))
        self.assertEqual(f.trace[-1],[3,1])
        np.testing.assert_array_equal(f.best_position,[1])

    def test_all_algorithms_use_exact_non_generation_multiple_budget(self):
        cfg=dict(instance_seed=42,trace_points=20)
        with tempfile.TemporaryDirectory() as folder:
            for algo in DEFAULT_ALGORITHMS:
                j=dict(dimension=6,function="f1_sphere",algorithm=algo,seed=0,population=5,budget=17)
                r=trial(j,cfg,folder,"test")
                self.assertEqual(r["status"],"budget_reached",(algo,r["error"]))
                self.assertEqual(r["evaluations"],17)
                self.assertEqual(read_result(result_path(folder,j))["trace"][-1][0],17)

    def test_cma_negative_functions_do_not_stop_on_target(self):
        with tempfile.TemporaryDirectory() as folder:
            for algo in ("CMA-ES","Sep-CMA-ES"):
                j=dict(dimension=6,function="f13_michalewicz",algorithm=algo,seed=0,population=5,budget=31)
                r=trial(j,dict(instance_seed=42,trace_points=20),folder,"test")
                self.assertEqual((r["status"],r["evaluations"]),("budget_reached",31))

    def test_duplicate_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            with shard_lease(p,0):
                with self.assertRaises(RuntimeError):
                    with shard_lease(p,0):pass
            self.assertFalse((p/".shard-0.lock").exists())

    def test_nonfinite_remains_failure_not_success(self):
        f=StrictObjective(lambda x:np.nan,1)
        with self.assertRaises(FloatingPointError):f([0])
