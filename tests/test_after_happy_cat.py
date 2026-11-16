import unittest
from experiments.after_happy_cat import finished_happy_cat


class TransitionTests(unittest.TestCase):
    def test_only_completed_full_1000d_happy_cat_rows_count(self):
        base=dict(phase="full",dimension=1000,function="f15_happy_cat",algorithm="HPPSO",seed=0,status="budget_exhausted")
        rows=[base,dict(base,phase="smoke"),dict(base,dimension=30),
              dict(base,function="f16_hgbat"),dict(base,seed=1)]
        self.assertEqual(set(finished_happy_cat(rows)),{("HPPSO",0),("HPPSO",1)})

    def test_repeated_journal_rows_do_not_count_twice(self):
        row=dict(phase="full",dimension=1000,function="f15_happy_cat",algorithm="SHADE",seed=0)
        self.assertEqual(len(finished_happy_cat([row,row])),1)
