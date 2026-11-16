import json
import sys
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from hppso.benchmarks.additional import AdditionalProblem, NAMES, build_additional_suite
from hppso.benchmarks.classical import problem_definitions
from hppso.benchmarks.suite import BENCHMARK_NAMES, build_benchmark_suite


class AdditionalTests(unittest.TestCase):
    def test_no_duplicate_or_scaling_named_variants(self):
        np.random.seed(72)
        problems=build_benchmark_suite(30,michalewicz_dimension=None)
        names=[p["name"] for p in problems]
        self.assertEqual(tuple(names),BENCHMARK_NAMES)
        self.assertEqual(len(set(names)),30)
        self.assertTrue(all(not n.startswith("scaling_") for n in names))
        self.assertFalse(any("sphere" in n or "rosenbrock" in n for n in NAMES))
        manifest=json.loads(json.dumps(problem_definitions(problems)))
        self.assertTrue(all(manifest[n]["version"]=="additional-v3" for n in NAMES))
        self.assertIsNone(manifest[NAMES[0]]["rotation"])

    def test_stable_formulas_match_direct_equations(self):
        z=np.array([.2,-.5,.7,1.2,-1.1,.3])
        r2=z*z+np.roll(z,-1)**2
        ref=np.sum(.5+(np.sin(np.sqrt(r2))**2-.5)/(1+.001*r2)**2)
        self.assertAlmostEqual(AdditionalProblem.expanded_schaffer(z),ref,places=12)
        ref=-20*np.exp(-.2*np.sqrt(np.mean(z*z)))-np.exp(np.mean(np.cos(2*np.pi*z)))+20+np.e
        self.assertAlmostEqual(AdditionalProblem.ackley(z),ref,places=12)
        self.assertEqual(AdditionalProblem.ackley(np.zeros(6)),0)
        self.assertEqual(AdditionalProblem.expanded_schaffer(np.zeros(6)),0)

    def test_composition_is_distinct_center_weighted_and_globally_nonnegative(self):
        p=build_additional_suite(6,names=["f25_composition"])[0]
        self.assertEqual(p(p.shift),0)
        for i,bias in enumerate([0,100,200]):
            self.assertAlmostEqual(p(p.shift+p.center_offsets[i]),bias,places=8)
        rng=np.random.default_rng(5)
        lo,hi=p.bounds
        for x in rng.uniform(lo,hi,(30,6)):
            self.assertGreaterEqual(p(x),0)
            z=x-p.shift
            delta=z-p.center_offsets
            d=np.sum(delta*delta,axis=1)
            w=np.exp(-d/12)/np.sqrt(d)
            w/=w.sum()
            values=[p.ellipsoid(p.rotate(delta[0],0)),p.rastrigin(p.rotate(delta[1],1))+100,p.ackley(p.rotate(delta[2],2))+200]
            self.assertAlmostEqual(p(x),w@values,places=8)

    def test_saved_instance_reconstructs_evaluation(self):
        for p in build_additional_suite(30):
            d=json.loads(json.dumps(p.definition()))
            n=len(d["permutations"])
            restored=AdditionalProblem(d["name"],d["dimension"],np.array(d["shift_vector"]),
                np.array(d["permutations"],int).reshape(n,30),np.array(d["signs"]).reshape(n,30),
                np.array(d["center_offsets"]).reshape(-1,30))
            x=p.shift+np.linspace(-1,1,30)
            self.assertEqual(p(x),restored(x))

    def test_feasible_optima_and_finite_values_at_requested_dimensions(self):
        for dim in (30,250,500,750,1000):
            for p in build_additional_suite(dim):
                lo,hi=p.bounds
                self.assertTrue(np.all((lo<p.shift)&(p.shift<hi)))
                self.assertEqual(len(np.unique(p.shift)),dim)
                self.assertEqual(p(p.shift),0)
                with np.errstate(over="raise",invalid="raise",divide="raise"):
                    for x in (lo,hi,p.shift+np.linspace(-1,1,dim)):
                        self.assertTrue(np.isfinite(p(x)))

    def test_invalid_names_dimensions_and_shifts(self):
        for names in (["sphere"],["composition"],[NAMES[0],NAMES[0]]):
            with self.assertRaises(ValueError):build_additional_suite(30,names=names)
        with self.assertRaises(ValueError):build_additional_suite(1)
        with self.assertRaises(ValueError):build_additional_suite(30,fixed_shifts={NAMES[0]:[np.nan]*30})


    def test_all_additions_have_cross_coordinate_interactions(self):
        for p in build_additional_suite(6):
            a=np.zeros(6); a[0]=.6
            b=np.zeros(6); b[1]=-.4
            interaction=p(p.shift+a+b)-p(p.shift+a)-p(p.shift+b)+p(p.shift)
            self.assertGreater(abs(interaction),1e-8,p.name)
            for k in range(len(p.permutations)):
                z=np.arange(6,dtype=float)-2
                self.assertAlmostEqual(p.rotate(z,k)@p.rotate(z,k),z@z,places=10)

    def test_explicit_shifts_preserve_seeded_rotation(self):
        first=build_additional_suite(30)
        shifts={p.name:p.shift for p in first}
        rebuilt=build_additional_suite(30,fixed_shifts=shifts)
        self.assertEqual([p.definition() for p in first],[p.definition() for p in rebuilt])


    def test_new_base_functions_match_reference_equations(self):
        z=np.array([.2,-.5,.7,1.2])
        g=np.sum(z*z)/4000+1-np.prod(np.cos(z/np.sqrt(np.arange(1,5))))
        self.assertAlmostEqual(AdditionalProblem.griewank(z),g,places=12)
        self.assertAlmostEqual(AdditionalProblem.discus(z),1e6*z[0]**2+sum(z[1:]**2),places=9)
        self.assertAlmostEqual(AdditionalProblem.schwefel_1_2(z),sum(sum(z[:i+1])**2 for i in range(4)),places=12)
        ref=sum((z[i]**2+z[i+1]**2)**.25*(1+np.sin(50*(z[i]**2+z[i+1]**2)**.1)**2) for i in range(3))**2/9
        self.assertAlmostEqual(AdditionalProblem.schaffer_f7(z),ref,places=10)

    def test_second_composition_has_distinct_components_and_widths(self):
        p=build_additional_suite(30,names=["f30_composition_multimodal"])[0]
        d=p.definition()
        self.assertEqual(d["composition_components"],["rastrigin","griewank","expanded_schaffer_f6"])
        self.assertEqual(d["composition_widths"],[1.,2.,3.])
        for i,bias in enumerate([0,100,200]):
            self.assertAlmostEqual(p(p.shift+p.center_offsets[i]),bias,places=8)
