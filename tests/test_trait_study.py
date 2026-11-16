import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from experiments.trait_plan import ROOT,load_config,build_plan
from experiments.trait_batch import BatchFunction,BatchBudget,BatchedTraitPSO
from experiments.final_benchmark import suite,read_result
from experiments.trait_study import trial,result_path
from hppso.algorithms.trait_study import TraitPSO
from hppso.algorithms.pso import PSO


class TraitTests(unittest.TestCase):
    def config(self):return load_config(ROOT/"experiments/configs/trait_study.json")

    def test_oat_protocol_dedup_and_all_trait_ablations(self):
        cfg=self.config();p=build_plan(cfg)
        self.assertEqual(cfg["grid_mode"],"one_at_a_time")
        self.assertEqual(cfg["functions"],["f1_sphere","f23_rotated_rastrigin","f25_composition"])
        self.assertEqual(len(p["variants"]),59)
        self.assertEqual(len(p["factorial"]),16)
        self.assertEqual(len(p["groups"]["hp_ablation"]),6)
        for e in p["groups"]["openness_grid"]:
            if e["settings"].get("theta")==1:self.assertEqual(e["variant"],p["baseline"])

    def test_all_nine_vectorized_formulas_match_scalar_objectives(self):
        cfg=self.config();problems=suite(100,42);rng=np.random.default_rng(8)
        functions=json.loads((ROOT/"experiments/configs/openness_ablation_final9.json").read_text())["functions"]
        for name in functions:
            p=problems[name];lo,hi=p["bounds"]
            x=rng.uniform(lo,hi,(4,7,100));batch=BatchFunction(p)
            expected=np.array([[p["function"](row) for row in block] for block in x])
            np.testing.assert_allclose(batch(x),expected,rtol=5e-13,atol=1e-10,err_msg=name)
            optimum=np.array(p["shift_vector"])+(1 if name in ("f4_rosenbrock","f10_levy") else 0)
            self.assertLess(abs(float(batch(optimum))),1e-10)
        p=problems["f25_composition"];x=p["function"].shift+p["function"].center_offsets
        np.testing.assert_allclose(BatchFunction(p)(x),[0,100,200],atol=1e-12)

    def test_batched_paths_match_independent_scalar_arms_including_resets(self):
        cfg=self.config();cfg["stagnation_threshold"]=2
        p=build_plan(cfg);specs=[v["spec"] for v in p["variants"].values()]
        bounds=(np.full(4,-10.),np.full(4,10.))
        objective=lambda x:float(np.sum(x*x))
        tracker=BatchBudget(lambda x:np.sum(x*x,axis=-1),1000,len(specs),4,10)
        batch=BatchedTraitPSO(tracker,bounds,5,7,specs,cfg)
        singles=[TraitPSO(objective,bounds,5,7,s,cfg) for s in specs]
        self.assertTrue(np.all(batch.velocities == 0))
        for opt in singles:
            self.assertTrue(np.all(opt.velocities == 0))
        for _ in range(8):
            batch.step()
            for i,opt in enumerate(singles):
                opt.step()
                for name in ("positions","velocities","pbests","pbest_scores","global_best","global_score"):
                    np.testing.assert_allclose(getattr(batch,name)[i],getattr(opt,name),rtol=1e-12,atol=1e-12,
                                               err_msg=str((i,name,specs[i])))
        self.assertGreater(int(batch.reset_events.sum()),0)
        full=p["no_reinitialization"];i=list(p["variants"]).index(full)
        self.assertEqual(batch.reset_events[i],0)

    def test_pso_path_matches_repository_pso_and_exact_partial_budget(self):
        cfg=self.config();p=build_plan(cfg);spec=p["variants"][p["baseline"]]["spec"]
        bounds=(np.full(4,-10.),np.full(4,10.));objective=lambda x:float(np.sum(x*x))
        np.random.seed(13)
        old=PSO(objective,list(zip(*bounds)),5,7,mutation_rate=0)
        position,score,_=old.optimize()
        tracker=BatchBudget(lambda x:np.sum(x*x,axis=-1),45,1,4,20)
        opt=BatchedTraitPSO(tracker,bounds,5,13,[spec],cfg)
        for _ in range(7):opt.step()
        np.testing.assert_allclose(opt.global_best[0],position,rtol=1e-12,atol=1e-12)
        self.assertAlmostEqual(opt.global_score[0],score)
        tracker=BatchBudget(lambda x:np.sum(x*x,axis=-1),23,1,4,20)
        opt=BatchedTraitPSO(tracker,bounds,5,13,[spec],cfg);opt.optimize()
        self.assertEqual(tracker.count,23)
        self.assertEqual(tracker.trace[-1][0],23)
        self.assertEqual(tracker.trace[-1][1][0],tracker.best[0])
        with self.assertRaises(RuntimeError):tracker.evaluate(opt.positions)

    def test_one_seed_function_saves_every_path_separately_in_atomic_batch(self):
        cfg=self.config();cfg.update(dimension=4,population=5,budget=23,trace_points=10)
        p=build_plan(cfg);job=dict(function="f25_composition",seed=0)
        with tempfile.TemporaryDirectory() as folder:
            r=trial(job,cfg,p,folder,"test")
            self.assertEqual(r["status"],"budget_reached",r["error"])
            saved=read_result(result_path(folder,job))
            self.assertEqual(len(saved["variant_ids"]),59)
            self.assertEqual(saved["logical_evaluations"],59*23)
            self.assertEqual(np.shape(saved["best_positions"]),(59,4))
            self.assertEqual(saved["trace_evaluations"][-1],23)
            np.testing.assert_array_equal(np.array(saved["trace_best"])[-1],saved["best_scores"])

    def test_factorial_interactions_detect_synergy_without_cross_trait_leakage(self):
        from experiments.trait_interactions import factorial_contrasts,normalized_response
        masks=np.arange(16);a=(masks&1)>0;b=(masks&2)>0
        response=np.tile(.5-.1*a-.2*b-.05*(a&b),(3,1))
        effects=factorial_contrasts(response)
        np.testing.assert_allclose(effects[:,2],-.05,atol=1e-14) # subset bits 1|2
        for subset in (5,6,7,9,10,11,12,13,14,15):
            np.testing.assert_allclose(effects[:,subset-1],0,atol=1e-14)
        np.testing.assert_allclose(normalized_response(response*100,np.full_like(response,100)),response)
        np.testing.assert_array_equal(normalized_response(np.zeros((2,16)),np.zeros((2,16))),0)

    def test_full_hp_path_matches_production_hppso_update_given_the_same_draws(self):
        from hppso.algorithms.hppso import HPPSO
        cfg=self.config();cfg.update(alpha_mut=1.,stagnation_threshold=100)
        p=build_plan(cfg);spec=p["variants"][p["factorial"]["15"]]["spec"]
        bounds=(np.full(4,-10.),np.full(4,10.));seed=7;n=5
        objective=lambda x:float(np.sum(x*x))
        tracker=BatchBudget(lambda x:np.sum(x*x,axis=-1),1000,1,4,10)
        batch=BatchedTraitPSO(tracker,bounds,n,seed,[spec],cfg)
        batch.traits[:,:,3]=1
        old=HPPSO(objective,n_pop=n,dimensions=4,max_it=1,bounds=bounds,
                  alpha_curio=1,stagnation_threshold=100)
        old.X=batch.positions[0].copy();old.V=batch.velocities[0].copy()
        old.P=old.X.copy();old.fitness=batch.scores[0].copy();old.p_best_fitness=old.fitness.copy()
        old.g_best=batch.global_best[0].copy();old.g_best_fitness=float(batch.global_score[0])
        old.personality=batch.traits[0].copy()
        r=np.random.RandomState(seed);r.uniform(*bounds,(n,4))
        r1,r2=r.rand(n,1),r.rand(n,1)
        uniforms=np.random.default_rng(np.random.SeedSequence([seed,904])).random((n,4))
        noise=np.random.default_rng(np.random.SeedSequence([seed,905])).standard_normal((n,4))
        peers=np.random.default_rng(np.random.SeedSequence([seed,902])).integers(0,n-1,n)
        with patch('numpy.random.rand',side_effect=[r1,r2,uniforms]), \
             patch('numpy.random.randint',return_value=peers.copy()), \
             patch('numpy.random.normal',return_value=noise*cfg['eta']*(bounds[1]-bounds[0])):
            old.optimize()
        batch.step()
        np.testing.assert_allclose(batch.positions[0],old.X,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(batch.velocities[0],old.V,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(batch.global_best[0],old.g_best,rtol=1e-12,atol=1e-12)


    def test_extension_selection_and_all_supported_vectorized_functions(self):
        from experiments.trait_functions import trait_suite,SUPPORTED_FUNCTIONS
        cfg=load_config(ROOT/'experiments/configs/trait_study_added10.json')
        selection=json.loads((ROOT/'experiments/configs/trait_extension_selection.json').read_text())
        self.assertEqual(cfg['functions'],selection['shifted']+selection['rotated']+selection['composition'])
        self.assertEqual([len(selection[k]) for k in ('shifted','rotated','composition')],[5,3,2])
        self.assertFalse(set(cfg['functions'])&set(selection['original_functions']))
        problems=trait_suite(100,42);rng=np.random.default_rng(917)
        for name in SUPPORTED_FUNCTIONS:
            problem=problems[name];lo,hi=problem['bounds']
            x=rng.uniform(lo,hi,(3,4,100))
            scalar=np.array([[problem['function'](row) for row in block] for block in x])
            np.testing.assert_allclose(BatchFunction(problem)(x),scalar,rtol=5e-12,atol=1e-9,err_msg=name)
            if name == 'f13_michalewicz':
                # Its minimum is negative and the minimizing coordinates differ.
                continue
            offset = {
                'f4_rosenbrock': 1.0,
                'f10_levy': 1.0,
                'f11_schwefel_2_26': 420.9687462275036,
                'f19_penalized_1': -1.0,
            }.get(name, 0.0)
            optimum = np.asarray(problem['shift_vector']) + offset
            self.assertTrue(np.all((lo <= optimum) & (optimum <= hi)))
            # The published Schwefel offset is rounded to four decimal places.
            tolerance = optimum.size * 0.00005 if name == 'f11_schwefel_2_26' else 1e-9
            self.assertLess(abs(float(BatchFunction(problem)(optimum))), tolerance, name)
        for name in ('f30_composition_multimodal','f31_composition_ridges'):
            obj=problems[name]['function'];centers=obj.shift+obj.center_offsets
            np.testing.assert_allclose(BatchFunction(problems[name])(centers),[0,100,200],rtol=1e-7,atol=1e-9)
            self.assertGreater(np.ptp(obj.shift),0)

    def test_extension_all_paths_keep_exact_partial_budget(self):
        cfg=load_config(ROOT/'experiments/configs/trait_study_added10.json')
        cfg.update(dimension=4,population=5,budget=23,trace_points=10)
        protocol=build_plan(cfg)
        with tempfile.TemporaryDirectory() as folder:
            for name in cfg['functions']:
                r=trial(dict(function=name,seed=0),cfg,protocol,folder,'test')
                self.assertEqual(r['status'],'budget_reached',r['error'])
                self.assertEqual(r['evaluations_per_path'],23)

    def test_combining_preserves_raw_fingerprints_and_rejects_different_budgets(self):
        from experiments.combine_trait_studies import combine
        from experiments.trait_study import fingerprint
        from experiments.trait_functions import trait_suite
        from hppso.benchmarks.classical import problem_definitions
        with tempfile.TemporaryDirectory() as temp:
            sources=[];plans=[]
            for i,name in enumerate(('f14_alpine_1','f3_zakharov')):
                cfg=self.config();cfg.update(functions=[name],seeds=[0],dimension=4,population=5,budget=23,trace_points=10)
                protocol=build_plan(cfg);digest=fingerprint(cfg)
                source=Path(temp)/str(i);source.mkdir();sources.append(source)
                plan=dict(config=cfg,protocol=protocol,fingerprint=digest);plans.append(plan)
                (source/'plan.json').write_text(json.dumps(plan))
                (source/'problem_definitions.json').write_text(json.dumps(problem_definitions([trait_suite(4,42)[name]])))
                trial(dict(function=name,seed=0),cfg,protocol,source,digest)
            dest=Path(temp)/'combined';combine(sources,dest)
            joined=json.loads((dest/'plan.json').read_text())
            self.assertEqual(len(joined['analysis_sources']),2)
            import hashlib
            self.assertEqual(joined['fingerprint'],hashlib.sha256(json.dumps(joined['analysis_sources'],sort_keys=True).encode()).hexdigest())
            for source,plan in zip(sources,plans):
                job=dict(function=plan['config']['functions'][0],seed=0)
                self.assertEqual(result_path(source,job).read_bytes(),result_path(dest,job).read_bytes())
                self.assertEqual(read_result(result_path(dest,job))['fingerprint'],plan['fingerprint'])
            plans[1]['config']['budget']=24
            (sources[1]/'plan.json').write_text(json.dumps(plans[1]))
            with self.assertRaisesRegex(ValueError,'Scientific settings'):
                combine(sources,Path(temp)/'incompatible')


if __name__=="__main__":unittest.main()
