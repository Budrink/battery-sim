import importlib.util
import math
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fit',ROOT/'scripts/fit_upv_kinetics.py')
fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)

class KineticSearchContract(unittest.TestCase):
    def test_trial_preserves_current_phase_and_integrator_configuration(self):
        from check_upv_phase_mixing import CASES
        extensions=copy.deepcopy(CASES['combined'])
        controls=dict(coupled_gas_films=True,coupled_transport_interval_s=.01)
        scene,_=fit.trial_scene(fit.model.DEFAULT,1200.,extensions,controls)
        plain,_=fit.model.build('no-cooling',duration=1200.,gas_extensions=extensions)
        self.assertEqual(scene['cells'],plain['cells'])
        self.assertEqual(scene['interfaces'],plain['interfaces'])
        self.assertEqual(scene['gas_network']['mixing_flow_fraction'],.25)
        self.assertTrue(scene['gas_network']['coupled_gas_films'])
        self.assertEqual(scene['gas_network']['coupled_transport_interval_s'],.01)
        self.assertEqual(scene['simulation']['minimum_dt_s'],.005)
        self.assertEqual(scene['metadata']['gas_extensions'],extensions)
        self.assertEqual(extensions,CASES['combined'])
        for bad in ({'dt_floor':1.},{'coupled_gas_films':1},{'coupled_transport_interval_s':0.},{'minimum_dt_s':.004}):
            with self.assertRaises(ValueError):fit.trial_scene(fit.model.DEFAULT,1200.,extensions,bad)

    def proposal(self):
        return dict(k1_75=3e-5,ea1=160000.,k2_75=fit.rate_at(1e-7,220000.,298.15,348.15),ea2=220000.)

    def test_only_four_coefficients_change_and_anchors_match(self):
        base=dict(fit.model.DEFAULT);prior=base.copy();q=self.proposal();p=fit.parameters(base,q)
        self.assertEqual(base,prior)
        self.assertTrue(all(p[k]==v for k,v in base.items()if k not in fit.FREE))
        self.assertAlmostEqual(fit.rate_at(p['r1_rate'],p['r1_activation'],400.,348.15),q['k1_75'],places=13)
        self.assertAlmostEqual(fit.rate_at(p['r2_rate'],p['r2_activation'],450.,348.15),q['k2_75'],places=13)

    def test_cold_reagent_stock_is_retained(self):
        p=fit.parameters(dict(fit.model.DEFAULT),self.proposal())
        k=fit.rate_at(p['r2_rate'],p['r2_activation'],450.,298.15)
        self.assertLessEqual(-math.expm1(-k*3600),.00036)

    def test_previous_fast_cold_reaction_is_not_a_valid_candidate(self):
        q=self.proposal();q['ea2']=110000.;q['k2_75']=.5264280613678021
        with self.assertRaises(ValueError):fit.parameters(dict(fit.model.DEFAULT),q)

    def test_bad_parameters_are_rejected(self):
        for key,value in [('k1_75',-1.),('ea1',float('nan')),('k2_75',0.),('ea2',-1.)]:
            q=self.proposal();q[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):fit.parameters(dict(fit.model.DEFAULT),q)

    def test_electrical_parameters_require_explicit_opt_in(self):
        base=dict(fit.model.DEFAULT);q=self.proposal();q['resistance_ohm']=.018
        with self.assertRaises(ValueError):fit.parameters(base,q)
        p=fit.parameters(base,q,allow_electrical=True)
        self.assertEqual(p['resistance_ohm'],.018)
        self.assertTrue(all(p[k]==v for k,v in base.items()if k not in fit.FREE|fit.ELECTRICAL))
        self.assertEqual(p['product_conductivity_ratio'],base['product_conductivity_ratio'])

    def test_electrical_overrides_keep_declared_bounds(self):
        for key,value in [('resistance_ohm',.1),('resistance_temperature',float('nan')),('resistance_charge',-1.)]:
            q=self.proposal();q[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                fit.parameters(dict(fit.model.DEFAULT),q,allow_electrical=True)

    def test_charging_branch_is_separate_and_opt_in(self):
        base=dict(fit.model.DEFAULT);q=self.proposal();q['charge_formation_gain']=5.
        with self.assertRaises(ValueError):fit.parameters(base,q,allow_electrical=True)
        p=fit.parameters(base,q,allow_electrical=True,allow_charge_formation=True)
        self.assertEqual(p['charge_formation_gain'],5.)
        self.assertTrue(all(p[k]==v for k,v in base.items()if k not in fit.FREE|fit.CHARGE_FORMATION))
        q['charge_formation_gain']=-1.
        with self.assertRaises(ValueError):fit.parameters(base,q,allow_charge_formation=True)

    def test_explicit_migration_does_not_reset_other_parameters(self):
        from migrate_upv_chemistry_profile import migrate
        old=dict(fit.model.DEFAULT);gain=old.pop('charge_formation_gain')
        old['overcharge_gain']=gain;old['precursor_voltage']=0.;old['iss_h']=1234.
        prior=old.copy();p=migrate(old)
        self.assertEqual(old,prior)
        self.assertEqual(p['charge_formation_gain'],gain)
        self.assertEqual(p['iss_h'],1234.)
        self.assertEqual(set(p),set(fit.model.DEFAULT))
        old['precursor_voltage']=.1
        with self.assertRaises(ValueError):migrate(old)
        with self.assertRaises(ValueError):fit.model.build('no-cooling',{'overcharge_gain':1.})

    def test_fast_hot_r1_with_cold_stability_is_allowed(self):
        q=self.proposal();q.update(k1_75=.1,ea1=240000.)
        p=fit.parameters(dict(fit.model.DEFAULT),q)
        self.assertLess(fit.rate_at(p['r1_rate'],p['r1_activation'],400.,298.15),1e-7)
        self.assertAlmostEqual(fit.rate_at(p['r1_rate'],p['r1_activation'],400.,348.15),.1)

    def test_fast_cold_r1_is_rejected(self):
        q=self.proposal();q.update(k1_75=.1,ea1=60000.)
        with self.assertRaisesRegex(ValueError,'R1 degrades'):fit.parameters(dict(fit.model.DEFAULT),q)

    def test_destructive_parameters_are_explicit_and_heat_contacts_remain_frozen(self):
        base=dict(fit.model.DEFAULT);q=self.proposal();q['shell_strength_pa']=2e6
        with self.assertRaisesRegex(ValueError,'Destruction fitting'):fit.parameters(base,q)
        p=fit.parameters(base,q,allow_destruction=True)
        self.assertEqual(p['shell_strength_pa'],2e6)
        self.assertTrue(all(p[k]==v for k,v in base.items()if k not in fit.FREE|fit.DESTRUCTION))
        q['opening_ratte']=.1
        with self.assertRaisesRegex(ValueError,'Unknown proposal'):fit.parameters(base,q,allow_destruction=True)

    def test_slower_positive_hot_rates_are_not_excluded_by_old_search_box(self):
        q=self.proposal();q.update(k1_75=1e-9,k2_75=1e-7)
        p=fit.parameters(dict(fit.model.DEFAULT),q)
        self.assertGreater(p['r1_rate'],0.)
        self.assertGreater(p['r2_rate'],0.)

    def test_explosion_objective_requires_completion_and_scores_post_event(self):
        times=[0.,50.,99.,100.,101.,110.,130.,200.,400.]
        run=dict(time_s=times,temperature_k={k:[300.]*len(times)for k in fit.dataio.CHANNELS},
                 gas={'chamber':{'pressure_pa':[1e5]*len(times)}})
        obs={k:np.full(len(times),300.)for k in fit.dataio.CHANNELS};obs['pressure_bar']=np.ones(len(times))
        meta=dict(objective_mode='explosion',fit_end_s=400.,event=dict(time_s=100.,previous_s=99.),data={})
        with patch.object(fit.dataio,'observed',return_value=obs),patch.object(fit.cal,'metrics',return_value={}):
            self.assertEqual(fit.score(run,meta)['objective'],0.)
            later=copy.deepcopy(run);later['temperature_k']['101'][-1]+=100.
            self.assertGreater(fit.score(later,meta)['objective'],0.)
            pressure=copy.deepcopy(run);pressure['gas']['chamber']['pressure_pa'][5]+=1e5
            self.assertGreater(fit.score(pressure,meta)['objective'],0.)
            self.assertIsNone(fit.score(dict(run,partial=True),meta))
            self.assertIsNone(fit.score(dict(run,time_s=times[:-1]),meta))

if __name__=='__main__':unittest.main()
