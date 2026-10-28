from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_fsri30_sequence import physical_error, promotable, assess_run, search_levels, read_observed_trial, BASE, trial, delivery_signal
from fsri30_data import sha
from fsri30_sequence_model import initial, suggest, sequence_rank


class ControllerTests(unittest.TestCase):
    def test_delivery_signal_excludes_source_and_future(self):
        scene={'metadata':{'heated_component':'B10'}}
        run=dict(time_s=[0.,2135.,2290.],cell_mean_temperature_k={
            'B10':[300.,2000.,2200.],**{f'B{i}':[300.,350.+i,900.] for i in range(5)}})
        signal=delivery_signal(run,scene)
        self.assertEqual(signal['top4_neighbor_mean_k'],352.5)
        self.assertEqual(signal['maximum_neighbor_k'],354.)
        self.assertFalse(signal['qualifies_cascade'])
        run['time_s']=[0.,2100.]
        self.assertIsNone(delivery_signal(run,scene))

    def test_wave_match_outranks_hotter_unreacted_neighbors(self):
        p=initial(json.loads(BASE.read_text()))
        a=dict(parameters=p,assessment=dict(accepted=False,primary_offset_s=0.,pulse_match={'matched':1}),
               delivery=dict(top4_neighbor_mean_k=700.))
        b=dict(parameters=p,assessment=dict(accepted=False,primary_offset_s=8.,pulse_match={'matched':2}),
               delivery=dict(top4_neighbor_mean_k=450.))
        self.assertGreater(sequence_rank(b),sequence_rank(a))
        self.assertFalse(promotable(a));self.assertFalse(promotable(b))

    def test_recenter_changed_gas_contact_before_next_geometry(self):
        base=json.loads(BASE.read_text());p=initial(base);p['r2_rate_scale']=.023
        seed=dict(parameters=p,trial='aligned-original',assessment=dict(primary_offset_s=0.))
        changed=dict(parameters=dict(p,gas_can_contact_scale=2.),trial='changed-contact',
                     status='pruned',assessment=dict(primary_offset_s=40.))
        next_p,why=suggest([seed,changed],base,anchor=p)
        self.assertEqual(why['method'],'recenter-primary-after-gas-change')
        self.assertEqual(next_p['gas_can_contact_scale'],2.)
        self.assertGreater(next_p['r2_rate_scale'],p['r2_rate_scale'])
        for k in p:
            if k not in ('gas_can_contact_scale','r2_rate_scale'):self.assertEqual(next_p[k],p[k])

    def test_new_native_has_its_own_hash_and_explicit_executable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);probe=root/'probe';probe.write_bytes(b'new core test fixture')
            run=dict(time_s=[0.],wall_s=1.)
            with patch('calibrate_fsri30_sequence.candidate',return_value={}), \
                 patch('calibrate_fsri30_sequence.native',return_value=run) as native, \
                 patch('calibrate_fsri30_sequence.assess_run',return_value=dict(accepted=None,last_observed_s=0.)), \
                 patch('calibrate_fsri30_sequence.physical_error',return_value=None):
                row=trial(root/'fresh',{}, {},20,{}, {},lambda row:None,probe=probe)
            self.assertEqual(row['native_sha256'],sha(probe))
            self.assertEqual(native.call_args.args[2]['restart_contract'],sha(probe))
            self.assertEqual(native.call_args.kwargs['probe'],probe)
            self.assertFalse(row['calibrated']);self.assertFalse(row['validation_passed'])

    def test_explicit_fine_grid_search_never_runs_coarse_levels(self):
        self.assertEqual(search_levels(20,Path('/explicit/seed')),[20])
        self.assertEqual(search_levels(40,Path('/explicit/seed')),[40,20])
        self.assertEqual(search_levels(80,None),[80,40,20])
        with self.assertRaises(ValueError):search_levels(20,None)

    def test_fine_search_does_not_apply_the_coarse_time_correction(self):
        base=json.loads(BASE.read_text());anchor=initial(base)
        anchor['r2_rate_scale']=.04919319392060936
        row=dict(parameters=anchor,trial='observed-20mm',assessment=dict(
            primary_offset_s=-151.,last_observed_s=1630.,constraint=.51))
        proposed,_=suggest([row],base,anchor=anchor)
        self.assertAlmostEqual(proposed['r2_rate_scale'],.01988081664251654)
        for key in anchor:
            if key!='r2_rate_scale':self.assertEqual(proposed[key],anchor[key])

    def test_expired_first_arrival_brackets_late_source_without_fabricating_offset(self):
        base=json.loads(BASE.read_text());anchor=initial(base);anchor['r2_rate_scale']=.1
        row=dict(parameters=anchor,trial='censored',status='pruned',assessment=dict(
            primary_offset_s=None,last_observed_s=1905.,arrival_match=dict(matched=0,missing=[{'reference_time_s':1765.}])))
        proposed,why=suggest([row],base,anchor=anchor)
        self.assertEqual(why['method'],'censored-primary-rate-bracket')
        self.assertGreater(proposed['r2_rate_scale'],anchor['r2_rate_scale'])
        self.assertIsNone(row['assessment']['primary_offset_s'])

    def test_promotion_requires_complete_real_nonfailed_mesh_run(self):
        good=dict(status='complete',assessment=dict(accepted=True,phase_window_complete=True))
        self.assertTrue(promotable(good))
        for change in (dict(status='pruned'),dict(physical_error='balance'),
                       dict(status='numerical-failed'),dict(assessment=dict(accepted=None)),
                       dict(assessment=dict(accepted=True,phase_window_complete=False))):
            self.assertFalse(promotable(dict(good,**change)))

    def test_all_ledgers_and_step_floor_are_mandatory(self):
        run=dict(max_thermal_residual_j=1e-9,max_chemical_residual_j=1e-9,
                 max_mass_residual_kg=1e-15,max_oxidizer_residual_kg=1e-15,
                 solver_profile=dict(minimum_regular_step_s=.01))
        self.assertIsNone(physical_error(run))
        for key in ('max_thermal_residual_j','max_chemical_residual_j',
                    'max_mass_residual_kg','max_oxidizer_residual_kg'):
            broken=deepcopy(run);broken[key]=float('nan')
            self.assertIsNotNone(physical_error(broken))
            broken.pop(key);self.assertIsNotNone(physical_error(broken))
        run['solver_profile']['minimum_regular_step_s']=.001
        self.assertIsNotNone(physical_error(run))

    def test_initial_observation_does_not_extrapolate_waves(self):
        self.assertIsNone(assess_run(dict(time_s=[0.]),{})['accepted'])

    @unittest.skipUnless(BASE.exists(),'Retained FSRI fixture not installed')
    def test_refinement_carries_learned_parameters_and_no_old_scores(self):
        base=json.loads(BASE.read_text());anchor=initial(base)
        anchor.update(r2_rate_scale=.04,gas_can_contact_scale=3.)
        p,how=suggest([],base,anchor=anchor)
        self.assertEqual(p,anchor)
        self.assertFalse(how['scores_transferred'])
        row=dict(parameters=p,trial='coarse-anchor',assessment=dict(primary_offset_s=-200.,last_observed_s=1600.))
        next_p,how=suggest([row],base,anchor=anchor)
        self.assertLess(next_p['r2_rate_scale'],p['r2_rate_scale'])
        self.assertEqual(next_p['gas_can_contact_scale'],3.)
        self.assertEqual(anchor['r2_rate_scale'],.04)


if __name__=='__main__':unittest.main()
