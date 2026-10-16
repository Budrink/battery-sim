"""Opt-in GPU checks: BATTERY_SIM_EXPERIMENT_PROBE=/path/to/binary."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from calibrate_upv import CONTACTS,Worker,verify_fixed_properties
    from generate_upv import ROOT,generate
finally:
    sys.path.pop(0)


@unittest.skipUnless(os.environ.get('BATTERY_SIM_EXPERIMENT_PROBE'),'CUDA probe not requested')
class UPVNativeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='upv-native-contract-')
        cls.scene=generate(ROOT/'data/experiments/upv','no-cooling')
        cls.scene['simulation'].update(duration_s=2.,output_interval_s=1.)
        # Synthetic cooling edge exercises the fixed-boundary rule without
        # opening any validation observations.
        edge=next(b for b in cls.scene['boundaries'] if b['id']=='ambient-Al1-z')
        edge['id']='cooling-test';edge['conductance_w_per_k']=12.
        path=Path(cls.temp.name)/'scenario.json';path.write_text(json.dumps(cls.scene))
        cls.worker=Worker(Path(os.environ['BATTERY_SIM_EXPERIMENT_PROBE']),path)

    @classmethod
    def tearDownClass(cls):
        cls.worker.close();cls.temp.cleanup()

    def test_material_groups_and_source_use_the_same_contact_law(self):
        params=dict(zip(CONTACTS,(2.,3.,5.,7.,11.)));params['ambient_scale']=1.3
        result=self.worker.run(params);verify_fixed_properties(result,self.scene)
        audit=result['applied_model'];original={e['id']:e for e in self.scene['interfaces']}
        self.assertEqual({e['group'] for e in audit['contacts']},set(CONTACTS))
        for e in audit['contacts']:
            base=original[e['id']]['law']['background_conductance_w_per_k']
            self.assertAlmostEqual(e['conductance_w_per_k'],base*params[e['group']])
        original={e['id']:e for e in self.scene['boundaries']}
        for e in audit['boundaries']:
            factor=params.get(e['group'],1.)
            self.assertAlmostEqual(e['conductance_w_per_k'],original[e['id']]['conductance_w_per_k']*factor)
            if e['id']=='source-C2-ISS':self.assertEqual(e['group'],'battery_iss_scale')
            if e['id']=='source-C2-TIM':self.assertEqual(e['group'],'battery_tim_scale')
            if e['id']=='cooling-test':self.assertEqual(e['group'],'fixed')

    def test_only_pad_capacity_changes_and_request_does_not_persist(self):
        result=self.worker.run({'iss_cp_scale':1.7,'tim_cp_scale':.6})
        verify_fixed_properties(result,self.scene)
        original={c['id']:c for c in self.scene['cells']}
        for c in result['applied_model']['objects']:
            factor={'ISS':1.7,'TIM':.6}.get(c['material'],1.)
            self.assertAlmostEqual(c['heat_capacity_j_per_k'],original[c['id']]['thermal']['heat_capacity_j_per_k']*factor)
        baseline=self.worker.run({})
        for c in baseline['applied_model']['objects']:
            self.assertEqual(c['heat_capacity_j_per_k'],original[c['id']]['thermal']['heat_capacity_j_per_k'])

    def test_invalid_and_forbidden_parameters_fail_closed(self):
        for key in ('cell_k_scale','cell_capacity_scale','iss_k_scale','tim_k_scale','typo_scale'):
            with self.assertRaisesRegex(RuntimeError,'Unsupported calibration parameter'):self.worker.run({key:1.})
        with self.assertRaisesRegex(RuntimeError,'Cannot combine'):
            self.worker.run({'contact_scale':2.,'battery_tim_scale':3.})
        with self.assertRaisesRegex(RuntimeError,'Positive finite'):
            self.worker.run({'iss_cp_scale':0.})


if __name__=='__main__':unittest.main()
