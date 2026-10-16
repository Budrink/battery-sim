"""Dynamic connectivity calibration must not turn into a bulk-property fit."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from calibrate_upv import CONTACTS,PARAMETER_SETS,Worker,transform,validate_parameters,verify_fixed_properties
    from generate_upv import ROOT,generate
    from export_upv_state_geometry import calibrated_scene,contact_group
finally:
    sys.path.pop(0)


class UPVStateGeometryTests(unittest.TestCase):
    def test_signed_response_and_fixed_materials_contract(self):
        self.assertNotIn('iss_cp_scale',PARAMETER_SETS['state_geometry'])
        self.assertNotIn('tim_cp_scale',PARAMETER_SETS['state_geometry'])
        params={'surface_linear':-3.,'surface_quadratic':2.,'battery_iss_scale':.2}
        validate_parameters(params)
        encoded=transform(list(params.values()),list(params))
        self.assertEqual(list(transform(encoded,list(params),inverse=True)),list(params.values()))
        with self.assertRaisesRegex(ValueError,'fixed'):validate_parameters(dict(params,iss_cp_scale=1.))

    def test_generated_model_enables_both_contacts_and_source_boundary(self):
        scene=generate(ROOT/'data/experiments/upv','no-cooling',connectivity='state_geometry')
        self.assertTrue(scene['metadata']['fixed_pad_cp'])
        for c in scene['cells']:
            self.assertIn('surface_connectivity',c['thermal'])
            self.assertNotIn('bulk_connectivity',c['thermal'])
            if c['role']=='battery':
                self.assertEqual(c['thermal']['heat_capacity_j_per_k'],.410*1007)
                self.assertEqual(c['thermal']['conductivity_w_per_m_k'],[.9,28.3,28.3])
        for e in scene['interfaces']:self.assertEqual(e['law']['kind'],'state_geometry')
        for b in scene['boundaries']:
            if b['id'].startswith('source-'):
                self.assertEqual(b['connectivity']['kind'],'state_geometry')
                self.assertIn('remote_surface_connectivity',b)
            else:self.assertNotIn('connectivity',b)
        audit={'applied_model':{'objects':[dict(c['thermal'],id=c['id']) for c in scene['cells']]}}
        verify_fixed_properties(audit,scene)
        next(c for c in audit['applied_model']['objects'] if c['id']=='TIM1')['heat_capacity_j_per_k']*=1.01
        with self.assertRaisesRegex(ValueError,'Fixed heat capacity'):verify_fixed_properties(audit,scene)

    def test_export_bakes_only_calibrated_properties_and_preserves_baseline(self):
        scene=generate(ROOT/'data/experiments/upv','no-cooling',connectivity='state_geometry');original=copy.deepcopy(scene)
        params=dict(zip(CONTACTS,(2.,3.,5.,7.,11.)),ambient_scale=1.3,surface_linear=-2.,surface_quadratic=3.)
        profile=dict(parameter_set='state_geometry',parameters=params,optimizer_success=True,model_sha256='test')
        out=calibrated_scene(scene,profile);self.assertEqual(scene,original)
        for before,after in zip(scene['cells'],out['cells']):
            for key in ('heat_capacity_j_per_k','conductivity_w_per_m_k'):
                self.assertEqual(before['thermal'][key],after['thermal'][key])
            self.assertEqual(after['thermal']['surface_connectivity']['linear'],[-2.]*3)
        cells={c['id']:c for c in out['cells']};ports={p['id']:p for p in out['ports']}
        import math
        for e in out['interfaces']:
            a=cells[ports[e['port_a']]['cell_id']]['material_id'];b=cells[ports[e['port_b']]['cell_id']]['material_id']
            recovered=1+9999/(1+math.exp(-e['law']['geometry_reference_logit']))
            self.assertAlmostEqual(recovered,100*params[contact_group(a,b)])
        for before,after in zip(scene['boundaries'],out['boundaries']):
            if after['id'].startswith('ambient-'):
                self.assertAlmostEqual(after['conductance_w_per_k'],before['conductance_w_per_k']*1.3)
            elif after['id'].startswith('source-'):
                material=cells[ports[after['port_id']]['cell_id']]['material_id']
                expected=params[contact_group('battery',material)]
                self.assertAlmostEqual(after['conductance_w_per_k'],before['conductance_w_per_k']*expected)
                self.assertEqual(after['remote_surface_connectivity']['quadratic'],[3.]*3)

    @unittest.skipUnless(os.environ.get('BATTERY_SIM_EXPERIMENT_PROBE'),'CUDA probe not requested')
    def test_native_dynamic_probe_local_snapshot_and_reset(self):
        scene=generate(ROOT/'data/experiments/upv','no-cooling',connectivity='state_geometry')
        scene['simulation'].update(duration_s=2.,output_interval_s=1.)
        for c in scene['cells']:c['thermal']['initial_temperature_k']=350.
        for b in scene['boundaries']:
            b['temperature_k']=400.
            b['temperature_profile']=[[0,400.],[2,400.]]
        with tempfile.TemporaryDirectory(prefix='upv-geometry-test-') as directory:
            path=Path(directory)/'scenario.json';path.write_text(json.dumps(scene))
            worker=Worker(Path(os.environ['BATTERY_SIM_EXPERIMENT_PROBE']),path)
            try:
                baseline=worker.run({});params={k:2. for k in CONTACTS}
                dynamic=worker.run(dict(params,surface_linear=-2.,surface_quadratic=3.))
                verify_fixed_properties(dynamic,scene)
                tracks=dynamic['interface_snapshot']
                self.assertTrue(any(v['conductance_w_per_k'][0]!=v['conductance_w_per_k'][-1] for v in tracks.values()))
                self.assertTrue(any(tracks[k]['conductance_w_per_k'][0]!=baseline['interface_snapshot'][k]['conductance_w_per_k'][0] for k in tracks))
                self.assertLess(abs(dynamic['energy_residual_j']),1e-7)
                for b in dynamic['applied_model']['boundaries']:
                    if b['id'].startswith('source-'):self.assertEqual(b['law'],'state_geometry')
                for key in ('iss_cp_scale','tim_cp_scale','cell_k_scale','cell_capacity_scale'):
                    with self.assertRaises(RuntimeError):worker.run({key:1.})
                self.assertEqual(baseline['temperature_k'],worker.run({})['temperature_k'])
            finally:worker.close()


if __name__=='__main__':unittest.main()
