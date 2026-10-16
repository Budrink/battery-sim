"""Ordered connectivity, fixed-property fit, source-role and observation contracts."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from calibrate_upv import CONTACTS,CONTRASTS,PARAMETER_SETS,Worker,validate_parameters,verify_fixed_properties
    from generate_upv import ROOT,generate
    from export_upv_state_geometry import calibrated_scene
finally:sys.path.pop(0)


class DirectedGeometryContractTests(unittest.TestCase):
    def scene(self):
        s=generate(ROOT/'data/experiments/upv','no-cooling',connectivity='directed_geometry')
        s['simulation'].update(duration_s=2.,output_interval_s=1.)
        for i,c in enumerate(s['cells']):c['thermal']['initial_temperature_k']=310.+i*3
        for b in s['boundaries']:b.update(temperature_k=400.,temperature_profile=[[0,400.],[2,400.]])
        return s

    def parameters(self):
        p={k:2. for k in CONTACTS};p.update(zip(CONTRASTS,(-3.,2.,-1.,4.,-2.)))
        p.update(ambient_scale=.5,surface_linear=-1.,surface_quadratic=2.);return p

    def test_law_and_fit_contract(self):
        s=self.scene();p=self.parameters();validate_parameters(p)
        self.assertEqual(set(p),set(PARAMETER_SETS['directed_geometry']))
        for key in ('iss_cp_scale','tim_cp_scale','cell_k_scale','cell_capacity_scale'):
            with self.assertRaises(ValueError):validate_parameters(dict(p,**{key:1.}))
        for c in s['cells']:self.assertNotIn('bulk_connectivity',c['thermal'])
        for e in s['interfaces']:self.assertEqual(e['law']['kind'],'directed_geometry')
        for b in s['boundaries']:
            if b['id'].startswith('source-'):self.assertEqual(b['connectivity']['kind'],'directed_geometry')

    def test_export_preserves_material_roles_not_edge_order(self):
        scene=self.scene();p=self.parameters()
        profile=dict(parameter_set='directed_geometry',parameters=p,optimizer_success=True,model_sha256='test')
        out=calibrated_scene(scene,profile)
        for original,c in zip(scene['cells'],out['cells']):
            self.assertEqual(c['thermal']['heat_capacity_j_per_k'],original['thermal']['heat_capacity_j_per_k'])
            self.assertEqual(c['thermal']['conductivity_w_per_m_k'],original['thermal']['conductivity_w_per_m_k'])
            self.assertEqual(c['thermal']['surface_connectivity']['temperature_contrast'],[p[c['material_id'].lower()+'_temperature_contrast']]*3)
        for b in out['boundaries']:
            if b['id'].startswith('source-'):
                self.assertEqual(b['remote_surface_connectivity']['temperature_contrast'],[-3.]*3)

    @unittest.skipUnless(os.environ.get('BATTERY_SIM_EXPERIMENT_PROBE'),'CUDA probe not requested')
    def test_gpu_export_sensor_permutation_and_directed_readout(self):
        scene=self.scene();params=self.parameters()
        profile=dict(parameter_set='directed_geometry',parameters=params,optimizer_success=True,model_sha256='test')
        exported=calibrated_scene(scene,profile)
        reordered=copy.deepcopy(scene)
        for name in ('cells','ports','interfaces'):reordered[name].reverse()
        for edge in reordered['interfaces']:edge['port_a'],edge['port_b']=edge['port_b'],edge['port_a']
        with tempfile.TemporaryDirectory(prefix='upv-directed-') as directory:
            def run(s,p,label):
                path=Path(directory)/(label+'.json');path.write_text(json.dumps(s));w=Worker(Path(os.environ['BATTERY_SIM_EXPERIMENT_PROBE']),path)
                try:return w.run(p)
                finally:w.close()
            base=run(scene,params,'base');verify_fixed_properties(base,scene)
            self.assertLess(abs(base['energy_residual_j']),1e-7)
            for b in base['applied_model']['boundaries']:
                if b['id'].startswith('source-'):self.assertEqual(b['remote_temperature_contrast'],-3.)
            self.assertTrue(any(abs(v['a_to_b_conductance_w_per_k'][0]-v['b_to_a_conductance_w_per_k'][0])>1e-4 for v in base['interface_snapshot'].values()))
            for variant,p,label in ((exported,{},'export'),(reordered,params,'reorder')):
                other=run(variant,p,label)
                for channel,ts in base['temperature_k'].items():
                    for x,y in zip(ts,other['temperature_k'][channel]):self.assertAlmostEqual(x,y,places=9)
                for edge,v in base['interface_snapshot'].items():
                    alt=other['interface_snapshot'][edge]
                    key='b_to_a_conductance_w_per_k' if label=='reorder' else 'a_to_b_conductance_w_per_k'
                    for x,y in zip(v['a_to_b_conductance_w_per_k'],alt[key]):self.assertAlmostEqual(x,y,places=9)
                    for x,y in zip(v['net_a_to_b_power_w'],alt['net_a_to_b_power_w']):self.assertAlmostEqual(x,(-y if label=='reorder' else y),places=8)


if __name__=='__main__':unittest.main()
