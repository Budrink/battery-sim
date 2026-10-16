"""Transport and storage fits must never adjust each other's parameters."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from calibrate_upv_staged import table_scene, capacity_scene, bake_transport, bake_capacity, audit_stage, G_NAMES, C_NAMES
    from calibrate_upv import Worker
    from generate_upv import generate, ROOT
finally:sys.path.pop(0)

G=dict(zip(G_NAMES,[-1.,0.,1.,1.,2.]))
C=dict(zip(C_NAMES,[1.1,.9]))

class StagedTests(unittest.TestCase):
    def test_python_reference_rejects_node_capacity_law(self):
        from battery_sim.io import load_scenario
        from battery_sim.model import ModelError
        scene=json.loads((ROOT/'examples/three-cell-cascade.json').read_text())
        scene['cells'][0]['thermal']['heat_capacity_law']={'factor':[[298.15,1.],[343.15,1.1]]}
        with tempfile.TemporaryDirectory(prefix='capacity-reference-') as directory:
            path=Path(directory)/'scene.json';path.write_text(json.dumps(scene))
            with self.assertRaisesRegex(ModelError,'native CUDA'):load_scenario(path)

    def scene(self):
        s=generate(ROOT/'data/experiments/upv','no-cooling',connectivity='directed_geometry')
        s['simulation'].update(duration_s=2.,output_interval_s=1.)
        return table_scene(s)

    def test_capacity_export_preserves_all_transport(self):
        g=bake_transport(self.scene(),G);snapshot=copy.deepcopy(g)
        out=bake_capacity(capacity_scene(g),C)
        for key in ('ports','interfaces','boundaries','measurement_probes'):self.assertEqual(out[key],g[key])
        for original,changed in zip(g['cells'],out['cells']):
            if changed['role']=='battery':
                law=changed['thermal'].pop('heat_capacity_law')
                self.assertEqual(law['factor'],[[298.15,1.],[318.15,1.1],[343.15,.9]])
                self.assertEqual(law['damage_factor_at_one'],1.)
            self.assertEqual(original,changed)
        self.assertEqual(g,snapshot)
        for bad in ({'capacity_45':1.},dict(zip(C_NAMES,[.79,1.])),dict(zip(C_NAMES,[float('nan'),1.]))):
            with self.assertRaises(ValueError):bake_capacity(capacity_scene(g),bad)

    def test_transport_has_no_capacity_or_polynomial_change(self):
        scene=self.scene();out=bake_transport(scene,G)
        for a,b in zip(scene['cells'],out['cells']):
            self.assertNotIn('heat_capacity_law',b['thermal'])
            self.assertEqual(a['thermal']['heat_capacity_j_per_k'],b['thermal']['heat_capacity_j_per_k'])
            self.assertEqual(a['thermal']['conductivity_w_per_m_k'],b['thermal']['conductivity_w_per_m_k'])
            self.assertEqual(a['thermal']['surface_connectivity'],b['thermal']['surface_connectivity'])
            if b['role']=='battery':self.assertEqual(b['thermal']['bulk_response']['quadratic'],[0.]*3)

    @unittest.skipUnless(os.environ.get('BATTERY_SIM_EXPERIMENT_PROBE'),'CUDA probe not requested')
    def test_native_stage_guards_reset_audit_and_export(self):
        binary=Path(os.environ['BATTERY_SIM_EXPERIMENT_PROBE'])
        with tempfile.TemporaryDirectory(prefix='staged-upv-') as directory:
            path=Path(directory)/'scene.json';g=self.scene()
            for scene,params,bake,rejected in (
                (g,G,bake_transport,('capacity_45','surface_linear','ambient_scale','internal_linear')),
                (capacity_scene(bake_transport(g,G)),C,bake_capacity,('internal_log_35','battery_iss_scale','surface_linear','ambient_scale'))):
                path.write_text(json.dumps(scene));w=Worker(binary,path)
                try:
                    base=w.run({});run=w.run(params);audit_stage(run,scene,params,base['applied_model'])
                    for key in rejected:
                        with self.assertRaises(RuntimeError):w.run({key:1.})
                    self.assertEqual(w.run({})['temperature_k'],base['temperature_k'])
                finally:w.close()
                path.write_text(json.dumps(bake(scene,params)));w=Worker(binary,path)
                try:self.assertEqual(w.run({})['temperature_k'],run['temperature_k'])
                finally:w.close()

    @unittest.skipUnless(os.environ.get('BATTERY_SIM_EXPERIMENT_PROBE'),'CUDA probe not requested')
    def test_invalid_constitutive_laws_fail_explicitly(self):
        s=capacity_scene(bake_transport(self.scene(),G))
        with tempfile.TemporaryDirectory(prefix='invalid-upv-') as directory:
            path=Path(directory)/'scene.json'
            for mutation in ('negative','duplicate','unsorted','unanchored','isothermal','polynomial','oversized'):
                scene=copy.deepcopy(s);t=next(c for c in scene['cells'] if c['role']=='battery')['thermal']
                if mutation=='negative':t['heat_capacity_law']['factor'][1][1]=-.1
                elif mutation=='duplicate':t['heat_capacity_law']['factor'][1][0]=298.15
                elif mutation=='unsorted':t['heat_capacity_law']['factor'].reverse()
                elif mutation=='unanchored':t['heat_capacity_law']['factor'][0][1]=1.1
                elif mutation=='isothermal':t['internal_heat_transfer']='isothermal'
                elif mutation=='polynomial':t['bulk_response']['linear']=[1.]*3
                else:t['bulk_response']['temperature_log_response']=[[298.+i,0.] for i in range(9)]
                path.write_text(json.dumps(scene))
                result=subprocess.run([os.environ['BATTERY_SIM_EXPERIMENT_PROBE'],str(path)],input='{}\n',text=True,capture_output=True)
                self.assertNotEqual(result.returncode,0,mutation)

if __name__=='__main__':unittest.main()
