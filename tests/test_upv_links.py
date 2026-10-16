"""Links-only calibration freezes every node capacity and unselected contact."""
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
    from calibrate_upv_links import (links_scene, current_parameters, check_parameters, bake, bindings,
        audit_run, pack, unpack, parameter_names, CONTACTS, INTERNAL_CONTRAST, PRIOR_DIR)
    from calibrate_upv import Worker
finally:sys.path.pop(0)

class LinksTests(unittest.TestCase):
    def scene(self,contrast=False):
        s=json.loads((PRIOR_DIR/'no-cooling-conductivity-scenario.json').read_text())
        s['simulation'].update(duration_s=2.,output_interval_s=1.)
        return links_scene(s,contrast)

    def params(self,scene):
        p=current_parameters(scene)
        p.update(internal_log_35=-2.,internal_log_50=-1.,internal_log_70=-3.,battery_iss_scale=3.,iss_aluminium_scale=4.)
        if INTERNAL_CONTRAST in p:p[INTERNAL_CONTRAST]=-3.
        return p

    def test_bake_changes_only_selected_links(self):
        for contrast in (False,True):
            scene=self.scene(contrast);snapshot=copy.deepcopy(scene);out=bake(scene,self.params(scene))
            for original,changed in zip(scene['cells'],out['cells']):
                if original['role']=='battery':
                    changed['thermal']['bulk_response']=original['thermal']['bulk_response']
                self.assertEqual(original,changed)
                self.assertNotIn('heat_capacity_law',changed['thermal'])
            for old,new in zip(bindings(scene)[0],bindings(out)[0]):
                if old[1] in CONTACTS:new[0]['law']['geometry_reference_logit']=old[0]['law']['geometry_reference_logit']
            for old,new in zip(bindings(scene)[1],bindings(out)[1]):
                if old[1] in CONTACTS:new[0]['connectivity']['geometry_reference_logit']=old[0]['connectivity']['geometry_reference_logit']
            out['metadata']=scene['metadata'];self.assertEqual(out,scene);self.assertEqual(scene,snapshot)

    def test_parameters_and_capacity_rejection(self):
        for contrast in (False,True):
            scene=self.scene(contrast);p=self.params(scene)
            roundtrip=unpack(pack(p),parameter_names(p))
            for k in p:self.assertAlmostEqual(roundtrip[k],p[k])
            for key,value in (('capacity_45',1.),('ambient_scale',1.),('internal_log_35',-7.),('battery_iss_scale',0.)):
                invalid=dict(p);invalid[key]=value
                with self.assertRaises(ValueError):check_parameters(invalid)
        for role in ('battery','passive'):
            scene=self.scene();c=next(c for c in scene['cells'] if c['role']==role)
            c['thermal']['heat_capacity_law']={'factor':[[298.15,1.],[343.15,1.]]}
            with self.assertRaisesRegex(ValueError,'capacity'):links_scene(scene)

    @unittest.skipUnless(os.environ.get('BATTERY_SIM_EXPERIMENT_PROBE'),'CUDA probe not requested')
    def test_native_guard_export_audit_reset(self):
        binary=Path(os.environ['BATTERY_SIM_EXPERIMENT_PROBE'])
        with tempfile.TemporaryDirectory(prefix='links-only-') as directory:
            path=Path(directory)/'scene.json'
            for contrast in (False,True):
                scene=self.scene(contrast);params=self.params(scene);path.write_text(json.dumps(scene));w=Worker(binary,path)
                try:
                    prior=w.run({});run=w.run(params);audit_run(run,scene,params,prior['applied_model'])
                    rejected=['capacity_45','capacity_70','iss_cp_scale','tim_cp_scale','cell_cp_scale','battery_tim_scale',
                              'iss_steel_scale','tim_aluminium_scale','ambient_scale','surface_linear','battery_temperature_contrast']
                    if not contrast:rejected.append(INTERNAL_CONTRAST)
                    for key in rejected:
                        with self.assertRaises(RuntimeError):w.run({key:1.})
                    for item in ({'battery_iss_scale':.019},{'internal_log_35':6.1},{INTERNAL_CONTRAST:12.1}):
                        with self.assertRaises(RuntimeError):w.run(item)
                    self.assertEqual(prior['temperature_k'],w.run({})['temperature_k'])
                    tampered=copy.deepcopy(run);tampered['applied_model']['objects'][0]['heat_capacity_j_per_k']+=1.
                    with self.assertRaises(ValueError):audit_run(tampered,scene,params,prior['applied_model'])
                    tampered=copy.deepcopy(run)
                    b=next(b for b in tampered['applied_model']['boundaries'] if b['id'].startswith('source-') and b['group'] in CONTACTS)
                    b['reference_h_w_m2_k']+=1.
                    with self.assertRaisesRegex(ValueError,'Source'):audit_run(tampered,scene,params,prior['applied_model'])
                finally:w.close()
                path.write_text(json.dumps(bake(scene,params)));w=Worker(binary,path)
                try:
                    exported=w.run({});audit_run(exported,scene,params,prior['applied_model'])
                    self.assertEqual(exported['temperature_k'],run['temperature_k'])
                finally:w.close()

    @unittest.skipUnless(os.environ.get('BATTERY_SIM_EXPERIMENT_PROBE'),'CUDA probe not requested')
    def test_native_rejects_capacity_law_on_passive_solid(self):
        scene=self.scene();next(c for c in scene['cells'] if c['role']=='passive')['thermal']['heat_capacity_law']={
            'factor':[[298.15,1.],[343.15,1.]]}
        with tempfile.TemporaryDirectory(prefix='links-capacity-') as directory:
            path=Path(directory)/'scene.json';path.write_text(json.dumps(scene))
            r=subprocess.run([os.environ['BATTERY_SIM_EXPERIMENT_PROBE'],str(path)],input='{}\n',text=True,capture_output=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('ALL capacities constant',r.stderr)

if __name__=='__main__':unittest.main()
