"""Two-parameter fit must not retune contacts or measured material properties."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from calibrate_upv_internal import internal_scene, export_scene, verify_internal, grid_protocol, PROTOCOL, NAMES
    from calibrate_upv import Worker
    from generate_upv import generate, ROOT
finally:sys.path.pop(0)


class InternalFitTests(unittest.TestCase):
    def test_grid_is_the_only_protocol_or_scene_change(self):
        self.assertEqual(grid_protocol(3),PROTOCOL)
        original=internal_scene(self.scene())
        for grid in (4,8):
            base=self.scene()
            for c in base['cells']:
                if c['role']=='battery':c['thermal']['divisions']=[grid]*3
            with self.assertRaises(ValueError):internal_scene(base)
            other=internal_scene(base,grid)
            for a,b in zip(original['cells'],other['cells']):
                if a['role']=='battery':b['thermal']['divisions']=a['thermal']['divisions']
                self.assertEqual(a,b)
            for name in ('ports','interfaces','boundaries','measurement_probes'):self.assertEqual(original[name],other[name])
            protocol=other['metadata']['internal_protocol'];self.assertEqual(protocol.pop('grid'),[grid]*3)
            protocol.pop('revision');expected=dict(PROTOCOL);expected.pop('grid');expected.pop('revision')
            self.assertEqual(protocol,expected)
        for grid in (2,5,True,3.):
            with self.assertRaises(ValueError):grid_protocol(grid)

    def scene(self):
        s=generate(ROOT/'data/experiments/upv','no-cooling',connectivity='directed_geometry')
        s['simulation'].update(duration_s=2.,output_interval_s=1.)
        return s

    def test_independent_law_and_export_preserve_contacts(self):
        base=self.scene();snapshot=copy.deepcopy(base);scene=internal_scene(base)
        out=export_scene(scene,dict(zip(NAMES,(-2.,3.))))
        for name in ('interfaces','ports','boundaries','measurement_probes'):self.assertEqual(out[name],base[name])
        for a,b in zip(base['cells'],out['cells']):
            if a['role']=='battery':
                self.assertEqual(b['thermal'].pop('bulk_response')['linear'],[-2.]*3)
                law=b['thermal'].pop('bulk_connectivity');self.assertEqual(law['kind'],'directed_geometry')
            self.assertEqual(a,b)
        self.assertEqual(base,snapshot)
        with self.assertRaises(ValueError):export_scene(scene,{'internal_linear':0.})
        with self.assertRaises(ValueError):export_scene(scene,dict(zip(NAMES,(13.,0.))))

    @unittest.skipUnless(os.environ.get('BATTERY_SIM_EXPERIMENT_PROBE'),'CUDA probe not requested')
    def test_native_frozen_audit_reset_and_export(self):
        s=internal_scene(self.scene());params=dict(zip(NAMES,(-2.,3.)))
        with tempfile.TemporaryDirectory(prefix='upv-internal-') as directory:
            path=Path(directory)/'scene.json';path.write_text(json.dumps(s));w=Worker(Path(os.environ['BATTERY_SIM_EXPERIMENT_PROBE']),path)
            try:
                zero=w.run({});audit=zero['applied_model'];run=w.run(params);verify_internal(run,s,params,audit)
                for key in ('ambient_scale','battery_tim_scale','surface_linear','iss_cp_scale','cell_k_scale'):
                    with self.assertRaises(RuntimeError):w.run({key:1.})
                with self.assertRaises(RuntimeError):w.run({'internal_linear':13.})
                self.assertEqual(w.run({})['temperature_k'],zero['temperature_k'])
            finally:w.close()
            baked=export_scene(s,params);path.write_text(json.dumps(baked));w=Worker(Path(os.environ['BATTERY_SIM_EXPERIMENT_PROBE']),path)
            try:
                exported=w.run({});verify_internal(exported,baked,params,audit)
                self.assertEqual(exported['temperature_k'],run['temperature_k'])
                self.assertIn('internal_snapshot',exported)
            finally:w.close()
            path.write_text(json.dumps(self.scene()));w=Worker(Path(os.environ['BATTERY_SIM_EXPERIMENT_PROBE']),path)
            try:
                with self.assertRaisesRegex(RuntimeError,'opt-in'):w.run(params)
            finally:w.close()


if __name__=='__main__':unittest.main()
