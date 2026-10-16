"""Core/shell selection is a mesh test, not a material-parameter refit."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from check_upv_core_shell import grid_scene, compare
    from calibrate_upv import Worker, verify_fixed_properties
    from generate_upv import ROOT, generate
finally:sys.path.pop(0)


class CoreShellTests(unittest.TestCase):
    def test_default_and_explicit_grid_preserve_physics(self):
        default=generate(ROOT/'data/experiments/upv','no-cooling',connectivity='directed_geometry')
        four=generate(ROOT/'data/experiments/upv','no-cooling',connectivity='directed_geometry',battery_divisions=(4,4,4))
        for a,b in zip(default['cells'],four['cells']):
            if a['role']=='battery':
                self.assertEqual(a['thermal']['divisions'],[3,3,3])
                self.assertEqual(b['thermal']['divisions'],[4,4,4])
                b['thermal']['divisions']=a['thermal']['divisions']
            self.assertEqual(a,b)
        for key in ('ports','interfaces','boundaries','measurement_probes'):self.assertEqual(default[key],four[key])
        for bad in ((0,3,3),(3,3),(3.,3,3),(True,3,3),(257,3,3)):
            with self.assertRaises(ValueError):generate(ROOT/'data/experiments/upv','no-cooling',battery_divisions=bad)

    def test_refinement_cannot_change_materials_or_contact_bindings(self):
        original=generate(ROOT/'data/experiments/upv','no-cooling',connectivity='directed_geometry')
        snapshot=copy.deepcopy(original)
        for n in (3,4,8,12):
            scene=grid_scene(original,n)
            self.assertEqual(scene['metadata']['core_shell_grid']['core_volumes'],(n-2)**3)
            self.assertEqual(scene['metadata']['core_shell_grid']['shell_volumes'],n**3-(n-2)**3)
            self.assertFalse(scene['metadata']['core_shell_grid']['parameters_refitted'])
            for a,b in zip(original['cells'],scene['cells']):
                if a['role']=='battery':
                    self.assertEqual(b['thermal']['divisions'],[n]*3)
                    b['thermal']['divisions']=a['thermal']['divisions']
                    self.assertEqual(b['thermal'].pop('internal_heat_transfer'),'finite_volume')
                self.assertEqual(a,b)
            scene['metadata']=original['metadata'];self.assertEqual(scene,original)
        self.assertEqual(original,snapshot)
        for n in (0,1,2,3.5):
            with self.assertRaises(ValueError):grid_scene(original,n)

    def test_time_alignment_tolerates_only_roundoff(self):
        a=dict(time_s=[0.,60.,120.],temperature_k={c:[300.,320.,330.] for c in ('101','102','107','108')})
        b=copy.deepcopy(a);b['time_s'][1]+=1e-10
        self.assertEqual(compare(a,b)['max_change_k'],0.)
        b['time_s'][1]+=.001
        with self.assertRaisesRegex(ValueError,'sample times'):compare(a,b)

    @unittest.skipUnless(os.environ.get('BATTERY_SIM_EXPERIMENT_PROBE'),'CUDA probe not requested')
    def test_native_region_temperatures_and_capacities(self):
        for n in (2,3,4):
            scene=generate(ROOT/'data/experiments/upv','no-cooling',connectivity='directed_geometry',battery_divisions=(n,n,n))
            scene['simulation'].update(duration_s=2.,output_interval_s=1.)
            for c in scene['cells']:
                if c['role']=='battery':
                    c['thermal']['initial_temperature_field_k']=[
                        350. if 0<x<n-1 and 0<y<n-1 and 0<z<n-1 else 300.
                        for z in range(n) for y in range(n) for x in range(n)]
            with tempfile.TemporaryDirectory(prefix='battery-core-shell-') as directory:
                path=Path(directory)/'scene.json';path.write_text(json.dumps(scene))
                worker=Worker(Path(os.environ['BATTERY_SIM_EXPERIMENT_PROBE']),path)
                try:result=worker.run({})
                finally:worker.close()
            verify_fixed_properties(result,scene)
            self.assertLess(abs(result['energy_residual_j']),1e-7)
            for obj in result['battery_regions'].values():
                self.assertEqual(obj['core']['volume_count'],(n-2)**3)
                self.assertEqual(obj['shell']['volume_count'],n**3-(n-2)**3)
                self.assertAlmostEqual(sum(r['heat_capacity_j_per_k'] for r in obj.values()),412.87)
                for name,r in obj.items():
                    ts=r['mean_temperature_k'];self.assertEqual(len(ts),len(result['time_s']))
                    if r['volume_count']==0:self.assertEqual(ts,[None]*len(ts));continue
                    self.assertAlmostEqual(ts[0],350. if name=='core' else 300.)
                    for t,lo,hi in zip(ts,r['min_temperature_k'],r['max_temperature_k']):
                        self.assertLessEqual(lo-1e-10,t);self.assertLessEqual(t,hi+1e-10)


if __name__=='__main__':unittest.main()
