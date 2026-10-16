"""Mesh sensitivity must not silently refit contacts or change material data."""
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
    from check_upv_battery_mesh import variant
    from plot_upv_battery_mesh import iss_to_tim_sign
    from generate_upv import ROOT, generate
finally:sys.path.pop(0)


class BatteryMeshChecks(unittest.TestCase):
    def test_only_battery_depth_and_closure_change(self):
        base=generate(ROOT/'data/experiments/upv','no-cooling',connectivity='directed_geometry',battery_divisions=(2,2,2))
        snapshot=copy.deepcopy(base)
        for nx,iso in [(2,False),(4,False),(8,False),(2,True)]:
            alt=variant(base,nx,iso)
            for original,c in zip(base['cells'],alt['cells']):
                if c['role']=='battery':
                    self.assertEqual(c['thermal']['divisions'],[nx,2,2])
                    self.assertEqual(c['thermal'].pop('internal_heat_transfer'),'isothermal' if iso else 'finite_volume')
                    c['thermal']['divisions'][0]=original['thermal']['divisions'][0]
                self.assertEqual(c,original)
            del alt['metadata']['mesh_diagnostic']
            self.assertEqual(alt,base)
        self.assertEqual(base,snapshot)
        for n in (0,1,3,-2):
            with self.assertRaises(ValueError):variant(base,n)

    def test_flux_sign_follows_ports_not_witness_numbering(self):
        for topology in ('published','swapped-witnesses'):
            scene=generate(ROOT/'data/experiments/upv','no-cooling',topology=topology,connectivity='directed_geometry')
            for cell in ('C1','C3'):self.assertEqual(iss_to_tim_sign(scene,cell),1.)
            for edge in scene['interfaces']:edge['port_a'],edge['port_b']=edge['port_b'],edge['port_a']
            for cell in ('C1','C3'):self.assertEqual(iss_to_tim_sign(scene,cell),1.)

    @unittest.skipUnless(os.environ.get('BATTERY_SIM_PROFILE_PROBE'),'CUDA depth diagnostic not requested')
    def test_profile_probe_steady_linear_flux_and_isothermal_null(self):
        # A unit cube with k=1 and exactly linear 300->400 K profile:
        # -100 W across EVERY x cut, independently of discretization.
        for nx,iso,reverse in [(2,False,False),(4,False,False),(8,False,False),(4,False,True),(2,True,False)]:
            tl,tr=(400.,300.) if reverse else (300.,400.)
            s=dict(name='Depth diagnostic manufactured solution',
                   simulation=dict(duration_s=.2,max_dt_s=.01,output_interval_s=.1,stability_factor=.4),
                   cells=[dict(id='battery',role='battery',geometry=dict(kind='pouch',dimensions_m=[1.,1.,1.]),
                               pose=dict(position_m=[0.,0.,0.]),thermal=dict(heat_capacity_j_per_k=100.,initial_temperature_k=350.,
                               initial_temperature_field_k=[350. if iso else tl+(tr-tl)*(i+.5)/nx for i in range(nx)],
                               divisions=[nx,1,1],conductivity_w_per_m_k=[1.,1.,1.],
                               internal_heat_transfer='isothermal' if iso else 'finite_volume'),
                               runaway=dict(energy_j=0.,duration_s=1.))],
                   ports=[dict(id='left',cell_id='battery',kind='contact',surface='-x',area_m2=1.,centroid_m=[-.5,0.,0.],normal=[-1.,0.,0.]),
                          dict(id='right',cell_id='battery',kind='contact',surface='+x',area_m2=1.,centroid_m=[.5,0.,0.],normal=[1.,0.,0.])],
                   interfaces=[],boundaries=[dict(id='left',port_id='left',temperature_k=tl,conductance_w_per_k=100. if iso else 1e12),
                                             dict(id='right',port_id='right',temperature_k=tr,conductance_w_per_k=100. if iso else 1e12)],
                   primary_failures=[])
            with tempfile.TemporaryDirectory(prefix='battery-depth-test-') as directory:
                path=Path(directory)/'scenario.json';path.write_text(json.dumps(s))
                result=subprocess.run([os.environ['BATTERY_SIM_PROFILE_PROBE'],str(path)],check=True,capture_output=True,text=True)
                out=json.loads(result.stdout)
            self.assertLess(abs(out['energy_residual_j']),1e-8)
            track=out['objects']['battery']
            self.assertEqual(len(track['slab_temperature_k']),3)
            for profile in track['slab_temperature_k']:
                for i,t in enumerate(profile):self.assertAlmostEqual(t,350. if iso else tl+(tr-tl)*(i+.5)/nx,places=8)
            for power in track['midplane_left_to_right_power_w']:
                if iso:self.assertIsNone(power)
                else:self.assertAlmostEqual(power,tl-tr,places=7)


if __name__=='__main__':unittest.main()
