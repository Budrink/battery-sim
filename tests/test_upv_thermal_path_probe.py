"""Opt-in CUDA regression for the read-only thermal-path observer."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import calibrate_upv_r123 as cal
import upv_r123_model as model


@unittest.skipUnless(os.environ.get('BATTERY_SIM_CUDA_TESTS')=='1','Requires explicit CUDA test opt-in')
class ThermalPathObserver(unittest.TestCase):
    def test_serialization_filter_preserves_physics_and_audits(self):
        scene,_=model.build('no-cooling',dict(model.DEFAULT),warmup=True,duration=10.)
        selected=copy.deepcopy(scene)
        gas_id=scene['gas_network']['compartments'][0]['id']
        channel_id=scene['gas_network']['channels'][0]['id']
        selected['process_history_selection']=dict(gas_prefixes=[gas_id],channel_prefixes=[channel_id])
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            plain=cal.replay(scene,folder,'plain',30.)
            filtered=cal.replay(selected,folder,'filtered',30.)
        for key in ('time_s','temperature_k','cell_mean_temperature_k','chemistry','final_fields_k',
                    'joule_energy_j','released_energy_j_history','remaining_energy_j','steps',
                    'max_thermal_residual_j','max_chemical_residual_j','max_mass_residual_kg'):
            self.assertEqual(plain[key],filtered[key],key)
        for section,prefix in (('gas',gas_id),('channels',channel_id)):
            self.assertEqual(set(filtered[section]),{k for k in plain[section]if k.startswith(prefix)})
            for key,value in filtered[section].items():self.assertEqual(value,plain[section][key])

    def test_observer_preserves_state_and_reports_physical_contacts(self):
        # Test the current schema, not an obsolete calibration profile whose
        # electrical fields no longer belong to the current reaction model.
        scene,_=model.build('no-cooling',dict(model.DEFAULT),warmup=True,duration=30.)
        traced=copy.deepcopy(scene);traced['thermal_path_trace']=True;traced['surface_probe_trace']=True
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            plain=cal.replay(scene,folder,'plain',30.)
            observed=cal.replay(traced,folder,'traced',30.)
        self.assertNotIn('thermal_paths',plain)
        for key in ('time_s','temperature_k','cell_mean_temperature_k','chemistry','gas',
                    'channels','joule_energy_j','released_energy_j_history','remaining_energy_j','steps'):
            self.assertEqual(plain[key],observed[key],key+' changed with read-only observation')
        trace=observed['thermal_paths'];n=len(observed['time_s'])
        self.assertNotIn('surface_probe_trace',plain)
        for key,envelope in observed['surface_probe_trace'].items():
            self.assertEqual(len(envelope['minimum_temperature_k']),n)
            for mean,lo,hi in zip(observed['temperature_k'][key],envelope['minimum_temperature_k'],envelope['maximum_temperature_k']):
                self.assertLessEqual(lo-1e-10,mean)
                self.assertLessEqual(mean,hi+1e-10)
        self.assertEqual(set(trace['objects']),{o['id']for o in observed['applied_objects']})
        self.assertIn('ISS3--C1',trace['interfaces'])
        for contact in trace['interfaces'].values():
            self.assertIn(contact['from'],trace['objects']);self.assertIn(contact['to'],trace['objects'])
            self.assertEqual(len(contact['a_to_b_power_w']),n)
            self.assertTrue(all(g>0 for g in contact['conductance_w_per_k']))
            # Initial endpoint fields are uniform. Later snapshots can have
            # local gradients: never replace sum(G_ij*dT_ij) by G*mean(dT).
            q=contact['a_to_b_power_w'][0];g=contact['conductance_w_per_k'][0]
            delta=contact['a_adjacent_temperature_k'][0]-contact['b_adjacent_temperature_k'][0]
            self.assertAlmostEqual(q,g*delta,delta=1e-9)


if __name__=='__main__':unittest.main()
