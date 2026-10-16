"""Regression tests for contact ownership, not just plotted channel labels."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
spec=importlib.util.spec_from_file_location('upv_geometry',Path(__file__).resolve().parents[1]/'scripts/generate_upv.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class UPVTopologyTests(unittest.TestCase):
    def scene(self,topology='swapped-witnesses',experiment='no-cooling'):
        data={'time_s':[0.,100.]}
        for channel in range(101,111):data[f'T{channel}_k']=[300.,300.]
        for channel in (103,104,105,106):data[f'T{channel}_k'][1]=430.
        for channel,value in [('101',301.),('102',303.),('107',307.),('108',309.)]:
            data[f'T{channel}_k']=[value,value]
        record={'split':'calibration' if experiment=='no-cooling' else 'validation','canonical_sha256':'test'}
        with patch.object(module,'load_case',return_value=(data,record,{'archive_sha256':'test'})):
            return module.generate(Path('/unused'),experiment,topology)

    def test_witness_positions_and_material_sides(self):
        for topology,left,right in [('swapped-witnesses','C3','C1'),('published','C1','C3')]:
            s=self.scene(topology);cells={c['id']:c for c in s['cells']}
            self.assertLess(cells[left]['pose']['position_m'][0],cells[right]['pose']['position_m'][0])
            self.assertEqual(cells['C1']['thermal']['initial_temperature_k'],302.)
            self.assertEqual(cells['C3']['thermal']['initial_temperature_k'],308.)
            for p,(channel,cell,material) in zip(s['measurement_probes'],module.CHANNEL_FACES):
                self.assertEqual((p['id'],p['cell_id'],p['neighbor_material']),(channel,cell,material))
                self.assertEqual(p['port_id'],cell+(':-x' if material=='ISS' else ':+x'))
        probes={p['id']:p for p in self.scene()['measurement_probes']}
        self.assertEqual(probes['101']['neighbor_port_id'],'ISS3:+x')
        self.assertEqual(probes['107']['neighbor_port_id'],'TIM1:-x')

    def test_outer_insulation_is_in_series_not_bypassed(self):
        s=self.scene();edges={e['id'] for e in s['interfaces']}
        self.assertIn('Al3--ISS4',edges);self.assertIn('ISS4--steel-R',edges)
        self.assertNotIn('Al3--steel-R',edges)

    def test_measured_battery_properties_are_fixed_in_both_cases(self):
        for experiment in ('no-cooling','active-cooling'):
            for cell in self.scene(experiment=experiment)['cells']:
                if cell['material_id']!='battery':continue
                self.assertEqual(cell['thermal']['conductivity_w_per_m_k'],[.9,28.3,28.3])
                self.assertEqual(cell['thermal']['heat_capacity_j_per_k'],.410*1007)

    def test_contact_planes_and_normals(self):
        s=self.scene();ports={p['id']:p for p in s['ports']}
        for e in s['interfaces']:
            a,b=ports[e['port_a']],ports[e['port_b']]
            self.assertEqual(a['area_m2'],b['area_m2'])
            for av,bv in zip(a['centroid_m'],b['centroid_m']):self.assertAlmostEqual(av,bv,places=12)
            self.assertEqual(a['normal'],[-v for v in b['normal']])

    def test_binding_is_invariant_to_storage_and_endpoint_order(self):
        s=self.scene();expected=copy.deepcopy(s['measurement_probes'])
        for name in ('cells','ports','interfaces'):s[name].reverse()
        for e in s['interfaces']:e['port_a'],e['port_b']=e['port_b'],e['port_a']
        module.bind_measurement_probes(s)
        self.assertEqual(s['measurement_probes'],expected)

    def test_missing_or_ambiguous_material_contact_is_rejected(self):
        s=self.scene();edge=s['measurement_probes'][0]['interface_id']
        s['interfaces']=[e for e in s['interfaces'] if e['id']!=edge]
        with self.assertRaisesRegex(ValueError,'Channel 101'):module.bind_measurement_probes(s)
        s=self.scene();duplicate=copy.deepcopy(next(e for e in s['interfaces'] if e['id']==edge))
        duplicate['id']='ambiguous-copy';s['interfaces'].append(duplicate)
        with self.assertRaisesRegex(ValueError,'Channel 101'):module.bind_measurement_probes(s)

    def test_forcing_is_not_swapped_with_witnesses(self):
        for experiment in ('no-cooling','active-cooling'):
            scenes=[self.scene(t,experiment) for t in module.TOPOLOGIES]
            def sources(s):return [b for b in s['boundaries'] if b['id'].startswith(('source-','cooling-'))]
            self.assertEqual(sources(scenes[0]),sources(scenes[1]))


if __name__=='__main__':unittest.main()
