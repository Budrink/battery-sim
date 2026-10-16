import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from upv_cooling_wall import attach, WALL_ID, SINK_ID
from replay_upv_cooling_wall import transfer_warm_fields
from generate_upv import generate
import generate_upv


class CoolingWallTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads((ROOT/'data/calibration/upv-physical-casing-v7/heat/active-cooling-fitted-scenario.json').read_text())

    def test_ordinary_material_and_no_legacy_bypass(self):
        before = copy.deepcopy(self.original)
        s = attach(self.original)
        self.assertEqual(before, self.original)
        self.assertEqual(s['cells'][:-1], before['cells'])
        wall = s['cells'][-1]
        self.assertEqual(wall['role'], 'passive')
        self.assertEqual(wall['degradation'], dict(initial_damage=0., rate_at_onset_per_s=0.))
        self.assertEqual(wall['runaway']['energy_j'], 0.)
        self.assertAlmostEqual(wall['thermal']['heat_capacity_j_per_k'], .07*.087*.013*2700*900)
        self.assertFalse(any(b['id'].startswith('cooling-Al') for b in s['boundaries']))
        sink = next(b for b in s['boundaries'] if b['id']==SINK_ID)
        self.assertNotIn('temperature_profile', sink)
        self.assertEqual(sink['conductance_w_per_k'], 10.)

    def test_contact_areas_positions_and_observer(self):
        s = attach(self.original); ports = {p['id']: p for p in s['ports']}
        for edge in s['interfaces'][-3:]:
            a, b = ports[edge['port_a']], ports[edge['port_b']]
            self.assertEqual(a['centroid_m'], b['centroid_m'])
            self.assertEqual(a['normal'], [-x for x in b['normal']])
            self.assertAlmostEqual(a['area_m2'], b['area_m2'])
            self.assertEqual(edge['law']['kind'], 'directed_geometry')
        observer = next(p for p in s['measurement_probes'] if p['id']=='110')
        self.assertEqual(observer['cell_id'], 'Al2')
        self.assertEqual(observer['interface_id'], 'Al2--cooling-plate')

    def test_old_contact_temperature_does_not_drive_wall(self):
        changed = copy.deepcopy(self.original)
        for b in changed['boundaries']:
            if b['id'].startswith('cooling-Al'):
                b['temperature_k'] = 1000.
                b['temperature_profile'] = [[0., 1000.], [2000., 2000.]]
        self.assertEqual(attach(self.original), attach(changed))

    def test_generator_default_and_historical_mode(self):
        wall = generate(ROOT/'data/experiments/upv', 'active-cooling')
        old = generate(ROOT/'data/experiments/upv', 'active-cooling', cooling_model='legacy-boundary')
        self.assertIn(WALL_ID, [c['id'] for c in wall['cells']])
        self.assertNotIn(WALL_ID, [c['id'] for c in old['cells']])
        self.assertTrue(any(b['id']=='cooling-Al1' for b in old['boundaries']))

    def test_channel_110_is_not_a_window_or_forcing_driver(self):
        inputs = generate_upv.load_case(ROOT/'data/experiments/upv', 'active-cooling')
        data, record, manifest = copy.deepcopy(inputs)
        data['T110_k'] = [float('nan')]*len(data['time_s'])
        with patch.object(generate_upv, 'load_case', return_value=(data, record, manifest)):
            s = generate(ROOT/'data/experiments/upv', 'active-cooling')
        self.assertIn(WALL_ID, [c['id'] for c in s['cells']])
        self.assertGreater(s['simulation']['duration_s'], 0.)

    def test_invalid_geometry_controls_and_duplicates(self):
        for options in [dict(sink_conductance_w_per_k=-1), dict(contact_h_w_per_m2_k=0),
                        dict(divisions=(0, 1, 1)), dict(coolant_temperature_k=float('nan'))]:
            with self.assertRaises(ValueError): attach(self.original, **options)
        with self.assertRaises(ValueError): attach(attach(self.original))
        changed = copy.deepcopy(self.original)
        next(p for p in changed['ports'] if p['id']=='Al2:-z')['centroid_m'][2] += .01
        with self.assertRaises(ValueError): attach(changed)

    def test_warmup_field_transferred_without_reset(self):
        s = attach(self.original)
        fields = {}
        for c in s['cells']:
            n = 1
            for size in c['thermal']['divisions']: n *= size
            fields[c['id']] = [320.+i*.01 for i in range(n)]
        for sh in s['shells']:
            for face in ('-x', '+x', '-y', '+y', '-z', '+z'):
                fields[sh['contents_id']+':case:'+face] = [330.]*9
        result = transfer_warm_fields(s, dict(final_fields_k=fields))
        wall = next(c for c in result['cells'] if c['id']==WALL_ID)
        self.assertEqual(wall['thermal']['initial_temperature_field_k'], fields[WALL_ID])
        self.assertNotIn('initial_temperature_field_k', s['cells'][-1]['thermal'])


if __name__ == '__main__':
    unittest.main()
