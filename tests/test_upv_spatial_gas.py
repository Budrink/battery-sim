"""Geometry contracts, not fits to witness temperatures."""
import collections
import copy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import upv_r123_model as model
from upv_connectivity import reference_gas_conductance


class SpatialGasGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, _ = model.build('no-cooling', duration=1.)

    def test_no_global_side_hubs_or_inlet_shortcuts(self):
        gas = self.scene['gas_network']
        self.assertFalse(any(n['id'].startswith('near-') or n['id'].endswith('-inlet')
                             for n in gas['compartments']))
        for e in gas['channels']:
            if e['to'] == 'chamber':
                self.assertTrue(e['from'].startswith('void-'))
                self.assertTrue(e['id'].startswith('void-face-'))

    def test_external_volume_and_pressure_inventory(self):
        gas = self.scene['gas_network']
        external = [n for n in gas['compartments'] if not n['id'].startswith('inside-')]
        self.assertAlmostEqual(sum(n['volume_m3'] for n in external), .120, places=14)
        pressures = [n['initial_mass_kg']*287*n['initial_temperature_k']/n['volume_m3']
                     for n in gas['compartments']]
        self.assertLess(max(pressures)-min(pressures), 1e-8)
        self.assertTrue(all(n['initial_oxidizer_mass_fraction'] == .232 for n in external))

    def test_all_external_voids_reachable_without_crossing_a_shell(self):
        gas = self.scene['gas_network']; graph = collections.defaultdict(set)
        for e in gas['channels']:
            if e['kind'] == 'passage':
                graph[e['from']].add(e['to']); graph[e['to']].add(e['from'])
        start=next(n['id']for n in gas['compartments']if n['id'].startswith('void-'))
        seen = {start}; queue = [start]
        for a in queue:
            for b in graph[a]-seen:
                seen.add(b); queue.append(b)
        external = {n['id'] for n in gas['compartments'] if not n['id'].startswith('inside-')}
        self.assertEqual(seen, external)

    def test_void_connections_are_shared_faces_with_geometric_areas(self):
        boxes = self.scene['metadata']['gas_geometry']['void_boxes']
        for e in self.scene['gas_network']['channels']:
            self.assertGreater(e['mixing_length_m'], 0.)
            if e['from'] not in boxes or e['to'] not in boxes:
                continue
            a, b = boxes[e['from']], boxes[e['to']]
            overlaps = [min(a['hi'][k], b['hi'][k])-max(a['lo'][k], b['lo'][k]) for k in range(3)]
            axes = [k for k, d in enumerate(overlaps) if abs(d) < 1e-11]
            self.assertEqual(len(axes), 1, e['id'])
            area = math.prod(d for k, d in enumerate(overlaps) if k != axes[0])
            self.assertGreater(area, 0.)
            self.assertAlmostEqual(e['area_m2'], area, places=13)

    def test_gaps_have_finite_two_sided_contacts_not_lte(self):
        gas = self.scene['gas_network']; ports = {p['id']: p for p in self.scene['ports']}
        for n in gas['compartments']:
            if not n['id'].startswith('gap-'):
                continue
            self.assertNotIn('local_equilibrium', n)
            contacts = [c for c in gas['thermal_contacts'] if c['compartment'] == n['id']]
            self.assertEqual(len(contacts), 2)
            targets = [ports[c['port_id']]['cell_id'] for c in contacts]
            self.assertEqual(sum(':case:' in t for t in targets), 1)
            self.assertTrue(all(c['connectivity']['kind']=='directed_geometry' for c in contacts))
            self.assertTrue(all(reference_gas_conductance(c,ports[c['port_id']]['area_m2']) > 0 for c in contacts))

    def test_rupture_is_unique_local_and_initially_closed(self):
        gas = self.scene['gas_network']; seen = set()
        routes = {r['id']: r for r in self.scene['metadata']['gas_geometry']['rupture_routes']}
        for e in gas['channels']:
            if e['kind'] != 'rupture':
                continue
            key = e['port_id'], e['shell_thermal_volume']
            self.assertNotIn(key, seen); seen.add(key)
            self.assertGreater(e['opening_pressure_pa'], 0)
            self.assertEqual(e['to'], routes[e['id']]['receiver'])
            self.assertNotIn(e['to'], ('chamber', 'outside_domain'))
        self.assertEqual(len(seen), 3*6*9)

    def test_geometry_controls_do_not_change_solid_chemistry_or_contacts(self):
        changed, _ = model.build('no-cooling', duration=1.,
                                gas_geometry=dict(outer_extent_m=.03, face_gap_m=.0004))
        for field in ('cells', 'shells', 'interfaces', 'boundaries', 'measurement_probes'):
            self.assertEqual(changed[field], self.scene[field], field)
        solid_ports=lambda s:[p for p in s['ports'] if not p['id'].startswith('chamber-wall:gas-partition:')]
        self.assertEqual(solid_ports(changed),solid_ports(self.scene))

    def test_sealed_spatial_wall_and_pressure_observer(self):
        gas=self.scene['gas_network'];meta=self.scene['metadata']['gas_geometry']
        boxes=meta['void_boxes'];bounds=meta['envelope_m']
        self.assertNotIn('chamber',boxes)
        self.assertNotIn('chamber',{n['id']for n in gas['compartments']})
        self.assertFalse(any(e['to']=='outside_domain' for e in gas['channels']))
        contacts=[c for c in gas['thermal_contacts'] if c['port_id'].startswith('chamber-wall:gas-partition:')]
        ports={p['id']:p for p in self.scene['ports']}
        for c in contacts:
            box=boxes[c['compartment']]
            self.assertTrue(any(abs(box[key][k]-bounds[side][k])<1e-12
                for side,key in enumerate(('lo','hi')) for k in range(3)))
        self.assertAlmostEqual(sum(reference_gas_conductance(c,ports[c['port_id']]['area_m2']) for c in contacts),model.DEFAULT['chamber_gas_h'],places=10)
        observer=self.scene['metadata']['gas_pressure_observers']['chamber']['compartments']
        self.assertEqual(set(observer),{n['id']for n in gas['compartments']if not n['id'].startswith('inside-')})

    def test_closed_cooling_plate_has_no_fictitious_bottom_outlet(self):
        scene, _ = model.build('active-cooling', duration=1.)
        gas = scene['gas_network']
        self.assertFalse(any(e['id'].startswith('rupture-') and '--z-' in e['id']
                             for e in gas['channels']))
        self.assertFalse(any(e['id'].startswith('escape-') and e['id'].endswith('--z')
                             for e in gas['channels']))
        graph = collections.defaultdict(set)
        for e in gas['channels']:
            if e['kind'] == 'passage':
                graph[e['from']].add(e['to']); graph[e['to']].add(e['from'])
        start=next(n['id']for n in gas['compartments']if n['id'].startswith('void-'))
        seen = {start}; queue = [start]
        for a in queue:
            for b in graph[a]-seen:
                seen.add(b); queue.append(b)
        self.assertEqual(len(seen), sum(not n['id'].startswith('inside-') for n in gas['compartments']))

    def test_saved_diagnostic_solids_and_mechanical_laws_are_preserved(self):
        path = ROOT/'data/calibration/upv-separated-chemistry-v6/phase-mixing-20261019-r2/combined'
        if not path.exists():
            self.skipTest('Local diagnostic fixture unavailable')
        prior = json.loads((path/'scenario.json').read_text())
        p = json.loads((path/'parameters.json').read_text())
        from check_upv_phase_mixing import CASES
        scene, _ = model.build('no-cooling', p['parameters'], duration=1., gas_extensions=CASES['combined'])
        for field in ('cells', 'shells', 'interfaces', 'boundaries', 'measurement_probes'):
            self.assertEqual(prior[field], scene[field], field)
        edges = {e['id']: e for e in scene['gas_network']['channels']}
        for old in prior['gas_network']['channels']:
            if old['kind'] != 'rupture':
                continue
            a, b = copy.deepcopy(old), copy.deepcopy(edges[old['id']])
            a.pop('to'); b.pop('to')
            self.assertEqual(b.pop('mixing_axis'),'xyz'.index(old['port_id'].split(':')[-2][1]))
            a.pop('mixing_axis',None)
            self.assertEqual(a, b, old['id'])

    def test_invalid_geometry_rejected(self):
        for cfg in ({'typo': 1.}, {'face_gap_m': 0.}, {'outer_extent_m': float('nan')},
                    {'outer_extent_m': .2}, {'face_gap_m': .02}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                model.build('no-cooling', duration=1., gas_geometry=cfg)


if __name__ == '__main__':
    unittest.main()
