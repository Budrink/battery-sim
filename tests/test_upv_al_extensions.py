import copy
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from upv_al_extensions import candidate


class AluminiumExtensions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = Path(__file__).resolve().parents[1] / 'data/calibration/upv-separated-chemistry-v6/damaged-contact-20261022-r1/trial-002/scenario.json'
        if not p.exists():
            raise unittest.SkipTest('Current calibration geometry is optional')
        cls.source = json.loads(p.read_text())

    def test_neutral_and_source_immutable(self):
        self.assertEqual(candidate(self.source, 0.), self.source)
        before = copy.deepcopy(self.source)
        candidate(self.source, .01)
        self.assertEqual(self.source, before)

    def test_materials_and_cold_gas_pressure_preserved(self):
        new = candidate(self.source, .015)
        self.assertEqual(new['cells'][:-3], self.source['cells'])
        old_cells = {c['id']: c for c in self.source['cells']}
        for c in new['cells'][-3:]:
            old = old_cells[c['id'].split(':')[0]]
            self.assertAlmostEqual(c['thermal']['heat_capacity_j_per_k']/math.prod(c['geometry']['dimensions_m']),
                old['thermal']['heat_capacity_j_per_k']/math.prod(old['geometry']['dimensions_m']))
            self.assertEqual(c['thermal']['conductivity_w_per_m_k'], old['thermal']['conductivity_w_per_m_k'])
        for a, b in zip(self.source['gas_network']['compartments'], new['gas_network']['compartments']):
            self.assertAlmostEqual(a['initial_mass_kg']/a['volume_m3'], b['initial_mass_kg']/b['volume_m3'])
            for k in a:
                if k not in ('volume_m3', 'initial_mass_kg'):
                    self.assertEqual(a[k], b[k])
        for k in ('shells', 'primary_failures', 'measurement_probes', 'simulation', 'boundaries'):
            self.assertEqual(new[k], self.source[k])

    def test_mass_exclusion_and_no_open_face_through_aluminium(self):
        new = candidate(self.source, .01)
        m = new['metadata']['al_bottom_extensions']
        displaced = 3 * .001 * .087 * .01
        self.assertAlmostEqual(m['displaced_volume_m3'], displaced)
        self.assertAlmostEqual(sum(m['occupied_voids_m3'].values()), displaced)
        old = {e['id']: e for e in self.source['gas_network']['channels']}
        after = {e['id']: e for e in new['gas_network']['channels']}
        for row in m['narrowed_faces']:
            if row['channel'] in after:
                self.assertAlmostEqual(old[row['channel']]['area_m2']-after[row['channel']]['area_m2'], row['blocked_area_m2'])
        for key, edge in after.items():
            if 'area_m2' in edge:
                self.assertGreater(edge['area_m2'], 0.)
            if not key.startswith('void-face-'):
                self.assertEqual(edge, old[key])
        ports = {p['id']: p for p in new['ports']}
        area = sum(ports[c['port_id']]['area_m2'] for c in new['gas_network']['thermal_contacts']
                   if ':bottom-extension' in c['port_id'])
        self.assertAlmostEqual(area, 3 * (2*(.001+.087)*.01 + .001*.087))

    def test_new_contacts_reuse_the_existing_directed_material_law(self):
        new = candidate(self.source, .005)
        ident = new['metadata']['al_bottom_extensions']['gas_contact_law_source']
        old = next(c for c in self.source['gas_network']['thermal_contacts'] if c['port_id'] == ident)
        area = next(r['gas_exposed_area_m2'] for r in self.source['metadata']['gas_access_geometry']['backing_exposure'] if r['port'] == ident)
        ports = {p['id']: p for p in new['ports']}
        for c in new['gas_network']['thermal_contacts']:
            if ':bottom-extension' not in c['port_id']:
                continue
            for key in ('solid_surface_connectivity', 'gas_surface_connectivity'):
                self.assertEqual(old[key], c[key])
            for key, value in old['connectivity'].items():
                if 'conductance_w_per_k' in key:
                    self.assertAlmostEqual(c['connectivity'][key]/ports[c['port_id']]['area_m2'], value/area)
                else:
                    self.assertEqual(c['connectivity'][key], value)

    def test_invalid_or_double_extension_rejected(self):
        for length in (-.001, .02, float('nan')):
            with self.assertRaises(ValueError):
                candidate(self.source, length)
        with self.assertRaises(ValueError):
            candidate(candidate(self.source, .005), .005)


if __name__ == '__main__':
    unittest.main()
