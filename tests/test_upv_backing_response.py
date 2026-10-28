import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from upv_backing_response import candidate
from upv_gas_access import candidate as geometry_candidate


class BackingResponse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / 'data/calibration/upv-separated-chemistry-v6/gas-access-refinement-20261022-r4-fresh'
        if not root.exists():
            raise unittest.SkipTest('Optional latest local reference')
        cls.source = json.loads((root / 'trial-000/scenario.json').read_text())
        cls.physical = json.loads((root / 'physical-source.json').read_text())

    def test_only_existing_al_contact_areas_change(self):
        before = copy.deepcopy(self.source)
        new = candidate(self.source, .1)
        ports = {p['id']: p for p in self.source['ports']}
        changed_owners = set()
        for old_contact, new_contact in zip(self.source['gas_network']['thermal_contacts'], new['gas_network']['thermal_contacts']):
            owner = ports[old_contact['port_id']]['cell_id']
            if old_contact != new_contact:
                self.assertIn(owner, ('Al1', 'Al2', 'Al3'))
                changed_owners.add(owner)
                for key in old_contact:
                    if key != 'connectivity':
                        self.assertEqual(old_contact[key], new_contact[key])
                for key in old_contact['connectivity']:
                    if 'conductance_w_per_k' not in key:
                        self.assertEqual(old_contact['connectivity'][key], new_contact['connectivity'][key])
        # Al2 is occluded on BOTH rims: backing alone cannot open that route.
        self.assertEqual(changed_owners, {'Al1', 'Al3'})
        for key in self.source['gas_network']:
            if key != 'thermal_contacts':
                self.assertEqual(self.source['gas_network'][key], new['gas_network'][key])
        for key in self.source:
            if key not in ('gas_network', 'metadata'):
                self.assertEqual(self.source[key], new[key])
        self.assertEqual(before, self.source)

    def test_reconstructs_unmasked_geometry_without_area_compounding(self):
        g = self.source['metadata']['gas_access_geometry']
        import itertools
        for fraction, lip in itertools.product([.001, .01, .03, .1, 1.], [.001, .01, .1]):
            expected = geometry_candidate(self.physical, g['protected_accessible_fraction'],
                end=self.source['simulation']['duration_s'], lip_open_fraction=lip,
                backing_open_fraction=fraction, witness_edge_open_fraction=g['witness_edge_open_fraction'],
                outer_insulation_open_fraction=g['outer_insulation_open_fraction'],
                jet_wetted_fraction=g['jet_wetted_fraction'], iss3_accessible_fraction=g.get('iss3_accessible_fraction'),
                open_iss_pad_wetted_fraction=g.get('open_iss_pad_wetted_fraction', 1.))
            new = candidate(self.source, fraction, lip)
            for a, b in zip(expected['gas_network']['thermal_contacts'], new['gas_network']['thermal_contacts']):
                for key in a['connectivity']:
                    if isinstance(a['connectivity'][key], (float, int)):
                        self.assertAlmostEqual(a['connectivity'][key], b['connectivity'][key], places=10)
                    else:
                        self.assertEqual(a['connectivity'][key], b['connectivity'][key])
        self.assertEqual(candidate(self.source, .01)['gas_network'], self.source['gas_network'])
        for lip in [0., float('nan'), 1.1]:
            with self.assertRaises(ValueError):
                candidate(self.source, .1, lip)

    def test_rejects_invalid_fraction(self):
        for fraction in [0., -.1, 1.1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                candidate(self.source, fraction)


if __name__ == '__main__':
    unittest.main()
