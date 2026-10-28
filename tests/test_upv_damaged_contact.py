import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from upv_damaged_contact import candidate, hot_contact_candidate


class DamagedContact(unittest.TestCase):
    def setUp(self):
        self.source = dict(metadata={}, ports=[dict(id='a', cell_id='battery'),
            dict(id='b', cell_id='pad'), dict(id='c', cell_id='insulator')],
            cells=[dict(id='battery', material_id='battery'), dict(id='pad', material_id='TIM'),
                   dict(id='insulator', material_id='ISS')],
            interfaces=[dict(id=name, port_a='a', port_b=port, law=dict(kind='directed_geometry',
                background_conductance_w_per_k=.04, active_conductance_w_per_k=7.92,
                geometry_reference_logit=0., destroyed_background_conductance_w_per_k=.08,
                destroyed_active_conductance_w_per_k=0.)) for name, port in [('TIM', 'b'), ('ISS', 'c')]])

    def test_neutral_is_exact_and_only_destroyed_tim_branch_changes(self):
        before = copy.deepcopy(self.source)
        self.assertEqual(candidate(self.source, .02)['interfaces'], self.source['interfaces'])
        changed = candidate(self.source, .5)
        self.assertEqual(changed['interfaces'][0]['law']['destroyed_background_conductance_w_per_k'], 2.)
        expected = copy.deepcopy(before)
        expected['interfaces'][0]['law']['destroyed_background_conductance_w_per_k'] = 2.
        changed.pop('metadata'); expected.pop('metadata')
        self.assertEqual(changed, expected)
        self.assertEqual(self.source, before)

    def test_bad_fraction_or_unexpected_destroyed_branch_is_rejected(self):
        for fraction in [0., 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                candidate(self.source, fraction)
        self.source['interfaces'][0]['law']['destroyed_active_conductance_w_per_k'] = 1.
        with self.assertRaises(ValueError):
            candidate(self.source, .5)

    def test_hot_reference_is_bounded_by_intact_law_ceiling(self):
        before = copy.deepcopy(self.source)
        changed = candidate(self.source, 1., reference='hot_ceiling')
        law = changed['interfaces'][0]['law']
        self.assertEqual(law['destroyed_background_conductance_w_per_k'], 7.96)
        self.assertEqual(changed['interfaces'][1], before['interfaces'][1])
        self.assertEqual(self.source, before)
        with self.assertRaises(ValueError):
            candidate(self.source, .5, reference='unknown')

    def test_hot_contact_bracket_changes_no_bulk_or_intact_law(self):
        changed = hot_contact_candidate(self.source, 4.)
        expected = copy.deepcopy(self.source)
        expected['interfaces'][0]['law']['destroyed_background_conductance_w_per_k'] = 4.*7.96
        self.assertEqual(changed['interfaces'], expected['interfaces'])
        for key in self.source:
            if key != 'metadata':
                self.assertEqual(changed[key], expected[key])
        for scale in [.1, 4.1, float('nan')]:
            with self.assertRaises(ValueError):
                hot_contact_candidate(self.source, scale)

    def test_current_profile_selects_all_three_equal_material_pairs(self):
        path = Path(__file__).resolve().parents[1] / 'data/calibration/upv-separated-chemistry-v6/gas-response-refinement-20261022-r1/trial-005/scenario.json'
        if not path.exists():
            self.skipTest('Optional current local profile')
        source = json.loads(path.read_text())
        neutral = candidate(source, .02)
        self.assertEqual(neutral['interfaces'], source['interfaces'])
        changed = candidate(source, .5)
        selected = [a['id'] for a, b in zip(source['interfaces'], changed['interfaces']) if a != b]
        self.assertEqual(set(selected), {'C1--TIM3', 'C2--TIM2', 'C3--TIM1'})
        for key in source:
            if key not in ('interfaces', 'metadata'):
                self.assertEqual(source[key], changed[key])


if __name__ == '__main__':
    unittest.main()
