import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from upv_gas_route_refinement import candidate, assessment


class GasRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'data/calibration/upv-separated-chemistry-v6/damaged-contact-20261022-r1/trial-002/scenario.json'
        if not path.exists():
            raise unittest.SkipTest('Optional current local geometry')
        cls.source = json.loads(path.read_text())

    def test_neutral_is_physically_exact_and_source_not_mutated(self):
        before = copy.deepcopy(self.source)
        new = candidate(self.source)
        for key in before:
            if key != 'metadata':
                self.assertEqual(before[key], new[key])
        self.assertEqual(before, self.source)

    def test_routes_keep_topology_and_solid_physics(self):
        new = candidate(self.source, 3., .0135, .25)
        for key in self.source:
            if key not in ('metadata', 'gas_network'):
                self.assertEqual(self.source[key], new[key], key)
        a, b = self.source['gas_network'], new['gas_network']
        for key in a:
            if key not in ('channels', 'thermal_contacts'):
                self.assertEqual(a[key], b[key], key)
        for before, after in zip(a['channels'], b['channels']):
            for key in before:
                if key not in ('area_m2', 'mixing_length_m', 'discharge_coefficient'):
                    self.assertEqual(before[key], after[key])
            if before['id'].startswith('escape-ISS2-'):
                self.assertAlmostEqual(after['area_m2']/before['area_m2'], 3.)
                self.assertAlmostEqual(before['mixing_length_m']/after['mixing_length_m'], 3.)
            elif before['id'].startswith('escape-ISS3-'):
                self.assertAlmostEqual(after['area_m2']/before['area_m2'], 3.)
            elif before['kind'] == 'rupture' and before['to'].startswith('gap-ISS3-'):
                self.assertAlmostEqual(after['discharge_coefficient']/before['discharge_coefficient'], 3.)
            else:
                self.assertEqual(before, after)
        ports = {p['id']: p['cell_id'] for p in self.source['ports']}
        for before, after in zip(a['thermal_contacts'], b['thermal_contacts']):
            owner = ports[before['port_id']]
            selected = owner in ('ISS1', 'ISS2') and before['compartment'].startswith('gap-' + owner + '-')
            if not selected:
                self.assertEqual(before, after)
                continue
            for key in before:
                if key != 'connectivity':
                    self.assertEqual(before[key], after[key])
            for key, value in before['connectivity'].items():
                factor = .5 if 'conductance_w_per_k' in key else 1.
                self.assertEqual(after['connectivity'][key], value*factor if isinstance(value,(int,float)) else value)

    def test_invalid_geometry_rejected(self):
        for scale, mouth, pad in [(0., .01, .5), (11., .01, .5), (1., .1, .5),
                                  (1., .01, .1), (float('nan'), .01, .5)]:
            with self.assertRaises(ValueError):
                candidate(self.source, scale, mouth, pad)


class ArrivalObjective(unittest.TestCase):
    def score(self, delay):
        observed = dict(t10_s=886., t90_s=887.)
        predicted = dict(t10_s=886.+delay, t90_s=887.+delay)
        result = dict(objective=10., accepted=True,
            front=dict(features={'pressure':dict(predicted=predicted,observed=observed)}),
            shape=dict(shape_with_front_objective=2.))
        with patch('upv_gas_route_refinement.assess', return_value=result):
            return assessment({}, {}, 1100.)

    def test_delay_cannot_be_hidden_by_matching_tail_or_peak(self):
        a, b = self.score(0.), self.score(2.)
        self.assertTrue(a['accepted'])
        self.assertFalse(b['accepted'])
        self.assertEqual(b['objective']-a['objective'], 16.)
        self.assertEqual(b['arrival_times']['pressure']['error_s'], 2.)


if __name__ == '__main__':
    unittest.main()
