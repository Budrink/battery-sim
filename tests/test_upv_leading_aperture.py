from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import replay_upv_leading_aperture as replay


class LeadingAperture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before, _ = replay.old.geometry.inputs('no-cooling')
        cls.after = replay.build(cls.before, .2, 5e-5)

    def test_frozen_physics(self):
        self.assertEqual(replay.old.physical_signature(self.before), replay.old.physical_signature(self.after))
        old = replay.old.geometry.attach(self.before, replay.old.geometry.initial())
        self.assertEqual([c.get('conversion') for c in old['cells']],
                         [c.get('conversion') for c in self.after['cells']])

    def test_matching_grids_and_local_routes(self):
        cells = {c['id']: c for c in self.after['cells']}
        for shell in self.after['shells']:
            grid = cells[shell['contents_id']]['thermal']['divisions']
            self.assertEqual(shell['surface_divisions'], grid)
            for face, field in shell['initial_temperature_fields_k'].items():
                self.assertEqual(len(field), math.prod(grid)//grid['xyz'.index(face[1])])
        channels = [c for c in self.after['gas_network']['channels'] if c['kind'] == 'rupture']
        self.assertEqual(len(channels), 54)
        for c in channels:
            self.assertTrue(0 <= c['shell_thermal_volume'] < 9)
            self.assertNotEqual(c['to'], 'chamber')
            self.assertEqual(c['damage_opening']['mode'], 'deterministic_aperture')
            self.assertNotIn('seed', c['damage_opening'])
        self.assertEqual(len({(c['port_id'], c['shell_thermal_volume']) for c in channels}), 54)

    def test_controls(self):
        self.assertEqual(self.after['simulation']['max_dt_s'], 5e-5)
        self.assertEqual(self.after['failure_sample_times_s'][-1], .2)
        self.assertFalse(self.after['metadata']['leading_aperture']['calibration_accepted'])
        for duration in (0., -1., float('nan'), 31.):
            with self.assertRaises(ValueError):
                replay.build(self.before, duration)


if __name__ == '__main__':
    unittest.main()
