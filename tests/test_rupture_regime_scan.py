import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import scan_upv_rupture_regimes as scan


class RegimeScanTests(unittest.TestCase):
    def test_fixed_inventory_and_frozen_seed(self):
        seed = scan.model.initial_parameters()
        before = copy.deepcopy(seed)
        for case in scan.DEFAULT_CASES:
            result = scan.parameters(seed, case)
            self.assertAlmostEqual(result['solid_energy_j']+result['gas_energy_j'],
                                   seed['solid_energy_j']+seed['gas_energy_j'])
        self.assertEqual(seed, before)

    def test_no_capacity_area_or_inventory_knob(self):
        for key in ('total_area_m2', 'battery_cp', 'total_energy_j'):
            with self.assertRaises(ValueError):
                scan.parameters(scan.model.initial_parameters(), dict(id='bad', **{key: 1.}))

    def test_bad_controls_rejected(self):
        for change in (dict(gas_fraction=1.), dict(gas_fraction=float('nan')),
                       dict(rate_factor=-1.), dict(exposure_factor=float('inf')),
                       dict(id='../escape')):
            with self.assertRaises(ValueError):
                scan.parameters(scan.model.initial_parameters(), dict(dict(id='bad'), **change))


if __name__ == '__main__':
    unittest.main()
