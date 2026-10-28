import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fsri30_gas_mobility import with_internal_resistance


class GasMobilityTest(unittest.TestCase):
    def test_only_internal_links_are_changed_without_mutating_input(self):
        internal = dict(id='a', kind='passage', **{'from': 'aligned-inner-1', 'to': 'aligned-inner-2'}, hydraulic_resistance_factor=2.)
        vent = dict(id='v', kind='passage', **{'from': 'aligned-inner-1', 'to': 'air-1'})
        rupture = dict(id='r', kind='rupture', **{'from': 'battery', 'to': 'aligned-inner-1'})
        original = dict(metadata={}, gas_network=dict(channels=[internal, vent, rupture]))
        result = with_internal_resistance(original, 4.)
        self.assertEqual(result['gas_network']['channels'][0]['hydraulic_resistance_factor'], 8.)
        self.assertEqual(original['gas_network']['channels'][0]['hydraulic_resistance_factor'], 2.)
        self.assertEqual(result['gas_network']['channels'][1:], [vent, rupture])
        self.assertEqual(result['metadata']['internal_gas_resistance_probe']['channels'], 1)
        with self.assertRaises(ValueError): with_internal_resistance(result, 2.)

    def test_invalid_values(self):
        for value in (0., -1., math.inf, math.nan):
            with self.assertRaises(ValueError): with_internal_resistance({}, value)


if __name__ == '__main__': unittest.main()
