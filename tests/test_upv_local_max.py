import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from check_upv_local_max import local_maximum


class LocalMaximum(unittest.TestCase):
    def fixture(self):
        law = {k: 2. for k in ('background_conductance_w_per_k', 'active_conductance_w_per_k',
            'destroyed_background_conductance_w_per_k', 'destroyed_active_conductance_w_per_k')}
        def contact(gas, port):
            return dict(compartment=gas, port_id=port, connectivity=copy.deepcopy(law))
        def edge(name, a, b, kind='passage'):
            return dict(id=name, kind=kind, **{'from': a, 'to': b}, area_m2=.01,
                mixing_length_m=.1, discharge_coefficient=.7)
        return dict(cells=['frozen'], shells=['frozen'], gas_network=dict(
            compartments=[dict(id='gap-ISS1-0'), dict(id='gallery'), dict(id='elsewhere')],
            channels=[edge('inlet', 'gap-ISS1-0', 'gallery'), edge('supply', 'gallery', 'hot'),
                      edge('unrelated', 'elsewhere', 'hot'), edge('shell', 'inside', 'gallery', 'rupture')],
            thermal_contacts=[contact('gap-ISS1-0', 'case'), contact('gallery', 'opening-contact:ISS1:rim'),
                              contact('elsewhere', 'other')]))

    def test_only_local_passages_and_contacts_change(self):
        scene = self.fixture(); before = copy.deepcopy(scene)
        result, manifest = local_maximum(scene)
        self.assertEqual(scene, before)
        self.assertEqual([e['id'] for e in manifest['channels']], ['inlet', 'supply'])
        self.assertEqual(result['gas_network']['channels'][2:], scene['gas_network']['channels'][2:])
        self.assertEqual(result['gas_network']['thermal_contacts'][2], scene['gas_network']['thermal_contacts'][2])
        self.assertEqual(result['cells'], scene['cells']); self.assertEqual(result['shells'], scene['shells'])
        self.assertAlmostEqual(result['gas_network']['channels'][0]['area_m2'], 10.)
        self.assertAlmostEqual(result['gas_network']['channels'][0]['mixing_length_m'], .0001)
        self.assertEqual(result['gas_network']['thermal_contacts'][0]['connectivity']['active_conductance_w_per_k'], 2000.)

    def test_mass_only_preserves_all_thermal_contacts(self):
        scene = self.fixture(); result, _ = local_maximum(scene, heat_factor=1.)
        self.assertEqual(result['gas_network']['thermal_contacts'], scene['gas_network']['thermal_contacts'])

    def test_missing_target_and_nonfinite_multiplier_rejected(self):
        scene = self.fixture()
        for factor in (float('nan'), float('inf'), 0., -1.):
            with self.assertRaises(ValueError): local_maximum(scene, heat_factor=factor)
        scene['gas_network']['compartments'] = []
        with self.assertRaises(ValueError): local_maximum(scene)


if __name__ == '__main__': unittest.main()
