import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from check_upv_contact_ablation import without_contact, verify_disabled_heat, SCALES


class ContactAblation(unittest.TestCase):
    def fixture(self):
        def contact(port):
            return dict(compartment='gas', port_id=port, connectivity=dict(kind='directed_geometry',
                geometry_reference_logit=-3., **{key: 2. for key in SCALES}),
                gas_surface_connectivity=dict(reference_temperature_k=298.15))
        return dict(cells=['untouched'], shells=['untouched'], interfaces=['untouched'],
            ports=[dict(id=p, cell_id=c) for p, c in [('case', 'C3:case:-x'), ('plate', 'steel-L'), ('pad', 'ISS1')]],
            gas_network=dict(channels=['unchanged'], compartments=['unchanged'],
                thermal_contacts=[contact('case'), contact('plate'), contact('pad')]))

    def test_only_target_conductance_is_zeroed(self):
        scene = self.fixture(); original = copy.deepcopy(scene)
        result, manifest = without_contact(scene, 'C3:case:-x')
        self.assertEqual(scene, original)
        self.assertEqual(len(manifest['contacts']), 1)
        self.assertEqual(result['gas_network']['thermal_contacts'][1:], scene['gas_network']['thermal_contacts'][1:])
        for key in SCALES:
            self.assertEqual(result['gas_network']['thermal_contacts'][0]['connectivity'][key], 0.)
        for key in ('cells', 'shells', 'interfaces', 'ports'):
            self.assertEqual(result[key], scene[key])
        self.assertEqual(result['gas_network']['channels'], scene['gas_network']['channels'])

    def test_independent_plate_case_restores_casing(self):
        scene = self.fixture()
        without_contact(scene, 'C3:case:-x')
        result, _ = without_contact(scene, 'steel-L')
        self.assertEqual(result['gas_network']['thermal_contacts'][0], scene['gas_network']['thermal_contacts'][0])
        self.assertEqual(result['gas_network']['thermal_contacts'][2], scene['gas_network']['thermal_contacts'][2])

    def test_disabled_contact_actual_heat_is_verified(self):
        row = dict(solid='steel-L', heat_to_solid_j=[0., 0.], positive_step_heat_j=[0., 0.], negative_step_heat_j=[0., 0.])
        run = dict(gas_film_trace={'film': row})
        self.assertEqual(verify_disabled_heat({}, run, 'steel-L')['max_absolute_heat_j'], 0.)
        row['positive_step_heat_j'][1] = 1.
        with self.assertRaises(ValueError): verify_disabled_heat({}, run, 'steel-L')

    def test_wrong_or_missing_target_rejected(self):
        with self.assertRaises(ValueError): without_contact(self.fixture(), 'ISS1')
        scene = self.fixture(); scene['gas_network']['thermal_contacts'] = []
        with self.assertRaises(ValueError): without_contact(scene, 'steel-L')


if __name__ == '__main__': unittest.main()
