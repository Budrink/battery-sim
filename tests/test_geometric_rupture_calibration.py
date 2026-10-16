"""Calibration must not recover a hidden rupture ceiling or alter solid data."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import calibrate_upv_geometric_rupture as fit


class GeometricRuptureCalibration(unittest.TestCase):
    def setUp(self):
        self.scene, _ = fit.inputs('no-cooling')
        self.parameters = fit.initial_parameters()

    def test_frozen_materials_and_geometry(self):
        original = copy.deepcopy(self.scene)
        result = fit.bake(self.scene, self.parameters)
        self.assertEqual(self.scene, original)
        for a, b in zip(original['cells'], result['cells']):
            for key in ('thermal', 'geometry', 'pose', 'degradation'):
                self.assertEqual(a.get(key), b.get(key))
        for key in ('shells', 'interfaces', 'ports', 'boundaries', 'measurement_probes'):
            self.assertEqual(original[key], result[key])
        self.assertEqual(original['gas_network']['compartments'], result['gas_network']['compartments'])

    def test_no_area_ceiling_and_single_energy_inventory(self):
        result = fit.bake(self.scene, self.parameters)
        for ch in result['gas_network']['channels']:
            self.assertNotIn('area_m2', ch)
            self.assertEqual(ch['opening_pressure_pa'], self.parameters['strength_pa'])
            self.assertEqual(ch['damage_opening']['pressure_damage_rate_per_s'], self.parameters['damage_rate_per_s'])
            self.assertEqual(ch['damage_opening']['fracture_energy_j_per_m2'], fit.FIXED_WORK)
        source = next(c['conversion'] for c in result['cells'] if c['id']=='C2')
        total = self.parameters['solid_energy_j']+self.parameters['gas_energy_j']
        self.assertAlmostEqual(source['mass_kg']*source['net_heat_j_per_kg'], total)
        self.assertAlmostEqual(source['gas_chemical_fraction']*total, self.parameters['gas_energy_j'])

    def test_exposure_is_absolute_not_compounded(self):
        result = fit.bake(self.scene, self.parameters)
        for a, b in zip(self.scene['gas_network']['thermal_contacts'], result['gas_network']['thermal_contacts']):
            if a['port_id'] != 'wall:inner':
                self.assertAlmostEqual(a['h_w_per_m2_k'], b['h_w_per_m2_k'])

    def test_undeclared_knobs_and_nonfinite_values_rejected(self):
        for key in ('total_area_m2', 'battery_cp', 'battery_k', 'gas_volume_m3'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                fit.bake(self.scene, dict(self.parameters, **{key:1.}))
        for value in (float('nan'), float('inf'), 0., -1.):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fit.bake(self.scene, dict(self.parameters, strength_pa=value))


if __name__ == '__main__':
    unittest.main()
