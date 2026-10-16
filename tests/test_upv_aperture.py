"""Calibration cannot absorb post-failure errors into measured solid properties."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import calibrate_upv_aperture as fit
import check_upv_pressure_feedback as feedback


class ApertureCalibrationContract(unittest.TestCase):
    def test_geometric_replay_removes_only_area_ceiling(self):
        scene, _ = fit.shell.inputs('no-cooling')
        baked = fit.bake(scene, fit.initial_parameters())
        original = copy.deepcopy(baked)
        replay = feedback.prepare(baked, [0., 1.])
        self.assertEqual(baked, original)
        for key in ('cells', 'shells', 'interfaces', 'ports', 'boundaries', 'measurement_probes'):
            self.assertEqual(replay[key], original[key])
        for a, b in zip(original['gas_network']['channels'], replay['gas_network']['channels']):
            expected = dict(a); expected.pop('area_m2')
            self.assertEqual(b, expected)

    def test_frozen_geometry_and_materials(self):
        scene, _ = fit.shell.inputs('no-cooling')
        original = copy.deepcopy(scene)
        parameters = fit.initial_parameters()
        baked = fit.bake(scene, parameters)
        self.assertEqual(scene, original)
        for key in ('shells', 'interfaces', 'ports', 'boundaries', 'measurement_probes'):
            self.assertEqual(baked[key], original[key])
        for a, b in zip(baked['cells'], original['cells']):
            for key in ('thermal', 'geometry', 'pose', 'degradation'):
                self.assertEqual(a.get(key), b.get(key))
        self.assertEqual(len(baked['gas_network']['channels']), 6)
        for channel in baked['gas_network']['channels']:
            self.assertEqual(channel['surface_link']['h_w_per_m2_k'], parameters['surface_h'])
            self.assertEqual(channel['damage_opening']['fracture_energy_j_per_m2'], parameters['fracture_work_j_m2'])

    def test_unrequested_battery_knob_is_rejected(self):
        scene, _ = fit.shell.inputs('no-cooling')
        parameters = dict(fit.initial_parameters(), battery_cp=500)
        with self.assertRaises(ValueError):
            fit.bake(scene, parameters)

    def test_legacy_ablation_has_no_new_transfer_or_work(self):
        scene, _ = fit.shell.inputs('no-cooling')
        baked = fit.bake(scene, fit.initial_parameters(), enabled=False)
        for channel in baked['gas_network']['channels']:
            self.assertNotIn('surface_link', channel)
            self.assertNotIn('fracture_energy_j_per_m2', channel['damage_opening'])


if __name__ == '__main__':
    unittest.main()
