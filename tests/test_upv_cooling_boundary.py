import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import upv_cooling_boundary as cooling
import upv_r123_data as dataio

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/'data/calibration/upv-separated-chemistry-v6/final-training-calibration-20261022-r3/process-shape-min01-heldout-20261023-r1/scenario.json'


class CoolingBoundary(unittest.TestCase):
    def test_baseline_and_flow_calorimetry(self):
        t = np.array([0., 200., 450., 700.])
        data = dict(time_s=t, T110_k=t*0+300., T111_k=t*0+298.,
                    T112_k=np.array([297.98, 297.98, 297.98, 298.18]))
        values = cooling.measurements(data, [200., 700.])
        self.assertAlmostEqual(values['incremental_water_power_w'][0], 0.)
        self.assertAlmostEqual(values['incremental_water_power_w'][1], cooling.FLOW_HEAT_CAPACITY_W_K*.2)
        with self.assertRaisesRegex(ValueError, 'extrapolation'): cooling.measurements(data, [900.])

    def test_donor_direction_not_averaged(self):
        law = dict(background_conductance_w_per_k=.1, active_conductance_w_per_k=1.8)
        a = dict(linear=[0., 0., 1.], reference_temperature_k=300., temperature_scale_k=10.)
        forward = float(cooling.directed_g(law, a, {}, 320., 300., 0.))
        reverse = float(cooling.directed_g(law, a, {}, 300., 320., 0.))
        self.assertGreater(forward, reverse)
        self.assertAlmostEqual(reverse, 1.)

    def test_candidate_scope_and_native_surface_observer(self):
        if not CASE.exists(): self.skipTest('Local current cooled case is optional')
        source = json.loads(CASE.read_text()); before = copy.deepcopy(source)
        data, _ = dataio.load('active-cooling')
        trial = cooling.candidate(source, data, dict(contact_h_w_per_m2_k=20000., water_g_w_per_k=6.))
        self.assertEqual(source, before)
        for key in ('cells', 'shells', 'ports', 'gas_network', 'simulation', 'failure_sample_times_s'):
            self.assertEqual(trial[key], source[key], key)
        for old, new in zip(source['interfaces'], trial['interfaces']):
            if old['id'] not in cooling.COOLING_CONTACTS: self.assertEqual(old, new)
        for old, new in zip(source['boundaries'], trial['boundaries']):
            if old['id'] != 'water': self.assertEqual(old, new)
        self.assertEqual(trial['measurement_probes'][:-1], source['measurement_probes'])
        self.assertEqual(trial['measurement_probes'][-1], dict(id='110', cell_id='Al2', interface_id='Al2--cooling'))
        water = next(b for b in trial['boundaries'] if b['id'] == 'water')
        self.assertEqual(water['temperature_profile'][0][0], 0.)
        self.assertEqual(water['temperature_profile'][-1][0], trial['simulation']['duration_s'])
        self.assertTrue(trial['metadata']['frozen_validation']['cooling_boundary_only_refitted'])
        self.assertFalse(trial['metadata']['cooling_boundary_identification']['chemistry_refitted'])
        self.assertAlmostEqual(cooling.half_resistances(trial)[0], 1.1433113230618732)

    def test_objective_never_uses_other_channels(self):
        t = np.arange(650., 1251., 5.)
        observed = dict(contact_k=t*0+310., incremental_water_power_w=t*0+20.,
                        C2_TIM=np.full(len(t), np.nan), pressure_bar=np.full(len(t), np.nan))
        residual = cooling.objective(t*0+310., t*0+20., observed, t)
        self.assertTrue(np.all(residual == 0.))

    def test_native_prediction_cannot_be_overwritten_by_measured_power(self):
        if not CASE.exists(): self.skipTest('Local current cooled case is optional')
        scene = json.loads(CASE.read_text())
        t = np.array([0., 200., 450., 700.])
        data = dict(time_s=t, T110_k=t*0+310., T111_k=t*0+298.,
                    T112_k=np.array([297.98, 297.98, 297.98, 298.18]))
        run = dict(time_s=t, temperature_k={}, thermal_paths=dict(
            interfaces={'Al2--cooling': dict(a_adjacent_temperature_k=t*0+310., a_to_b_power_w=t*0+2.)},
            objects={'cooling-plate': dict(reference_capacity_weighted_temperature_k=t*0+300.)}))
        values = cooling.native_observations(scene, run, data)
        self.assertAlmostEqual(values['incremental_water_power_w'][-1], 0.)
        self.assertGreater(cooling.measurements(data,t)['incremental_water_power_w'][-1], 20.)
