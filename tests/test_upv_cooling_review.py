import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import upv_cooling_review as review


class CoolingReview(TestCase):
    def report(self, t, error, missing=False):
        data = {'time_s': np.array([0., 200.])}
        run = dict(time_s=t, temperature_k={'101': 300.+np.asarray(error)})
        def observed(_, at):
            target = np.full_like(at, 300.)
            if missing:
                target[(at >= 180.) & (at <= 190.)] = np.nan
            return {'101': target}
        with patch.object(review.dataio, 'event', return_value={'time_s': 100.}), \
             patch.object(review.dataio, 'CHANNELS', {'101': ('101',)}), \
             patch.object(review.dataio, 'observed', side_effect=observed):
            return review.phase_report(run, data)

    def test_dense_peak_does_not_reweight_cooling(self):
        t = np.unique(np.r_[np.arange(0., 201., 5.), np.arange(95., 120., .01)])
        error = np.where((t >= 95.) & (t <= 120.), 1000., 2.)
        r = self.report(t, error)
        self.assertTrue(r['complete_record'])
        self.assertAlmostEqual(r['phases']['cooling']['channels']['101']['time_weighted_rmse_k'], 2.)
        self.assertGreater(r['phases']['burst_uncertain']['channels']['101']['time_weighted_rmse_k'], 900.)

    def test_prefix_has_no_simulation_extrapolation(self):
        t = np.arange(170., 186.)
        r = self.report(t, np.full_like(t, 3.))
        self.assertFalse(r['complete_record'])
        self.assertEqual(r['phases']['cooling']['interval_s'], [170., 185.])
        self.assertFalse(r['phases']['cooling']['complete'])
        self.assertNotIn('burst_uncertain', r['phases'])

    def test_missing_scans_are_not_filled(self):
        t = np.arange(0., 201.)
        c = self.report(t, np.full_like(t, -3.), missing=True)['phases']['cooling']['channels']['101']
        self.assertLess(c['covered_s'], 40.)
        self.assertAlmostEqual(c['time_weighted_bias_k'], -3.)
        self.assertAlmostEqual(c['time_weighted_rmse_k'], 3.)

    def test_no_fabricated_cooling_from_two_distant_endpoints(self):
        r = self.report([0., 200.], [0., 0.])
        self.assertEqual(r['phases']['cooling']['channels'], {})
        self.assertEqual(r['max_observation_gap_s'], 200.)

    def test_kelvin_relative_error_is_distinct_from_heating_rise(self):
        t=np.arange(201.,dtype=float)
        def observed(_,at):return {'101':np.where(np.asarray(at)<10.,300.,350.)}
        run=dict(time_s=t,temperature_k={'101':np.where(t<10.,300.,345.)})
        with patch.object(review.dataio,'event',return_value={'time_s':100.}), \
             patch.object(review.dataio,'CHANNELS',{'101':('101',)}), \
             patch.object(review.dataio,'observed',side_effect=observed):
            report=review.phase_report(run,{'time_s':np.array([0.,200.])})
        row=report['phases']['cooling']['channels']['101']
        self.assertAlmostEqual(row['time_weighted_rmse_k'],5.)
        self.assertAlmostEqual(row['relative_kelvin_rms'],5./350.)
        self.assertAlmostEqual(row['rmse_over_observed_rise'],5./50.)
        self.assertNotAlmostEqual(row['relative_kelvin_rms'],5./(350.-273.15))

    def test_no_percentage_of_a_zero_heating_rise(self):
        row=self.report(np.arange(201.),np.ones(201))['phases']['cooling']['channels']['101']
        self.assertIsNone(row['rmse_over_observed_rise'])
