import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_burst_fit import assess,objective
from fsri30_cooling_fit import burst_rank


def fixture(shift=0.,width=30.,amplitude=900.,end=3600.,converted=30):
    t=np.arange(5946,dtype=float);data=dict(time_s=t,runaway=t==1700.)
    measured=300.+900.*np.exp(-.5*((t-2580.)/30.)**2)
    for i in range(1,10):data[f'T{i}_k']=measured.copy()
    times=np.arange(0.,end+.1,5.);v=300.+amplitude*np.exp(-.5*((times-2580.-shift)/width)**2)
    scene=dict(cells=[]);run=dict(time_s=times.tolist(),cell_mean_temperature_k={},chemistry={})
    for i in range(30):
        key=f'B{i:02d}';scene['cells'].append(dict(id=key,role='battery',conversion=dict(mass_kg=.025,gas_yield=.33)))
        run['cell_mean_temperature_k'][key]=v.tolist()
        run['chemistry'][key]=dict(r2_generated_gas_kg=[.025*.33 if i<converted else 0.])
    return scene,run,data


class BurstGateTests(unittest.TestCase):
    def test_near_ten_second_match_is_accepted(self):
        b=assess(*fixture(shift=5.))
        self.assertTrue(b['accepted']);self.assertTrue(b['timing_target_met']);self.assertIsNotNone(objective(b))

    def test_more_than_hundred_seconds_early_or_late_is_rejected(self):
        for shift in (-130.,130.):
            b=assess(*fixture(shift=shift))
            self.assertFalse(b['accepted']);self.assertIn('timing error',b['reason'])

    def test_intermediate_delay_is_admissible_but_not_target(self):
        b=assess(*fixture(shift=50.))
        self.assertTrue(b['accepted']);self.assertFalse(b['timing_target_met']);self.assertGreater(objective(b),0.)

    def test_no_cascade_and_no_collective_peak_are_rejected(self):
        for kwargs in (dict(converted=1),dict(amplitude=10.)):
            b=assess(*fixture(**kwargs));self.assertFalse(b['accepted'])

    def test_gradual_ramp_cannot_replace_a_sharp_burst(self):
        b=assess(*fixture(width=250.))
        self.assertFalse(b['accepted']);self.assertIn('too gradual',b['reason'])

    def test_uncertain_peak_height_is_softened_but_not_ignored(self):
        self.assertTrue(assess(*fixture(amplitude=1050.))['accepted'])
        self.assertFalse(assess(*fixture(amplitude=1900.))['accepted'])

    def test_uncomputed_future_remains_pending_without_an_observed_failure(self):
        b=assess(*fixture(end=2200.))
        self.assertFalse(b['evaluated']);self.assertIsNone(b['accepted']);self.assertIsNone(objective(b))

    def test_observed_early_primary_cannot_be_rescued_by_a_future_peak(self):
        b=assess(*fixture(shift=-200.,end=2500.))
        self.assertTrue(b['evaluated']);self.assertFalse(b['accepted']);self.assertIsNone(objective(b))

    def test_bad_explosion_cannot_win_by_better_cooling_or_longer_run(self):
        good=dict(burst=dict(accepted=True,phase_window_complete=True,timing_error_s=5.),
            full_trajectory=False,anchor_rmse_k=150.,last_observed_s=3720.)
        bad=dict(burst=dict(accepted=False,phase_window_complete=True,timing_error_s=200.),
            full_trajectory=True,full_cooling_rmse_k=0.,last_observed_s=5945.)
        delayed=dict(good,burst=dict(accepted=True,phase_window_complete=True,timing_error_s=65.),anchor_rmse_k=0.)
        self.assertGreater(burst_rank(good),burst_rank(bad));self.assertGreater(burst_rank(good),burst_rank(delayed))


if __name__=='__main__':unittest.main()
