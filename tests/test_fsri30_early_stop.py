from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_early_stop import rejection


def example(offset=0.,end=3900.):
    t=np.arange(4001,dtype=float)
    data=dict(time_s=t,runaway=t==1700.)
    for i in range(1,10):data[f'T{i}_k']=np.full(len(t),500.)
    times=np.arange(0.,end+.1,5.)
    run=dict(time_s=times.tolist(),cell_mean_temperature_k={f'B{i:02d}':np.full(len(times),500.+offset).tolist()for i in range(30)})
    return run,data


class EarlyStopTests(unittest.TestCase):
    def test_persistent_cooling_error_rejected_before_full_horizon(self):
        run,data=example(150.)
        decision=rejection({},run,data)
        self.assertIn('Cooling prefix',decision['reason'])
        self.assertEqual(decision['rmse_k'],150.)
        self.assertEqual(decision['limit_k'],80.)
        self.assertIsNone(rejection({},*example(5.)))

    def test_no_unobserved_future_or_premature_cooling_claim(self):
        self.assertIsNone(rejection({},*example(150.,3650.)))

    def test_peaks_are_softened_not_erased(self):
        run,data=example()
        for i in range(1,10):data[f'T{i}_k'][3870:3901]=600.
        self.assertIsNone(rejection({},run,data))
        for i in range(1,10):data[f'T{i}_k'][3870:3901]=1000.
        decision=rejection({},run,data)
        self.assertIn('softened',decision['reason'])
        self.assertEqual(decision['limit_k'],250.)

    def test_one_noisy_sensor_cannot_exempt_stable_overheated_branches(self):
        run,data=example(150.)
        for start in range(3780,3901,20):data['T1_k'][start:start+10]=1000.
        decision=rejection({},run,data)
        self.assertIn('Cooling prefix',decision['reason'])
        self.assertGreaterEqual(decision['stable_quantile_entries'],240)


if __name__=='__main__':unittest.main()
