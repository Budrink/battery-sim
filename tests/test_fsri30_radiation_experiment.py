from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_fsri30_radiation_experiment import decision, prefix


def example(temperature,end):
    times=np.arange(0.,end+.1,5.)
    run=dict(time_s=times.tolist(),cell_mean_temperature_k={
        f'B{i:02d}':np.full(len(times),temperature).tolist()for i in range(30)})
    data=dict(time_s=np.arange(4001,dtype=float),runaway=np.arange(4001)==1700)
    for i in range(1,10):data[f'T{i}_k']=np.full(4001,500.)
    return run,data


class RadiationReplayGuardTests(unittest.TestCase):
    def test_saved_prefix_cannot_be_extrapolated(self):
        run,_=example(500.,1200.)
        self.assertEqual(prefix(run,1000.)['time_s'][-1],1000.)
        with self.assertRaisesRegex(ValueError,'does not cover'):prefix(run,1205.)

    def test_known_warmup_failure_labelled_not_called_agreement(self):
        old,data=example(850.,1200.);run,_=example(855.,1200.)
        stop,inherited=decision({},run,old,data)
        self.assertIsNone(stop)
        self.assertIn('not accepted calibration',inherited['note'])
        self.assertGreater(inherited['candidate_rmse_k'],inherited['reference_rmse_k'])

    def test_new_warmup_divergence_and_absolute_safety_still_stop(self):
        old,data=example(850.,1200.)
        for temperature in (1000.,4000.):
            run,_=example(temperature,1200.)
            stop,inherited=decision({},run,old,data)
            self.assertIsNotNone(stop);self.assertIsNone(inherited)

    def test_cooling_rejection_is_not_exempted_by_bad_reference(self):
        old,data=example(650.,3900.);run,_=example(650.,3900.)
        stop,inherited=decision({},run,old,data)
        self.assertIn('Cooling prefix',stop['reason']);self.assertIsNone(inherited)


if __name__=='__main__':unittest.main()
