import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from refine_upv_tim_cooling import candidate_rank
from upv_observation_policy import CALIBRATION_CHANNELS


class CoolingRank(unittest.TestCase):
    def setUp(self):
        self.row=dict(arrival_accepted=True,
            preserved_front=dict(pressure_peak_relative_error=.01),
            channels={k:dict(rmse_k=4.,tolerance_k=5.)for k in CALIBRATION_CHANNELS},
            cooling_objective=3.2,objective=50.)

    def test_peak_improvement_cannot_hide_worse_cooling(self):
        other=copy.deepcopy(self.row)
        other['channels']['C2_TIM']['rmse_k']=20.
        other.update(objective=1.,cooling_objective=2.)
        self.assertLess(candidate_rank(self.row),candidate_rank(other))

    def test_ignored_channel_has_no_ranking_effect(self):
        other=copy.deepcopy(self.row)
        other['channels']['C2_ISS']=dict(rmse_k=4000.,tolerance_k=.01)
        self.assertEqual(candidate_rank(self.row),candidate_rank(other))

    def test_pressure_or_arrival_failure_is_not_a_cooling_gain(self):
        for field in ('pressure','arrival'):
            other=copy.deepcopy(self.row)
            other.update(objective=0.,cooling_objective=0.)
            for channel in other['channels'].values():channel['rmse_k']=0.
            if field=='pressure':other['preserved_front']['pressure_peak_relative_error']=.1
            else:other['arrival_accepted']=False
            self.assertLess(candidate_rank(self.row),candidate_rank(other))


if __name__=='__main__':unittest.main()
