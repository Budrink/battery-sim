import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upv_r123_data as dataio
import upv_cooling_fit as cooling
import upv_destruction_objective as destruction
import upv_gas_access as gas
from refine_upv_gas_access import temperature_pruner
from upv_observation_policy import CALIBRATION_CHANNELS


class InformationalC2Iss(unittest.TestCase):
    def setUp(self):
        self.times=np.arange(0.,1101.,.25)
        self.data={'time_s':self.times}
        self.event={'time_s':887.}
        self.meta=dict(data=self.data,event=self.event,fit_end_s=1100.)
        self.run=dict(time_s=self.times.tolist(),
                      temperature_k={k:np.full(len(self.times),300.).tolist() for k in dataio.CHANNELS})

    def observed(self,_data,t):
        return dict({k:np.full(len(t),300.) for k in dataio.CHANNELS},pressure_bar=np.ones(len(t)))

    def context(self):
        from contextlib import ExitStack
        stack=ExitStack()
        stack.enter_context(patch.object(dataio,'observed',side_effect=self.observed))
        stack.enter_context(patch.object(dataio,'event',return_value=self.event))
        stack.enter_context(patch.object(dataio,'chamber_pressure_pa',side_effect=lambda r:np.full(len(r['time_s']),1e5)))
        return stack

    def test_ignored_trace_cannot_change_score_acceptance_or_abort(self):
        changed=copy.deepcopy(self.run)
        changed['temperature_k']['C2_ISS']=[4000.]*len(self.times)
        with self.context(),patch.object(cooling,'front',return_value=dict(accepted=True,features={})):
            a,b=(cooling.score(r,self.data,1100.) for r in (self.run,changed))
            self.assertEqual(a['objective'],b['objective'])
            self.assertTrue(b['accepted'])
            self.assertFalse(b['channels']['C2_ISS']['used_for_calibration'])
            self.assertGreater(b['channels']['C2_ISS']['rmse_k'],3000.)
            self.assertEqual(destruction.loss(self.run,self.meta),destruction.loss(changed,self.meta))
            self.assertIsNone(destruction.reject_reason(changed,self.meta))
            self.assertIsNone(temperature_pruner(self.data)(changed))
            sa,sb=(gas.shape_report(r,self.data) for r in (self.run,changed))
            for key in ('shape_objective','front_concentration_objective','shape_with_front_objective','no_heat_rejected'):
                self.assertEqual(sa[key],sb[key])
            self.assertFalse(sb['channels']['C2_ISS']['used_for_calibration'])

    def test_each_remaining_channel_still_affects_loss_and_guard(self):
        with self.context(),patch.object(cooling,'front',return_value=dict(accepted=True,features={})):
            for key in CALIBRATION_CHANNELS:
                changed=copy.deepcopy(self.run);changed['temperature_k'][key]=[2000.]*len(self.times)
                score=cooling.score(changed,self.data,1100.)
                self.assertFalse(score['accepted'],key)
                self.assertGreater(score['objective'],0.,key)
                self.assertIsNotNone(temperature_pruner(self.data)(changed),key)
                self.assertIsNotNone(destruction.reject_reason(changed,self.meta),key)

    def test_ignored_observation_does_not_relax_c2_tim_abort(self):
        with patch.object(dataio,'observed',return_value=dict(C2_TIM=[320.],C2_ISS=[4000.])),\
             patch.object(cooling.guards,'reject_reason',return_value=None) as reject:
            cooling.rejection_reason(dict(time_s=[0.,1000.]),self.data,self.event)
            self.assertEqual(reject.call_args.args[2]['source_post_margin_k'],75.)


if __name__=='__main__':unittest.main()
