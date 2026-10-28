from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_fsri30_cascade import threshold_intervals,event_shape_details


class CascadeDiagnosisTests(unittest.TestCase):
    def test_event_time_is_an_observation_bracket(self):
        scene=dict(cells=[dict(role='battery',id='B10',conversion=dict(gas_yield=.5,mass_kg=1.))])
        raw=dict(time_s=[0.,5.,10.],chemistry=dict(B10=dict(r2_generated_gas_kg=[0.,.01,.1])))
        self.assertEqual(threshold_intervals(scene,raw),{'B10':[5.,10.]})

    def test_variable_yield_cannot_fake_reaction_progress(self):
        scene=dict(cells=[dict(role='battery',id='B10',conversion=dict(gas_yield=.5,mass_kg=1.,gas_yield_temperature_per_k=.01))])
        with self.assertRaisesRegex(ValueError,'constant gas yield'):threshold_intervals(scene,dict(time_s=[0.,5.]))

    def test_shifted_same_shape_does_not_change_amplitude(self):
        t=np.arange(0.,401.,5.)
        runs={m:dict(time_s=t.tolist(),cell_mean_temperature_k={'B10':(300.+np.maximum(0.,t-event)).tolist()}) for m,event in ((20,100.),(10,110.))}
        events={m:[dict(component='B10',time_s=event)] for m,event in ((20,100.),(10,110.))}
        row=event_shape_details(runs,events)['rows'][0]
        self.assertEqual(row['raw_arrival_delay_s'],10.)
        self.assertEqual(row['aligned_after_jump_rms_relative_kelvin'],0.)
        self.assertEqual(row['recorded_peak_relative_difference'],0.)

    def test_missing_late_shape_cannot_be_extrapolated(self):
        runs={m:dict(time_s=[0.,100.,150.],cell_mean_temperature_k={'B10':[300.,500.,400.]}) for m in (20,10)}
        events={m:[dict(component='B10',time_s=100.)] for m in (20,10)}
        with self.assertRaisesRegex(ValueError,'extrapolate'):event_shape_details(runs,events)


if __name__=='__main__':unittest.main()
