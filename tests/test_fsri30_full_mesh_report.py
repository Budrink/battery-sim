from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_fsri30_full_mesh import phase_comparison,prefix_check,physical_signature


def fixture():
    return dict(time_s=[0.,10.,20.,30.],cell_mean_temperature_k={'B00':[300.,500.,600.,400.]},
        process_totals=dict(gas_released_energy_j=[0.,100.,110.,110.]),
        gas_groups=dict(outside=dict(maximum_temperature_k=[300.,1500.,800.,400.])))


class FullMeshReportTests(unittest.TestCase):
    def test_only_horizon_and_readonly_labels_can_change(self):
        a=dict(simulation=dict(duration_s=1800.,minimum_dt_s=.01),gas_network=dict(pressure_pattern_predictor=False),
               metadata=dict(calibrated=False),failure_sample_times_s=[0.,1800.])
        b=deepcopy(a);b['simulation']['duration_s']=5945.;b['failure_sample_times_s']=[0.,5945.]
        self.assertEqual(physical_signature(a),physical_signature(b))
        b['gas_network']['pressure_pattern_predictor']=True
        self.assertNotEqual(physical_signature(a),physical_signature(b))

    def test_same_phase_passes(self):
        run=fixture();self.assertTrue(phase_comparison(run,run,0.,30.)['accepted'])

    def test_cold_tail_cannot_hide_bad_impulse(self):
        a=fixture();b=deepcopy(a);b['gas_groups']['outside']['maximum_temperature_k'][1]=2000.
        self.assertFalse(phase_comparison(a,b,0.,10.)['accepted'])
        self.assertTrue(phase_comparison(a,b,20.,30.)['accepted'])

    def test_relative_temperature_uses_kelvin_not_celsius(self):
        a=fixture();b=deepcopy(a)
        for run,t in ((a,300.),(b,310.)):
            run['cell_mean_temperature_k']['B00']=[t]*4
            run['gas_groups']['outside']['maximum_temperature_k']=[t]*4
        self.assertTrue(phase_comparison(a,b,0.,30.)['accepted'])

    def test_time_shift_is_not_a_fit_parameter(self):
        a=fixture();b=deepcopy(a);b['time_s'][2]+=.001
        with self.assertRaisesRegex(ValueError,'do not align'):phase_comparison(a,b,0.,30.)

    def test_full_prefix_must_repeat_pilot(self):
        full=fixture();pilot=deepcopy(full)
        pilot['time_s']=pilot['time_s'][:3]
        pilot['cell_mean_temperature_k']['B00']=pilot['cell_mean_temperature_k']['B00'][:3]
        pilot['process_totals']['gas_released_energy_j']=pilot['process_totals']['gas_released_energy_j'][:3]
        pilot['gas_groups']['outside']['maximum_temperature_k']=pilot['gas_groups']['outside']['maximum_temperature_k'][:3]
        self.assertTrue(prefix_check(pilot,full)['accepted'])
        full['cell_mean_temperature_k']['B00'][1]+=.1
        self.assertFalse(prefix_check(pilot,full)['accepted'])


if __name__=='__main__':unittest.main()
