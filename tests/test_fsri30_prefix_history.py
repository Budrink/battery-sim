from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_prefix_history import conversion_events,trim_observed,summary_at
from report_fsri30_full_mesh import aligned_times


def history():
    return dict(time_s=[0.,10.,20.,30.],cell_mean_temperature_k={'B00':[300.,400.,500.,600.]},
        process_totals=dict(gas_released_energy_j=[0.,10.,20.,30.]),
        gas_groups=dict(outside=dict(maximum_temperature_k=[300.,400.,500.,600.])),
        live_state=dict(time_s=35.,objects={'B00':dict(max_temperature_k=1000.)}))


class PrefixHistoryTests(unittest.TestCase):
    def test_live_state_is_not_an_observation(self):
        result=trim_observed(history(),35.)
        self.assertEqual(result['time_s'][-1],30.)
        self.assertNotIn('live_state',result)
        self.assertEqual(result['cell_mean_temperature_k']['B00'][-1],600.)

    def test_no_interpolation_at_arbitrary_cutoff(self):
        result=trim_observed(history(),24.)
        self.assertEqual(result['time_s'],[0.,10.,20.])
        self.assertEqual(result['process_totals']['gas_released_energy_j'][-1],20.)

    def test_common_prefix_does_not_hide_missing_observations(self):
        a=trim_observed(history(),20.);b=deepcopy(a);b['time_s'][1]=11.
        with self.assertRaisesRegex(ValueError,'do not align'):aligned_times(a,b)

    def test_future_event_cannot_enter_common_prefix_count(self):
        events=[dict(component='B00',time_s=15.),dict(component='B01',time_s=29.)]
        result=summary_at(trim_observed(history(),20.),events)
        self.assertEqual(result['r2_converted_components'],1)
        self.assertEqual(result['all_gas_r3_j'],20.)

    def test_r1_residue_is_not_mistaken_for_r2_conversion(self):
        scene=dict(cells=[dict(role='battery',id='B00',conversion=dict(mass_kg=1.,gas_yield=.5))])
        raw=dict(time_s=[0.,10.,20.],chemistry={'B00':dict(r2_generated_gas_kg=[0.,.025,.075],residue_kg=[1.,1.,1.])})
        result=conversion_events(scene,raw)
        self.assertEqual(result,[dict(component='B00',time_s=15.)])
        raw['chemistry']['B00']['r2_generated_gas_kg']=[0.,0.,0.]
        self.assertEqual(conversion_events(scene,raw),[])

    def test_variable_yield_requires_a_different_direct_ledger(self):
        scene=dict(cells=[dict(role='battery',id='B00',conversion=dict(mass_kg=1.,gas_yield=.5,gas_yield_temperature_per_k=.01))])
        with self.assertRaisesRegex(ValueError,'constant gas-yield'):conversion_events(scene,dict(time_s=[0.,10.]))


if __name__=='__main__':unittest.main()
