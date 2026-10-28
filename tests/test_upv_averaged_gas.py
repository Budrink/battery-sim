import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_upv_averaged_gas import averaged_scene


class AveragedScenario(unittest.TestCase):
    def fixture(self):
        return dict(gas_network=dict(mass_backend='cuda_network',channels=['frozen'],thermal_contacts=['frozen']),
            simulation=dict(duration_s=1200.,minimum_dt_s=.005),cells=['frozen'],shells=['frozen'],
            failure_sample_times_s=[0.,600.,900.,1000.,1200.])

    def test_frozen_physics_and_only_observation_truncation(self):
        source=self.fixture();original=copy.deepcopy(source)
        trial=averaged_scene(source,950.)
        self.assertEqual(source,original)
        self.assertEqual(trial['failure_sample_times_s'],[0.,600.,900.,950.])
        expected=copy.deepcopy(original);expected['gas_network']['network_integrator']='averaged'
        expected['simulation']['duration_s']=950.;expected['failure_sample_times_s']=[0.,600.,900.,950.]
        self.assertEqual(trial,expected)
        self.assertEqual(trial['simulation']['minimum_dt_s'],.005)

    def test_endpoint_and_duration_validation(self):
        self.assertEqual(averaged_scene(self.fixture(),900.)['failure_sample_times_s'],[0.,600.,900.])
        for duration in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):averaged_scene(self.fixture(),duration)


if __name__=='__main__':unittest.main()
