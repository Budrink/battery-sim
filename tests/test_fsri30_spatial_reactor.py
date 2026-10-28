import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_fsri30_spatial_reactor import trial_scene


class SpatialReactorContract(unittest.TestCase):
    def fixture(self):
        return dict(simulation=dict(duration_s=2000.,minimum_dt_s=.01,max_dt_s=.1),
            failure_sample_times_s=[0.,1100.,1230.,2000.],
            metadata=dict(gas_volume_geometry={'air':dict(low_m=[0.,0.,0.],high_m=[1.,1.,1.])}),
            gas_network=dict(pressure_pattern_predictor=False,reactive_transit=True,
                compartments=[dict(id='air',volume_m3=1.,reaction_rate_per_s=3.),dict(id='pore',volume_m3=1e-6)],
                channels=[dict(id='path',area_m2=.1)]),cells=[dict(conversion={'unchanged':True})])

    def test_only_numerical_controls_boxes_and_horizon_change(self):
        original=self.fixture();frozen=copy.deepcopy(original)
        trial=trial_scene(original,True,1230.)
        self.assertEqual(original,frozen)
        self.assertEqual(trial['simulation']['minimum_dt_s'],.01)
        self.assertEqual(trial['cells'],original['cells'])
        self.assertEqual(trial['gas_network']['channels'],original['gas_network']['channels'])
        self.assertEqual(trial['failure_sample_times_s'],[0.,1100.,1230.])
        self.assertTrue(trial['gas_network']['reactive_reconstruction'])
        self.assertNotIn('reconstruction_box',trial['gas_network']['compartments'][1])
        node=trial['gas_network']['compartments'][0]
        self.assertEqual(node.pop('reconstruction_box'),frozen['metadata']['gas_volume_geometry']['air'])
        self.assertEqual(node,original['gas_network']['compartments'][0])

    def test_off_retains_same_explicit_geometry(self):
        on=trial_scene(self.fixture(),True,1230.);off=trial_scene(self.fixture(),False,1230.)
        self.assertEqual(on['gas_network']['compartments'],off['gas_network']['compartments'])
        self.assertFalse(off['gas_network']['reactive_reconstruction'])

    def test_no_new_observation_boundary(self):
        with self.assertRaises(ValueError):trial_scene(self.fixture(),True,1229.)

    def test_floor_and_rejected_predictor_stay_constraints(self):
        for field,value in (('floor',.005),('predictor',True)):
            s=self.fixture()
            if field=='floor':s['simulation']['minimum_dt_s']=value
            else:s['gas_network']['pressure_pattern_predictor']=value
            with self.assertRaises(ValueError):trial_scene(s,True,1230.)


if __name__=='__main__':unittest.main()
