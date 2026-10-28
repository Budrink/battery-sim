import sys
from pathlib import Path
from unittest import TestCase
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_upv_gas_access_accuracy import series_error


class NumericalExcursionBudget(TestCase):
    def test_constant_temperature_offset_does_not_change_relative_error(self):
        t=[0.,1.,2.];a=np.array([0.,5.,10.]);b=np.array([0.,5.1,10.2])
        x=series_error(t,a,b,1.);y=series_error(t,a+300.,b+300.,1.)
        self.assertAlmostEqual(x['time_weighted_nrmse'],y['time_weighted_nrmse'])
        self.assertAlmostEqual(x['normalized_endpoint_error'],.02)

    def test_important_local_mass_error_is_not_hidden_by_air_inventory(self):
        r=series_error([0.,1.,2.],[.14,.15,.16],[.14,.152,.164],1e-8)
        self.assertAlmostEqual(r['normalized_endpoint_error'],.2)
        self.assertGreater(r['time_weighted_nrmse'],.05)

    def test_identity_and_invalid_observations(self):
        self.assertEqual(series_error([0.,1.],[2.,3.],[2.,3.],1.)['maximum_absolute_error'],0.)
        with self.assertRaises(ValueError):series_error([0.,0.],[2.,3.],[2.,3.],1.)
        with self.assertRaises(ValueError):series_error([0.,1.],[2.,3.],[2.,np.nan],1.)
