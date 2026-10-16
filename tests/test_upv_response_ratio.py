"""Temperature-rise ratios must be unit-origin independent and finite."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from analyze_upv_response_ratio import rise_ratio, sample_function
finally:sys.path.pop(0)


class ResponseRatioTests(unittest.TestCase):
    def test_sampled_function_never_extrapolates_or_sorts_a_loop(self):
        self.assertEqual(sample_function([30.,40.,50.],[.6,.8,.7],[25.,35.,45.,55.]),[[35.,.7],[45.,.75]])
        self.assertIsNone(sample_function([30.,40.,35.],[.6,.8,.7],[35.]))

    def test_kelvin_celsius_and_small_denominator(self):
        a=np.array([300.,300.1,301.,304.,310.]);b=np.array([302.,302.4,303.,310.,317.])
        r,mask=rise_ratio(a,b);other,mask2=rise_ratio(a-273.15,b-273.15)
        np.testing.assert_array_equal(mask,[False,False,True,True,True])
        np.testing.assert_array_equal(mask,mask2);np.testing.assert_allclose(r,other,equal_nan=True)
        np.testing.assert_allclose(r[mask],[1.,2.,1.5])

    def test_cooling_and_flat_series_are_not_heating_ratios(self):
        for a,b in (([300.,300.,300.],[300.,310.,320.]),([300.,310.,320.],[300.,290.,280.])):
            r,mask=rise_ratio(a,b);self.assertFalse(mask.any());self.assertTrue(np.isnan(r).all())


if __name__=='__main__':unittest.main()
