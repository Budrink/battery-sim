from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_linear_subcells import linear_reconstruction


class LinearSubcellTests(unittest.TestCase):
    def coords(self):return np.array([[1. if i&bit else -1. for bit in (4,2,1)] for i in range(8)])*.25

    def test_constant_preserved(self):
        x=np.full((2,8),300.);a,b,_=linear_reconstruction(x,self.coords(),np.ones(8))
        np.testing.assert_allclose(a,x);np.testing.assert_allclose(b,x)

    def test_linear_field_reconstructed_exactly(self):
        c=self.coords();x=(400.+c@np.array([10.,20.,30.]))[None,:]
        a,b,_=linear_reconstruction(x,c,np.ones(8))
        np.testing.assert_allclose(a,x);np.testing.assert_allclose(b,x)

    def test_positivity_limit_keeps_the_inventory(self):
        c=self.coords();x=np.zeros((1,8));x[0,0]=1.
        a,b,_=linear_reconstruction(x,c,np.ones(8))
        self.assertLess(a.min(),0.);self.assertGreaterEqual(b.min(),-1e-14)
        np.testing.assert_allclose(b.sum(axis=1),x.sum(axis=1))

    def test_quadratic_pattern_is_not_falsely_linear(self):
        c=self.coords();x=(400.+100.*c[:,0]*c[:,1])[None,:]
        a,_,_=linear_reconstruction(x,c,np.ones(8))
        np.testing.assert_allclose(a,400.);self.assertGreater(np.max(abs(a-x)),1.)

    def test_irregular_weights_conserve_the_mean(self):
        c=self.coords();x=np.array([[0.,1.,2.,3.,5.,8.,13.,21.]])
        w=np.array([1.,2.,3.,4.,5.,6.,7.,8.])
        a,b,_=linear_reconstruction(x,c,w)
        np.testing.assert_allclose(a@w,x@w);np.testing.assert_allclose(b@w,x@w)


if __name__=='__main__':unittest.main()
