import sys
from pathlib import Path
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'scripts'))
from battery_upv_electrical_audit import electrical_integrals, conditional_response


class ElectricalAudit(unittest.TestCase):
    def test_linear_product_is_integrated_exactly(self):
        # I=t/5, U=4+t/5: W=160/3 J, charge=10 C, integral I²=40/3.
        w,q,sq=electrical_integrals([0.,10.],[0.,2.],[4.,6.],[0.,10.])
        np.testing.assert_allclose(w,[0.,160/3])
        np.testing.assert_allclose(q,[0.,10.])
        np.testing.assert_allclose(sq,[0.,40/3])

    def test_query_grid_and_internal_knot_invariance(self):
        t=[0.,3.,10.];i=[0.,2.,1.];u=[4.,4.5,6.]
        coarse=electrical_integrals(t,i,u,[10.])
        fine=electrical_integrals(t,i,u,np.arange(0.,10.1,.1))
        for a,b in zip(coarse,fine):self.assertAlmostEqual(a[-1],b[-1],places=11)

    def test_does_not_extrapolate_work_outside_record(self):
        values=electrical_integrals([5.,10.],[2.,2.],[4.,4.],[0.,5.,10.,20.])
        np.testing.assert_allclose(values[0],[0.,0.,40.,40.])

    def test_zero_current_compliance_is_not_heat_or_resistance(self):
        power,r=conditional_response([4.5,60.],[35.,0.],4.2)
        self.assertAlmostEqual(power[0],10.5)
        self.assertEqual(power[1],0.)
        self.assertTrue(np.isnan(r[1]))

    def test_residual_is_not_clipped(self):
        power,r=conditional_response([3.],[2.],4.)
        self.assertEqual(power[0],-2.)
        self.assertEqual(r[0],-.5)

    def test_unordered_time_rejected(self):
        with self.assertRaises(ValueError):electrical_integrals([1.,0.],[2.,2.],[4.,4.],[0.])


if __name__=='__main__':unittest.main()
