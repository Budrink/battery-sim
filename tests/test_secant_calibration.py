import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_upv_geometric_secant import SecantJacobian


class SecantCalibration(unittest.TestCase):
    def test_affine_and_no_inplace_mutation(self):
        matrix = np.array([[2., 3.], [-1., 4.], [1., -2.]])
        target = np.array([.3, -.2])
        fun = lambda x: matrix@(x-target)
        jac = SecantJacobian(fun, np.array([-2., -2.]), np.array([2., 2.]))
        returned = jac(np.zeros(2)); returned[:] = 999.
        np.testing.assert_allclose(jac(np.array([.1, .1])), matrix, atol=1e-12)
        fit = least_squares(fun, np.zeros(2), jac=jac, bounds=(-2., 2.))
        np.testing.assert_allclose(fit.x, target, atol=1e-9)

    def test_nonlinear_bounded_fit_uses_true_residuals(self):
        fun = lambda x: np.array([np.exp(x[0])-1.5, 2*x[1]-.8, .1*(x[0]+x[1]-.7)])
        jac = SecantJacobian(fun, np.full(2, -1.), np.ones(2))
        fit = least_squares(fun, np.zeros(2), jac=jac, bounds=(-1., 1.), max_nfev=25)
        self.assertLess(np.linalg.norm(fun(fit.x)), .02)
        self.assertTrue(np.all(np.abs(fit.x)<=1.))

    def test_secant_satisfies_observed_change(self):
        fun = lambda x: np.array([x[0]**2+x[1], np.exp(x[1])])
        jac = SecantJacobian(fun, np.full(2, -1.), np.ones(2))
        a, b = np.array([.1, .2]), np.array([.3, -.1])
        jac(a); matrix = jac(b)
        np.testing.assert_allclose(matrix@(b-a), fun(b)-fun(a), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
