import math
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upv_film_surrogate import encode,propose


class LocalFilmSearch(unittest.TestCase):
    def test_bounded_native_only_minimax_proposal(self):
        base=dict(reverse_profile_power=3.,backing_fraction=.03,wall_hot_scale=.9)
        points=[base,dict(base,reverse_profile_power=2.),dict(base,reverse_profile_power=4.),
                dict(base,backing_fraction=.02),dict(base,wall_hot_scale=1.)]
        centre=encode(base);target=centre+np.array([.2,-.2,.05]);vectors=[]
        for p in points:
            x=encode(p)-target
            vectors.append(np.r_[x,x,[0.,0.,0.]])
        result,report=propose(points,vectors,[1]*6)
        self.assertLess(np.linalg.norm(encode(result)-target),1e-5)
        self.assertFalse(report['accepted_calibration'])
        self.assertTrue(report['requires_native_replay'])
        self.assertNotIn('C2_ISS',report['predicted_normalized_rmse'])

    def test_rank_deficiency_is_not_a_guessed_derivative(self):
        points=[dict(reverse_profile_power=m,backing_fraction=.03,wall_hot_scale=.9)for m in [2.,3.,4.]]
        with self.assertRaises(ValueError):propose(points,np.ones((3,9)),[1]*6)

    def test_hidden_kinetic_change_is_not_a_film_derivative(self):
        base=dict(reverse_profile_power=3.,backing_fraction=.03,wall_hot_scale=.9,r2_rate=1.)
        other=dict(base,reverse_profile_power=4.,r2_rate=2.)
        with self.assertRaisesRegex(ValueError,'Untracked'):
            propose([base,other],np.ones((2,9)),[1]*6)
