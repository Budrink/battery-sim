import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_cooling_fit import candidate,encode,decode,observation,converged,rank,suggest
from fsri30_data import ROOT


def fixture(end,temperature=500.):
    t=np.arange(0.,end+.1,5.)
    run=dict(time_s=t.tolist(),partial=end<5945.,cell_mean_temperature_k={
        f'B{i:02d}':np.full(len(t),temperature).tolist()for i in range(30)})
    data=dict(time_s=np.arange(5946,dtype=float),runaway=np.arange(5946)==1700)
    for i in range(1,10):data[f'T{i}_k']=np.full(5946,500.)
    return run,data


class CoolingFitTests(unittest.TestCase):
    def test_only_two_permitted_parameters_change(self):
        base=json.loads((ROOT/'data/development/fsri30-abs-seams-20261025-r2/scenario.json').read_text())
        p=dict(effective_emissivity=.22,thermal_stirring_coefficient=.15)
        trial=candidate(base,p)
        for key in ('cells','shells','interfaces','ports','simulation','heating_protocols'):
            self.assertEqual(base[key],trial[key],key)
        ga=dict(base['gas_network']);gb=dict(trial['gas_network'])
        ga.pop('thermal_stirring_coefficient');gb.pop('thermal_stirring_coefficient')
        self.assertEqual(ga,gb)
        self.assertEqual(base['gas_network']['thermal_stirring_length_m'],.02)
        self.assertEqual(base['gas_network']['thermal_stirring_coefficient'],.05)
        self.assertEqual(trial['gas_network']['thermal_stirring_coefficient'],.15)
        for a,b in zip(encode(p),encode(decode(encode(p)))):self.assertAlmostEqual(a,b)
        with self.assertRaises(ValueError):encode(dict(p,effective_emissivity=1.1))

    def test_partial_good_curve_cannot_pass_full_acceptance(self):
        run,data=fixture(3760.)
        self.assertFalse(converged(run,data,converted_components=30)['accepted'])
        self.assertFalse(observation(run,data)['full_trajectory'])
        self.assertEqual(observation(run,data)['anchor_rmse_k'],0.)

    def test_anchor_is_same_observed_window_and_no_future(self):
        run,data=fixture(3715.)
        self.assertNotIn('anchor_rmse_k',observation(run,data))
        a,data=fixture(3760.,520.);b,_=fixture(5945.,520.)
        self.assertEqual(observation(a,data)['anchor_residual_vector_k'],observation(b,data)['anchor_residual_vector_k'])

    def test_full_cooling_and_cascade_both_required(self):
        run,data=fixture(5945.)
        self.assertTrue(converged(run,data,converted_components=30)['accepted'])
        self.assertFalse(converged(run,data,converted_components=1)['accepted'])
        bad,_=fixture(5945.,650.)
        self.assertFalse(converged(bad,data,converted_components=30)['accepted'])

    def test_partial_prefix_never_outranks_completed_fit(self):
        short=dict(full_trajectory=False,last_observed_s=3760.,anchor_rmse_k=0.)
        full=dict(full_trajectory=True,last_observed_s=5945.,full_cooling_rmse_k=50.)
        self.assertGreater(rank(full),rank(short))
        worse=dict(full,full_cooling_rmse_k=60.,anchor_rmse_k=0.)
        self.assertGreater(rank(full),rank(worse))

    def test_next_proposal_uses_measured_improvement_and_never_repeats_initial(self):
        def row(e,error):
            return dict(parameters=dict(effective_emissivity=e,thermal_stirring_coefficient=.05),
                anchor_rmse_k=abs(error),anchor_residual_vector_k=[error]*8,
                last_observed_s=3760.,full_trajectory=False,constraint=.1)
        a=row(.1,100.)
        p,how=suggest([a],.22);self.assertEqual(p['effective_emissivity'],.22)
        p,how=suggest([a,row(.22,40.)],.22)
        self.assertAlmostEqual(p['effective_emissivity'],.3)
        self.assertIn('measured',how['method'])


if __name__=='__main__':unittest.main()
