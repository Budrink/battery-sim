import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from resolve_upv_rupture_impulse import shell_trial, crossing, hot_internal_recovery, steep_r3
import numpy as np


class RuptureImpulse(unittest.TestCase):
    def test_only_strength_and_area_rate_change_uniformly(self):
        rupture = dict(kind='rupture', opening_pressure_pa=3e6,
            damage_opening=dict(mode='deterministic_aperture', area_rate_per_s=1.,
                                seam_strength_factor=.2, fracture_energy_j_per_m2=2000.))
        s = dict(simulation=dict(duration_s=1000.,minimum_dt_s=.02),
            cells=['fixed chemistry'],shells=['fixed thermal damage'],interfaces=['fixed'],
            gas_network=dict(compartments=['fixed'],thermal_contacts=['fixed'],
                             channels=[dict(kind='passage',area_m2=.01),copy.deepcopy(rupture),copy.deepcopy(rupture)]))
        original = copy.deepcopy(s);trial = shell_trial(s,4.,100.,1186.639)
        self.assertEqual(s,original)
        for c in trial['gas_network']['channels'][1:]:
            self.assertEqual(c['opening_pressure_pa'],12e6)
            self.assertEqual(c['damage_opening']['area_rate_per_s'],100.)
            c['opening_pressure_pa'] /= 4.;c['damage_opening']['area_rate_per_s'] /= 100.
        trial.pop('metadata');trial.pop('surface_probe_trace');trial['simulation']['duration_s']=1000.
        self.assertEqual(trial,s)

    def test_invalid_factors_fail_before_gpu_work(self):
        for strength,rate,duration in ((0.,1.,1.),(100.,1.,1.),(1.,101.,1.),(1.,1.,0.),(float('nan'),1.,1.)):
            with self.assertRaises(ValueError):shell_trial({},strength,rate,duration)

    def test_crossing_is_observed_not_fitted_or_extrapolated(self):
        self.assertEqual(crossing([0.,1.,2.],[0.,.2,.8],.5),2.)
        self.assertIsNone(crossing([0.,1.,2.],[0.,.2,.8],1.))

    def test_internal_probe_preserves_cold_response_and_other_physics(self):
        response = dict(reference_temperature_k=298.15,temperature_scale_k=100.,
            linear=[-2.88]*3,quadratic=[0.]*3,damage=[.1]*3)
        s = dict(cells=[dict(role='battery',thermal=dict(bulk_response=response,
                    conductivity_w_per_m_k=[.9,28.3,28.3],heat_capacity_j_per_k=401.)),
                        dict(role='passive',thermal=dict(unchanged=True))],
                 shells=['fixed'],gas_network=['fixed'],interfaces=['fixed'])
        original=copy.deepcopy(s);trial=hot_internal_recovery(s)
        p=trial['cells'][0]['thermal']['bulk_response'];knots=np.asarray(p['temperature_log_response'])
        for t in np.linspace(273.15,425.,50):
            self.assertAlmostEqual(np.interp(t,knots[:,0],knots[:,1]),-2.88*(t-298.15)/100.)
        self.assertEqual(np.interp(500.,knots[:,0],knots[:,1]),0.)
        self.assertEqual(p['damage'],response['damage'])
        trial['cells'][0]['thermal']['bulk_response']=response;trial.pop('metadata')
        self.assertEqual(trial,original);self.assertEqual(s,original)

    def test_internal_probe_does_not_silently_remove_anisotropy(self):
        s=dict(cells=[dict(role='battery',thermal=dict(bulk_response=dict(linear=[-1.,-2.,-3.])))])
        with self.assertRaises(ValueError):hot_internal_recovery(s)

    def test_r3_probe_only_changes_activation_not_inventory_or_reference_rate(self):
        gas=dict(reaction_rate_per_s=20.,reaction_reference_k=600.,
            reaction_activation_j_per_mol=40000.,initial_mass_kg=.1,initial_oxidizer_mass_fraction=.23)
        s=dict(gas_network=dict(compartments=[gas,copy.deepcopy(gas)],channels=['fixed']),
               cells=['fixed R1 R2 and energy'],shells=['fixed'])
        original=copy.deepcopy(s);trial=steep_r3(s)
        def rate(e,t):return 20.*np.exp(e/8.314462618*(1/600.-1/t))
        self.assertEqual(rate(40000.,600.),rate(80000.,600.))
        self.assertLess(rate(80000.,300.),rate(40000.,300.)/1000.)
        self.assertGreater(rate(80000.,800.),rate(40000.,800.))
        for g in trial['gas_network']['compartments']:
            self.assertEqual(g['reaction_activation_j_per_mol'],80000.)
            g['reaction_activation_j_per_mol']=40000.
        trial.pop('metadata');self.assertEqual(trial,original);self.assertEqual(s,original)


if __name__=='__main__':unittest.main()
