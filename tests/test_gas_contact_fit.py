import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import upv_gas_contact_fit as fit
import upv_connectivity as links
import upv_r123_data as dataio


class GasContactFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene=json.loads((ROOT/'data/calibration/upv-separated-chemistry-v6/gas-connectivity-20261020-r3/case-1/scenario.json').read_text())

    def test_only_gas_contact_laws_change(self):
        original=copy.deepcopy(self.scene)
        result=fit.candidate(original,dict(fit.BASE,gas_ceiling=100.,metric_450=8.,metric_900=10.,wall_h=20.))
        for k in ('cells','interfaces','shells','boundaries','ports','measurement_probes'):
            self.assertEqual(result[k],original[k],k)
        for k in set(original['gas_network'])-{'thermal_contacts','surface_connectivity','mixing_connectivity'}:
            self.assertEqual(result['gas_network'][k],original['gas_network'][k],k)
        self.assertEqual(self.scene,original)

    def test_cold_scale_preserved_and_materials_share_profile(self):
        scene=fit.candidate(self.scene,dict(fit.BASE,gas_ceiling=100.,metric_450=8.,metric_900=10.))
        ports={p['id']:p for p in scene['ports']};profiles=[]
        for before,after in zip(self.scene['gas_network']['thermal_contacts'],scene['gas_network']['thermal_contacts']):
            area=ports[after['port_id']]['area_m2']
            self.assertAlmostEqual(links.reference_gas_conductance(before,area),links.reference_gas_conductance(after,area),places=10)
            law=after['connectivity'];g=law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1+np.exp(-law['geometry_reference_logit']-8.))
            self.assertGreater(g,links.reference_gas_conductance(after,area))
            profiles.append(after['gas_surface_connectivity'])
        self.assertTrue(all(p==profiles[0]for p in profiles))
        self.assertEqual(scene['gas_network']['surface_connectivity'],profiles[0])
        law=scene['gas_network']['mixing_connectivity']
        self.assertAlmostEqual(law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1+np.exp(-law['geometry_reference_logit'])),1.)
        self.assertEqual(law['endpoint_damage_weight'],0.)

    def test_hard_bounds_and_coordinate_roundtrip(self):
        for p in (fit.BASE,dict(gas_ceiling=100.,metric_450=8.,metric_900=10.,wall_h=20.)):
            decoded=fit.decode(fit.encode(p))
            for k in fit.KEYS:self.assertAlmostEqual(decoded[k],p[k])
        with self.assertRaises(ValueError):fit.candidate(self.scene,dict(fit.BASE,gas_ceiling=1.5))
        with self.assertRaises(ValueError):fit.decode([-1.,0.,0.,0.])

    def test_search_coordinates_cannot_produce_falling_temperature_knots(self):
        for x in np.random.default_rng(11).random((256,4)):
            p=fit.decode(x)
            self.assertGreaterEqual(p['metric_450'],0.)
            self.assertGreaterEqual(p['metric_900'],p['metric_450'])
            np.testing.assert_allclose(fit.encode(p),x,atol=1e-12)
        for bad in (dict(fit.BASE,metric_450=-.01),dict(fit.BASE,metric_450=4.,metric_900=3.)):
            with self.assertRaises(ValueError):fit.candidate(self.scene,bad)
            with self.assertRaises(ValueError):fit.encode(bad)

    def test_gas_donor_conductance_is_nondecreasing(self):
        # Scalar evaluation of the same directed logistic source response; this
        # is a constitutive test, NOT a replacement heat-transfer calculation.
        for p in (fit.BASE,dict(fit.BASE,gas_ceiling=30.,metric_450=0.,metric_900=2.08),
                  dict(fit.BASE,gas_ceiling=300.,metric_450=12.,metric_900=18.)):
            scene=fit.candidate(self.scene,p)
            for c in scene['gas_network']['thermal_contacts']:
                law=c['connectivity'];table=np.asarray(c['gas_surface_connectivity']['temperature_log_response'])
                z=law['geometry_reference_logit']+np.interp(np.linspace(298.15,1600.,256),table[:,0],table[:,1])
                for prefix in ('','destroyed_'):
                    g=law[prefix+'background_conductance_w_per_k']+law[prefix+'active_conductance_w_per_k']/(1+np.exp(-z))
                    self.assertTrue(np.all(np.diff(g)>=-1e-12))

    def test_suffix_loss_does_not_invent_a_cold_prefix(self):
        data,_=dataio.load('no-cooling');event=dataio.event(data);end=event['time_s']+300.
        t=np.arange(750.,821.,1.);obs=dataio.observed(data,t)
        run=dict(time_s=t,temperature_k={k:obs[k]for k in dataio.CHANNELS},gas_observers={'chamber':{'pressure_pa':obs['pressure_bar']*1e5}})
        score=fit.suffix_loss(run,dict(data=data,event=event,fit_end_s=end))
        self.assertLess(score['objective'],1e-20)
        self.assertEqual(score['started_at_s'],750.)

    def test_recovery_comparison_ignores_checkpoint_polling_overshoot(self):
        data,_=dataio.load('no-cooling');event=dataio.event(data)
        meta=dict(data=data,event=event,fit_end_s=event['time_s']+300.)
        def run(stop):
            t=np.arange(750.,stop+1.,1.);obs=dataio.observed(data,t)
            values={}
            for k in dataio.CHANNELS:
                finite=np.isfinite(obs[k]);values[k]=np.interp(t,t[finite],obs[k][finite])+10.
            return dict(time_s=t,temperature_k=values,gas_observers={'chamber':{'pressure_pa':obs['pressure_bar']*1e5}})
        row=dict(status='pruned',feasible=False,critical_violation=dict(time_s=930.),objective=999.)
        self.assertEqual(fit.ranking(row,run(940.),meta),fit.ranking(row,run(1000.),meta))
        self.assertIsNone(fit.ranking(dict(status='failed'),None,None))

    def test_noise_is_not_a_calibration_gain(self):
        self.assertFalse(fit.meaningful_gain((1,-930.,9.9999),dict(rank=[1,-930.,10.])))
        self.assertTrue(fit.meaningful_gain((1,-940.,100.),dict(rank=[1,-930.,10.])))
        self.assertTrue(fit.meaningful_gain((0,1000.),dict(rank=[1,-940.,10.])))


if __name__=='__main__':unittest.main()
