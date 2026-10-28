import copy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import upv_cooling_fit as fit


class CoolingFit(unittest.TestCase):
    def scene(self):
        law = dict(background_conductance_w_per_k=.01,
                   active_conductance_w_per_k=10000., geometry_reference_logit=0.)
        gains = [(298.15,1.),(320.,10.),(350.,100.),(400.,1000.),(600.,3000.)]
        knots = [[t,math.log((g-.01)/(10000.-g+.01))]for t,g in gains]
        surface=dict(damage=[0.,1.,2.],temperature_log_response=copy.deepcopy(knots))
        return dict(cells=[dict(conversion=dict(gas_yield=.4,gas_yield_max=.8,
                       net_heat_j_per_kg=2e6),thermal=dict(heat_capacity_j_per_k=400.))],
            gas_network=dict(mixing_connectivity=law,surface_connectivity=dict(temperature_log_response=knots),
                thermal_contacts=[dict(port_id='gap-ISS1-0:case-contact',connectivity=copy.deepcopy(law),solid_surface_connectivity=copy.deepcopy(surface)),
                                  dict(port_id='plate',connectivity=copy.deepcopy(law),solid_surface_connectivity=copy.deepcopy(surface))]),
            ports=[dict(id='gap-ISS1-0:case-contact',cell_id='C1:case:-x'),dict(id='plate',cell_id='steel')],
            simulation=dict(duration_s=10.,minimum_dt_s=.02),metadata={},
            measurement_probes=[dict(id='101',cell_id='C1:case:-x',interface_id='contact')])

    def test_candidate_preserves_energy_materials_reference_and_hot_law(self):
        source=self.scene();before=copy.deepcopy(source)
        trial=fit.candidate(source,dict(fit.BASE,gas_yield_scale=.6,casing_damage_metric=4.,warm_mixing_scale=10.),1500.)
        self.assertEqual(source,before)
        self.assertEqual(trial['cells'][0]['thermal'],source['cells'][0]['thermal'])
        self.assertEqual(trial['cells'][0]['conversion']['net_heat_j_per_kg'],2e6)
        self.assertAlmostEqual(trial['cells'][0]['conversion']['gas_yield'],.24)
        contacts=trial['gas_network']['thermal_contacts']
        self.assertEqual(contacts[0]['solid_surface_connectivity']['damage'],[4.,5.,6.])
        self.assertEqual(contacts[1],source['gas_network']['thermal_contacts'][1])
        a=source['gas_network']['surface_connectivity']['temperature_log_response']
        b=trial['gas_network']['surface_connectivity']['temperature_log_response']
        for old,new in zip(a,b):
            if old[0]==298.15 or old[0]>=400.:self.assertEqual(old,new)
        self.assertEqual(trial['measurement_probes'][0],source['measurement_probes'][0])
        self.assertEqual(trial['simulation']['minimum_dt_s'],.02)
        self.assertEqual(trial['metadata']['cooling_trial']['casing_contacts_changed'],1)

    def test_empty_selector_fails_instead_of_silent_no_op(self):
        source=self.scene();source['ports'][0]['cell_id']='not-a-casing'
        with self.assertRaisesRegex(ValueError,'No casing gas contacts'):
            fit.candidate(source,fit.BASE,1500.)

    def test_reverse_warm_fit_preserves_forward_wall_and_endpoint_laws(self):
        source=self.scene()
        source['gas_network']['thermal_contacts'][0]['gas_surface_connectivity']=dict(fixed='heating')
        wall=copy.deepcopy(source['gas_network']['thermal_contacts'][1]);wall['port_id']='chamber-wall:gas-partition:0'
        source['gas_network']['thermal_contacts'].append(wall)
        source['ports'].append(dict(id=wall['port_id'],cell_id='chamber-wall'))
        params=dict(fit.BASE,warm_surface_scale=10.)
        trial=fit.candidate(source,params,1500.)
        a=source['gas_network']['thermal_contacts'];b=trial['gas_network']['thermal_contacts']
        self.assertEqual(a[2],b[2])
        self.assertEqual(a[0]['gas_surface_connectivity'],b[0]['gas_surface_connectivity'])
        self.assertEqual(a[0]['connectivity'],b[0]['connectivity'])
        for old,new in zip(a[:2],b[:2]):
            x=dict(old['solid_surface_connectivity']['temperature_log_response'])
            y=dict(new['solid_surface_connectivity']['temperature_log_response'])
            self.assertGreater(y[350.],x[350.])
            for t in x:
                if t==298.15 or t>=400.:self.assertEqual(x[t],y[t])

    def test_free_gas_probe_really_changes_350_400k_without_changing_cold_gain(self):
        source=self.scene();trial=fit.candidate(source,dict(fit.BASE,gas_transport_response_scale=10.),1500.)
        def gain(scene,t):
            net=scene['gas_network'];law=net['mixing_connectivity']
            knots=np.array(net['surface_connectivity']['temperature_log_response'])
            value=np.interp(t,knots[:,0],knots[:,1])
            return law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1.+math.exp(-law['geometry_reference_logit']-value))
        self.assertAlmostEqual(gain(source,298.15),gain(trial,298.15))
        for t in (320.,350.,400.,600.):self.assertAlmostEqual(gain(trial,t)/gain(source,t),10.)
        self.assertEqual(source['gas_network']['thermal_contacts'],trial['gas_network']['thermal_contacts'])

    def test_partition_does_not_invent_energy_mass_or_capacity(self):
        source=dict(cells=[dict(conversion=dict(gas_chemical_fraction=.7,
            net_heat_j_per_kg=2e6,reaction_rate_per_s=.01,gas_yield=.3),thermal=dict(cp=1000.))])
        before=copy.deepcopy(source);trial=fit.partition_candidate(source,.84)
        old=source['cells'][0]['conversion'];new=trial['cells'][0]['conversion']
        self.assertEqual(source,before)
        self.assertAlmostEqual((1.-old['gas_chemical_fraction'])*old['reaction_rate_per_s'],
                               (1.-new['gas_chemical_fraction'])*new['reaction_rate_per_s'])
        self.assertEqual(old['net_heat_j_per_kg'],new['net_heat_j_per_kg'])
        self.assertEqual(old['gas_yield'],new['gas_yield'])
        self.assertEqual(source['cells'][0]['thermal'],trial['cells'][0]['thermal'])
        self.assertTrue(trial['metadata']['energy_partition_probe']['fresh_start_required'])

    def test_hot_source_abort_does_not_relax_witness_or_pressure_guards(self):
        original=copy.deepcopy(fit.guards.POLICY)
        for hot,expected in ((True,500.),(False,75.)):
            observation=dict(C2_ISS=[float('nan')],C2_TIM=[700. if hot else 320.])
            with patch.object(fit.dataio,'observed',return_value=observation),\
                 patch.object(fit.guards,'reject_reason',return_value=None)as reject:
                fit.rejection_reason(dict(time_s=[0.,140.]),{},dict(time_s=100.))
                policy=reject.call_args.args[2]
                self.assertEqual(policy['source_post_margin_k'],expected)
                for key in original:
                    if key not in ('source_post_margin_k','source_post_relative'):
                        self.assertEqual(policy[key],original[key])
        self.assertEqual(fit.guards.POLICY,original)

    def test_coordinate_roundtrip_and_bounds(self):
        for x in ([0.]*5,[1.]*5,[.3,.8,.1,.4,.6]):
            np.testing.assert_allclose(fit.encode(fit.decode(x)),x,atol=1e-14)
        for x in ([0.,0.],[-.1,0.,0.,0.,0.],[float('nan'),0.,0.,0.,0.]):
            with self.assertRaises(ValueError):fit.decode(x)

    def test_front_width_is_not_peak_height(self):
        t=np.arange(70.,121.,.1)
        sharp=np.maximum(0.,1.-abs(t-101.))
        a=fit.rise(t,sharp,100.)
        b=fit.rise(t,sharp*1000.,100.)
        self.assertAlmostEqual(a['width_10_90_s'],.8)
        self.assertAlmostEqual(a['width_10_90_s'],b['width_10_90_s'])
        broad=np.maximum(0.,1.-abs(t-105.)/10.)
        self.assertGreater(fit.rise(t,broad,100.)['width_10_90_s'],5.)

    def test_partial_loss_is_lower_bound_and_completion_is_json_safe(self):
        def observed(_data,times):
            return dict({k:np.full(len(times),300.)for k in fit.dataio.CHANNELS},pressure_bar=np.ones(len(times)))
        def run(end,partial):
            return dict(time_s=[0.,100.,end],partial=partial,
                temperature_k={k:[305.]*3 for k in fit.dataio.CHANNELS})
        with patch.object(fit.dataio,'event',return_value=dict(time_s=100.)),\
             patch.object(fit.dataio,'observed',side_effect=observed),\
             patch.object(fit.dataio,'chamber_pressure_pa',return_value=np.full(3,1e5)),\
             patch.object(fit,'front',return_value=dict(accepted=True,features={})):
            partial=fit.score(run(200.,True),{},300.)
            complete=fit.score(run(300.,False),{},300.)
            self.assertFalse(partial['accepted']);self.assertFalse(partial['complete'])
            self.assertLess(partial['objective'],complete['objective'])
            self.assertTrue(complete['accepted']);json.dumps(complete,allow_nan=False)

    def test_good_cooling_cannot_hide_broad_front(self):
        def observed(_data,times):
            return dict({k:np.full(len(times),300.)for k in fit.dataio.CHANNELS},pressure_bar=np.ones(len(times)))
        run=dict(time_s=[0.,100.,300.],temperature_k={k:[300.]*3 for k in fit.dataio.CHANNELS})
        with patch.object(fit.dataio,'event',return_value=dict(time_s=100.)),\
             patch.object(fit.dataio,'observed',side_effect=observed),\
             patch.object(fit.dataio,'chamber_pressure_pa',return_value=np.full(3,1e5)),\
             patch.object(fit,'front',return_value=dict(accepted=False,features={})):
            score=fit.score(run,{},300.)
            self.assertEqual(score['objective'],0.);self.assertFalse(score['accepted'])


if __name__=='__main__':unittest.main()
