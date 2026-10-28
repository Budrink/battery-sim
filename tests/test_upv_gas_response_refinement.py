import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upv_gas_response_refinement import candidate,gain,front_constraints,front_prune_reason


class GasResponseRefinement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=Path(__file__).resolve().parents[1]/'data/calibration/upv-separated-chemistry-v6/gas-access-refinement-20261022-r4-fresh/trial-000/scenario.json'
        if not p.exists():raise unittest.SkipTest('Optional latest local scenario')
        cls.source=json.loads(p.read_text())

    def build(self,scale=1.,damage=0.):
        with patch('upv_gas_response_refinement.dataio.load',return_value=({},{})),\
             patch('upv_gas_response_refinement.dataio.event',return_value={} ),\
             patch('upv_gas_response_refinement.dataio.observation_times',return_value=[0.,1100.]):
            return candidate(self.source,scale,damage)

    def test_mixing_scales_hot_knots_not_cold_reference_or_other_physics(self):
        before=copy.deepcopy(self.source)
        for scale in [.03,.1,.3,1.,3.]:
            new=self.build(scale)
            oldnet=self.source['gas_network'];net=new['gas_network']
            for t,_ in oldnet['surface_connectivity']['temperature_log_response']:
                ratio=1. if t<=298.15 else scale
                self.assertAlmostEqual(gain(net['mixing_connectivity'],net['surface_connectivity'],t)/gain(oldnet['mixing_connectivity'],oldnet['surface_connectivity'],t),ratio,places=11)
            for k in oldnet:
                if k not in ('mixing_connectivity','surface_connectivity'):self.assertEqual(net[k],oldnet[k],k)
            for k in ['cells','materials','shells','interfaces','ports','boundaries','measurement_probes']:
                self.assertEqual(new.get(k),self.source.get(k),k)
            self.assertEqual(new['simulation']['minimum_dt_s'],.02)
        self.assertEqual(before,self.source)

    def test_neutral_response_is_exact_and_reverse_damage_only_selects_casing(self):
        self.assertEqual(self.build()['gas_network'],self.source['gas_network'])
        new=self.build(damage=-4.);ports={p['id']:p['cell_id']for p in new['ports']}
        for a,b in zip(self.source['gas_network']['thermal_contacts'],new['gas_network']['thermal_contacts']):
            selected=':case:'in ports[a['port_id']]
            self.assertEqual(a!=b,selected)
            if selected:
                self.assertEqual(b['solid_surface_connectivity']['damage'],[v-4. for v in a['solid_surface_connectivity']['damage']])
                for k in a:
                    if k!='solid_surface_connectivity':self.assertEqual(a[k],b[k])
        for k in ['cells','shells','interfaces']:self.assertEqual(new[k],self.source[k])

    def test_invalid_parameters_and_molecular_scope_are_rejected(self):
        for scale,damage in [(0.,0.),(float('nan'),0.),(3.1,0.),(1.,-6.1),(1.,float('inf'))]:
            with self.assertRaises(ValueError):self.build(scale,damage)
        source=copy.deepcopy(self.source);source['gas_network']['free_gas_response_only']=False
        with self.assertRaises(ValueError):candidate(source,.3)


class FrontConstraints(unittest.TestCase):
    def result(self,jump=12.,pressure=8.):
        return dict(front=dict(reasons=[],features={'pressure':dict(predicted=dict(peak=pressure),observed=dict(peak=8.))}),
            shape=dict(no_heat_rejected=False,channels={
                '101':dict(calculated_max_rise_1_5s_k=jump,observed_max_rise_1_5s_k=12.),
                '102':dict(calculated_max_rise_1_5s_k=1.,observed_max_rise_1_5s_k=1.),
                '107':dict(calculated_max_rise_1_5s_k=1.,observed_max_rise_1_5s_k=1.)}))

    def test_lower_cooling_scalar_cannot_select_an_erased_pulse(self):
        r=front_constraints(self.result(jump=4.))
        self.assertFalse(r['accepted'])
        self.assertIn('C1/ISS',r['reasons'][0])
        self.assertTrue(front_constraints(self.result(jump=8.))['accepted'])
        self.assertTrue(front_constraints(self.result(jump=22.))['accepted'])

    def test_pressure_is_not_treated_as_an_uncertain_temperature_peak(self):
        self.assertFalse(front_constraints(self.result(pressure=12.))['accepted'])
        r=front_constraints(self.result(pressure=9.))
        self.assertTrue(r['accepted'])
        self.assertGreater(r['pressure_peak_relative_error'],r['pressure_peak_acceptance_limit'])

    def test_better_tail_cannot_invent_large_protected_face_impulses(self):
        self.assertFalse(front_constraints(self.result(jump=200.))['accepted'])
        result=self.result()
        result['shape']['channels']['107']['calculated_max_rise_1_5s_k']=18.
        self.assertFalse(front_constraints(result)['accepted'])
        result['shape']['channels']['107']['calculated_max_rise_1_5s_k']=4.
        self.assertTrue(front_constraints(result)['accepted'])

    def test_excitation_waits_for_the_entire_protected_window(self):
        result=self.result();result['shape']['no_heat_rejected']=True
        result['shape']['channels']['107']['excitation_penalty']=.01
        def check(time,extra=None):
            value=copy.deepcopy(result)
            if extra:value['front']['reasons'].append(extra)
            with patch('upv_gas_response_refinement.front',return_value=value['front']),\
                 patch('upv_gas_response_refinement.dataio.event',return_value=dict(time_s=886.6389999166131)),\
                 patch('upv_gas_response_refinement.shape_report',return_value=value['shape']):
                return front_prune_reason(dict(time_s=[time]),{})
        self.assertIsNone(check(916.))
        self.assertIn('Missing witness heating',check(930.))
        self.assertIn('pressure timing',check(916.,'pressure timing'))
        result['shape']['channels']['108']=dict(excitation_penalty=.1)
        self.assertIn('Missing witness heating',check(916.))


if __name__=='__main__':unittest.main()
