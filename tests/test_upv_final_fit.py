import copy
import json
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upv_final_fit import candidate,only_gas_contacts_changed

SOURCE=Path(__file__).resolve().parents[1]/'data/calibration/upv-separated-chemistry-v6/five-channel-through-runaway-20261022-r2/trial-000/scenario.json'

class FinalFit(unittest.TestCase):
    def test_wall_knee_changes_only_warm_ramp_and_is_inherited_from_actual_scene(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text())
        trial,p=candidate(source,dict(wall_hot_knee_k=700.))
        self.assertTrue(only_gas_contacts_changed(source,trial))
        self.assertFalse(trial['metadata']['final_calibration_trial']['fresh_required'])
        for a,b in zip(source['gas_network']['thermal_contacts'],trial['gas_network']['thermal_contacts']):
            expected=copy.deepcopy(a)
            if a['port_id'].startswith('chamber-wall:'):
                expected['gas_surface_connectivity']['temperature_log_response'][-1][0]=700.
            self.assertEqual(expected,b)
        inherited,_=candidate(trial,{})
        self.assertEqual(inherited['gas_network'],trial['gas_network'])

    def test_product_conductivity_refit_is_electrical_only_and_stays_below_intact(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text())
        trial,p=candidate(source,dict(product_conductivity_scale=3.))
        expected=copy.deepcopy(source)
        for cell in expected['cells']:
            if 'electrical'in cell:cell['electrical']['product_conductivity_ratio']*=3.
        expected.pop('metadata');actual=copy.deepcopy(trial);actual.pop('metadata')
        self.assertEqual(expected,actual)
        self.assertTrue(trial['metadata']['final_calibration_trial']['fresh_required'])
        for cell in source['cells']:
            if 'electrical'in cell:cell['electrical']['product_conductivity_ratio']=.8
        with self.assertRaisesRegex(ValueError,'product conductivity'):
            candidate(source,dict(product_conductivity_scale=2.))

    def test_electrical_refit_changes_only_declared_law_coefficients(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text())
        trial,p=candidate(source,dict(resistance_scale=1.02,
            resistance_temperature_delta=.003,resistance_charge_delta=-1.))
        expected=copy.deepcopy(source)
        for cell in expected['cells']:
            if 'electrical'not in cell:continue
            law=cell['electrical'];law['resistance_ohm']*=1.02
            law['temperature_log_slope_per_k']+=.003
            law['charge_log_slope']-=1.
        expected.pop('metadata');physical=copy.deepcopy(trial);physical.pop('metadata')
        self.assertEqual(expected,physical)
        self.assertTrue(trial['metadata']['final_calibration_trial']['fresh_required'])

    def test_electrical_bounds_are_on_actual_scene_not_relative_metadata(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text())
        for cell in source['cells']:
            if 'electrical'in cell:cell['electrical']['resistance_ohm']=.019
        with self.assertRaisesRegex(ValueError,'Reference resistance'):
            candidate(source,dict(resistance_scale=1.1))

    def test_r2_activation_refit_keeps_mass_energy_and_reference_temperature(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text())
        trial,p=candidate(source,dict(r2_activation_delta=-20000.))
        expected=copy.deepcopy(source)
        for cell in expected['cells']:
            if 'conversion'in cell:
                cell['conversion']['reaction_activation_j_per_mol']-=20000.
        expected.pop('metadata');actual=copy.deepcopy(trial);actual.pop('metadata')
        self.assertEqual(expected,actual)
        self.assertTrue(trial['metadata']['final_calibration_trial']['fresh_required'])

    def test_warmer_r2_fit_must_preserve_the_existing_cold_storage_prior(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text())
        with self.assertRaisesRegex(ValueError,'stored charge'):
            candidate(source,dict(r2_activation_delta=-40000.,r2_rate=1.))

    def test_damaged_contact_changes_keep_intact_law_and_wall_exact(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());trial,p=candidate(source,dict(damaged_span=.4,damaged_logit=2.))
        owners={v['id']:v['cell_id']for v in source['ports']};count=0
        for a,b in zip(source['gas_network']['thermal_contacts'],trial['gas_network']['thermal_contacts']):
            expected=copy.deepcopy(a)
            if ':case:'in owners[a['port_id']]:
                expected['connectivity']['destroyed_active_conductance_w_per_k']*=.4
                expected['connectivity']['destroyed_geometry_reference_logit']+=2.;count+=1
            self.assertEqual(expected,b)
        self.assertGreater(count,0)
        self.assertEqual(source['gas_network']['channels'],trial['gas_network']['channels'])
        self.assertEqual(source['gas_network']['surface_connectivity'],trial['gas_network']['surface_connectivity'])

    def test_opening_and_path_do_not_change_materials_current_energy(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());before=copy.deepcopy(source)
        trial,p=candidate(source,dict(entry=.003,path=2.,opening_rate=.01,strength=3.))
        self.assertEqual(source,before)
        for key in ('cells','shells','interfaces','boundaries'):self.assertEqual(source[key],trial[key])
        self.assertFalse(trial['metadata']['final_calibration_trial']['fresh_required'])
        for a,b in zip(source['gas_network']['channels'],trial['gas_network']['channels']):
            if a['kind']=='rupture':
                self.assertAlmostEqual(b['opening_pressure_pa'],a['opening_pressure_pa']*3.)
                self.assertAlmostEqual(b['damage_opening']['area_rate_per_s'],a['damage_opening']['area_rate_per_s']*.01)

    def test_material_constants_never_changed_by_fresh_overlay(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());trial,p=candidate(source,dict(case_h=2.,bulk_metric=.8,resistance_charge_delta=-.5,r2_rate=3.))
        self.assertTrue(trial['metadata']['final_calibration_trial']['fresh_required'])
        for a,b in zip(source['cells'],trial['cells']):
            self.assertEqual(a['thermal']['heat_capacity_j_per_k'],b['thermal']['heat_capacity_j_per_k'])
            self.assertEqual(a['thermal']['conductivity_w_per_m_k'],b['thermal']['conductivity_w_per_m_k'])
            if 'electrical'in a:
                self.assertEqual(a['electrical']['current_profile_a'],b['electrical']['current_profile_a'])
                self.assertEqual(a['electrical']['voltage_profile_v'],b['electrical']['voltage_profile_v'])
            if 'conversion'in a:self.assertEqual(a['conversion']['net_heat_j_per_kg'],b['conversion']['net_heat_j_per_kg'])

    def test_partition_keeps_total_energy_and_frozen_local_power(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());trial,p=candidate(source,dict(mobile_energy_fraction=.5,opening_rate=.01))
        self.assertTrue(trial['metadata']['final_calibration_trial']['fresh_required'])
        for a,b in zip(source['cells'],trial['cells']):
            if 'conversion'not in a:continue
            x,y=a['conversion'],b['conversion']
            self.assertEqual(x['net_heat_j_per_kg'],y['net_heat_j_per_kg'])
            self.assertEqual(x['gas_yield'],y['gas_yield'])
            self.assertAlmostEqual(x['reaction_rate_per_s']*(1.-x['gas_chemical_fraction']),y['reaction_rate_per_s']*(1.-y['gas_chemical_fraction']))
        self.assertEqual(source['shells'],trial['shells'])

    def test_default_fraction_comes_from_actual_scene(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text())
        for c in source['cells']:
            if 'conversion'in c:c['conversion']['gas_chemical_fraction']=.6
        trial,p=candidate(source,{})
        self.assertFalse(trial['metadata']['final_calibration_trial']['fresh_required'])
        for a,b in zip(source['cells'],trial['cells']):self.assertEqual(a,b)

    def test_common_jet_footprint_is_extensive_geometry_only(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());trial,p=candidate(source,dict(jet_fraction=.7))
        for key in ('cells','shells','interfaces','boundaries'):self.assertEqual(source[key],trial[key])
        count=0
        for a,b in zip(source['gas_network']['thermal_contacts'],trial['gas_network']['thermal_contacts']):
            expected=copy.deepcopy(a)
            if any(a['compartment'].startswith('gap-'+p+'-')for p in ('ISS1','ISS2')):
                for key in ('background_conductance_w_per_k','active_conductance_w_per_k','destroyed_background_conductance_w_per_k','destroyed_active_conductance_w_per_k'):
                    expected['connectivity'][key]*=.7;count+=1
            self.assertEqual(expected,b)
        self.assertGreater(count,0)

    def test_core_damage_response_does_not_boost_reverse_or_bulk(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());trial,p=candidate(source,dict(core_contact_damage=6.))
        self.assertTrue(trial['metadata']['final_calibration_trial']['fresh_required'])
        self.assertEqual(source['shells'],trial['shells'])
        self.assertEqual(source['gas_network'],trial['gas_network'])
        for a,b in zip(source['cells'],trial['cells']):
            expected=copy.deepcopy(a)
            if a.get('role')=='battery':expected['thermal']['surface_connectivity']['damage']=[6.,6.,6.]
            self.assertEqual(expected,b)

    def test_casing_state_changes_only_its_donor_profile(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());trial,p=candidate(source,dict(casing_contact_damage=-1.5))
        self.assertTrue(trial['metadata']['final_calibration_trial']['fresh_required'])
        self.assertEqual(source['cells'],trial['cells'])
        self.assertEqual(source['gas_network'],trial['gas_network'])
        for a,b in zip(source['shells'],trial['shells']):
            expected=copy.deepcopy(b);expected['surface_connectivity']=a['surface_connectivity']
            self.assertEqual(a,expected)
            self.assertEqual(b['surface_connectivity']['temperature_log_response'],a['surface_connectivity']['temperature_log_response'])
            self.assertEqual(b['surface_connectivity']['damage'],[-1.5,-1.5,-1.5])

    def test_witness_mouth_changes_existing_gas_routes_only(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());trial,p=candidate(source,dict(witness_mouth=.00675))
        for key in ('cells','shells','interfaces','boundaries'):self.assertEqual(source[key],trial[key])
        self.assertEqual(source['gas_network']['thermal_contacts'],trial['gas_network']['thermal_contacts'])
        self.assertEqual(source['gas_network']['compartments'],trial['gas_network']['compartments'])
        changed=0
        for a,b in zip(source['gas_network']['channels'],trial['gas_network']['channels']):
            if a==b:continue
            self.assertTrue(a['id'].startswith('escape-ISS3-')or(a['kind']=='rupture'and a['to'].startswith('gap-ISS3-')))
            changed+=1
        self.assertGreater(changed,0)

    def test_hot_wall_law_keeps_cold_reference_and_inventory(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());trial,p=candidate(source,dict(wall_hot_scale=1.3))
        for key in ('cells','shells','interfaces','boundaries'):self.assertEqual(source[key],trial[key])
        self.assertEqual(source['gas_network']['channels'],trial['gas_network']['channels'])
        self.assertEqual(source['gas_network']['compartments'],trial['gas_network']['compartments'])
        count=0
        for a,b in zip(source['gas_network']['thermal_contacts'],trial['gas_network']['thermal_contacts']):
            if not a['port_id'].startswith('chamber-wall:'):
                self.assertEqual(a,b);continue
            count+=1
            def g(c,i):
                law=c['connectivity'];z=law['geometry_reference_logit']+c['gas_surface_connectivity']['temperature_log_response'][i][1]
                return law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1+math.exp(-z))
            for i,(t,y)in enumerate(a['gas_surface_connectivity']['temperature_log_response']):
                self.assertAlmostEqual(g(b,i)/g(a,i),1. if t<=450 else 1.3,places=13)
        self.assertGreater(count,0)

    def test_hot_reverse_law_keeps_forward_heating_and_cold_cooling(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text())
        for delta in (-3., .25, .5):
            trial,p=candidate(source,dict(reverse_hot_logit=delta))
            self.assertTrue(only_gas_contacts_changed(source,trial))
            for key in ('cells','shells','interfaces','boundaries'):self.assertEqual(source[key],trial[key])
            for a,b in zip(source['gas_network']['thermal_contacts'],trial['gas_network']['thermal_contacts']):
                if a['port_id'].startswith('chamber-wall:'):
                    self.assertEqual(a,b);continue
                expected=copy.deepcopy(b);expected['solid_surface_connectivity']=a['solid_surface_connectivity']
                self.assertEqual(a,expected)
                old=dict(a['solid_surface_connectivity']['temperature_log_response']);new=dict(b['solid_surface_connectivity']['temperature_log_response'])
                self.assertLessEqual(len(new),8)
                for t,y in old.items():self.assertAlmostEqual(new[t]-y,delta if t>=450. else 0. if t<=375. else delta*(t-375.)/75.)

    def test_casing_contrast_is_shared_and_zero_for_equal_damage(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());trial,p=candidate(source,dict(casing_damage_contrast=-6.))
        self.assertTrue(trial['metadata']['final_calibration_trial']['fresh_required'])
        self.assertEqual(source['cells'],trial['cells'])
        self.assertEqual(source['gas_network'],trial['gas_network'])
        for a,b in zip(source['shells'],trial['shells']):
            expected=copy.deepcopy(b);expected['surface_connectivity']=a['surface_connectivity']
            self.assertEqual(a,expected)
            beta=b['surface_connectivity']['damage_contrast'][0]
            self.assertEqual(beta*(1.-1.),0.)
            self.assertEqual(beta*(1.-0.),-6.)

    def test_parameter_pivot_cannot_smuggle_solid_or_gas_route_edits(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text())
        films,_=candidate(source,dict(backing_fraction=.03,wall_hot_scale=.9,reverse_hot_logit=-5.))
        self.assertTrue(only_gas_contacts_changed(source,films))
        changed,_=candidate(source,dict(core_contact_damage=6.))
        self.assertFalse(only_gas_contacts_changed(source,changed))

    def test_power_reverse_is_monotone_and_keeps_forward_reference(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());trial,p=candidate(source,dict(reverse_profile_power=3.))
        self.assertTrue(only_gas_contacts_changed(source,trial))
        for a,b in zip(source['gas_network']['thermal_contacts'],trial['gas_network']['thermal_contacts']):
            if a['port_id'].startswith('chamber-wall:'):
                self.assertEqual(a,b);continue
            expected=copy.deepcopy(b);expected['solid_surface_connectivity']=a['solid_surface_connectivity']
            self.assertEqual(a,expected)
            profile=b['solid_surface_connectivity'];table=profile['temperature_log_response']
            self.assertEqual(dict(table)[profile['reference_temperature_k']],0.)
            self.assertTrue(all(y2>=y1 for (t1,y1),(t2,y2)in zip(table,table[1:])))
        with self.assertRaises(ValueError):candidate(source,dict(reverse_profile_power=3.,reverse_hot_logit=-1.))

    def test_gas_damage_leaves_every_intact_temperature_curve_unchanged(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());trial,p=candidate(source,dict(casing_gas_damage=-5.))
        self.assertTrue(only_gas_contacts_changed(source,trial))
        owners={v['id']:v['cell_id']for v in source['ports']};changed=0
        for a,b in zip(source['gas_network']['thermal_contacts'],trial['gas_network']['thermal_contacts']):
            expected=copy.deepcopy(a)
            if ':case:'in owners[a['port_id']]:
                expected['solid_surface_connectivity']['damage']=[-5.,-5.,-5.];changed+=1
            self.assertEqual(expected,b)
        self.assertGreater(changed,0)
        changed,_=candidate(source,dict(witness_mouth=.00675))
        self.assertFalse(only_gas_contacts_changed(source,changed))

    def test_residual_iss_contact_is_common_and_leaves_intact_materials_fixed(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text())
        trial,p=candidate(source,dict(damaged_iss_scale=.1))
        self.assertTrue(trial['metadata']['final_calibration_trial']['fresh_required'])
        for key in ('cells','shells','gas_network','boundaries'):
            self.assertEqual(source[key],trial[key])
        owners={v['id']:v['cell_id']for v in source['ports']};changed=0
        for a,b in zip(source['interfaces'],trial['interfaces']):
            expected=copy.deepcopy(a)
            pair=[owners[a[k]]for k in ('port_a','port_b')]
            if any(n.startswith('ISS')for n in pair)and any(':case:'in n for n in pair):
                for key in ('destroyed_background_conductance_w_per_k','destroyed_active_conductance_w_per_k'):
                    expected['law'][key]*=.1
                changed+=1
            self.assertEqual(expected,b)
        self.assertEqual(changed,3)
        self.assertFalse(only_gas_contacts_changed(source,trial))

    def test_core_surface_warm_response_never_changes_bulk_or_reverse(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());trial,p=candidate(source,dict(core_surface_power=4.))
        self.assertTrue(trial['metadata']['final_calibration_trial']['fresh_required'])
        for key in ('interfaces','shells','gas_network','boundaries'):
            self.assertEqual(source[key],trial[key])
        for a,b in zip(source['cells'],trial['cells']):
            expected=copy.deepcopy(b)
            if a.get('role')=='battery':
                expected['thermal']['surface_connectivity']=a['thermal']['surface_connectivity']
                profile=b['thermal']['surface_connectivity']
                self.assertEqual(dict(profile['temperature_log_response'])[profile['reference_temperature_k']],0.)
                ys=[y for t,y in profile['temperature_log_response']]
                self.assertEqual(ys,sorted(ys))
            self.assertEqual(a,expected)

    def test_iss_gradient_response_is_shared_and_bulk_is_frozen(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text())
        trial,p=candidate(source,dict(intact_iss_scale=3.,iss_temperature_contrast=-4.))
        self.assertTrue(trial['metadata']['final_calibration_trial']['fresh_required'])
        for key in ('shells','gas_network','boundaries'):
            self.assertEqual(source[key],trial[key])
        count=0
        for a,b in zip(source['cells'],trial['cells']):
            expected=copy.deepcopy(b)
            if a['id'].startswith('ISS'):
                self.assertEqual(b['thermal']['bulk_response'],a['thermal'].get('bulk_response',a['thermal']['surface_connectivity']))
                self.assertEqual(b['thermal']['bulk_connectivity'],a['thermal'].get('bulk_connectivity',dict(
                    kind='directed_geometry',background_factor=.01,active_factor=1.98,
                    destroyed_background_factor=.01,destroyed_active_factor=1.98,endpoint_damage_weight=1.)))
                self.assertEqual(b['thermal']['surface_connectivity']['temperature_contrast'],[-4.,-4.,-4.])
                expected['thermal']=a['thermal'];count+=1
            self.assertEqual(a,expected)
        self.assertEqual(count,4)
        owners={v['id']:v['cell_id']for v in source['ports']};count=0
        for a,b in zip(source['interfaces'],trial['interfaces']):
            expected=copy.deepcopy(a);pair=[owners[a[k]]for k in ('port_a','port_b')]
            if any(n.startswith('ISS')for n in pair)and any(':case:'in n for n in pair):
                for key in ('background_conductance_w_per_k','active_conductance_w_per_k'):
                    expected['law'][key]*=3.
                count+=1
            self.assertEqual(expected,b)
        self.assertEqual(count,3)

    def test_failed_tim_calibration_leaves_intact_pair_and_other_materials_exact(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());trial,p=candidate(source,dict(damaged_tim_scale=.2))
        self.assertTrue(trial['metadata']['final_calibration_trial']['fresh_required'])
        for key in ('cells','shells','gas_network','boundaries'):
            self.assertEqual(source[key],trial[key])
        owners={v['id']:v['cell_id']for v in source['ports']};count=0
        for a,b in zip(source['interfaces'],trial['interfaces']):
            expected=copy.deepcopy(a);pair=[owners[a[k]]for k in('port_a','port_b')]
            if any(n.startswith('TIM')for n in pair)and any(':case:'in n for n in pair):
                for key in ('destroyed_background_conductance_w_per_k','destroyed_active_conductance_w_per_k'):
                    expected['law'][key]*=.2
                count+=1
            self.assertEqual(expected,b)
        self.assertEqual(count,3)

    def test_al_surface_is_common_bounded_and_does_not_change_bulk_material(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());trial,p=candidate(source,dict(al_surface_softening=2.))
        self.assertTrue(trial['metadata']['final_calibration_trial']['fresh_required'])
        for key in ('interfaces','shells','gas_network','boundaries'):
            self.assertEqual(source[key],trial[key])
        profiles=[]
        for a,b in zip(source['cells'],trial['cells']):
            if a.get('material_id')!='aluminium':
                self.assertEqual(a,b);continue
            expected=copy.deepcopy(b);expected['thermal']=a['thermal'];self.assertEqual(a,expected)
            for key in ('heat_capacity_j_per_k','conductivity_w_per_m_k','divisions'):
                self.assertEqual(a['thermal'][key],b['thermal'][key])
            self.assertEqual(b['thermal']['bulk_response'],a['thermal']['surface_connectivity'])
            surface=b['thermal']['surface_connectivity'];profiles.append(surface)
            table=surface['temperature_log_response']
            self.assertLessEqual(len(table),8)
            self.assertEqual(dict(table)[surface['reference_temperature_k']],0.)
            self.assertTrue(all(y>=z for (t,y),(u,z)in zip(table,table[1:])))
            self.assertTrue(all(-2.<=y<=.01 for t,y in table))
        self.assertEqual(len(profiles),3)
        self.assertTrue(all(p==profiles[0]for p in profiles))

    def test_tim_al_contact_scale_is_shared_and_keeps_materials_and_failed_law(self):
        if not SOURCE.exists():self.skipTest('Optional local current reference')
        source=json.loads(SOURCE.read_text());trial,p=candidate(source,dict(tim_al_scale=6.))
        self.assertTrue(trial['metadata']['final_calibration_trial']['fresh_required'])
        for key in ('cells','shells','gas_network','boundaries'):
            self.assertEqual(source[key],trial[key])
        owners={v['id']:v['cell_id']for v in source['ports']}
        aluminium={c['id']for c in source['cells']if c.get('material_id')=='aluminium'};count=0
        for a,b in zip(source['interfaces'],trial['interfaces']):
            expected=copy.deepcopy(a);pair=[owners[a[k]]for k in('port_a','port_b')]
            if any(n.startswith('TIM')for n in pair)and any(n in aluminium for n in pair):
                for key in ('background_conductance_w_per_k','active_conductance_w_per_k'):
                    expected['law'][key]*=6.
                count+=1
            self.assertEqual(expected,b)
        self.assertEqual(count,3)

if __name__=='__main__':unittest.main()
