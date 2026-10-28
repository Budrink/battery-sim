import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upv_gas_access import candidate,shape_report,rise_front,front_concentration_error,CHANNELS,FEATURE_REFERENCE_EVENT_S


class FeatureWindows(unittest.TestCase):
    def test_later_experiment_uses_same_relative_windows_without_moving_curves(self):
        t=np.arange(870.,951.,.5)
        def values(at):return 300.+80./(1.+np.exp(-(at-887.)/2.))
        def report(shift):
            run=dict(time_s=(t+shift).tolist(),temperature_k={k:values(t).tolist()for k in CHANNELS})
            def observations(data,at):return {k:values(np.asarray(at)-shift)for k in CHANNELS}
            with patch('upv_gas_access.dataio.event',return_value=dict(time_s=FEATURE_REFERENCE_EVENT_S+shift)),\
                 patch('upv_gas_access.dataio.observed',side_effect=observations):
                return shape_report(run,{})
        a,b=report(0.),report(526.5)
        for key in ('shape_objective','front_concentration_objective','shape_with_front_objective'):
            self.assertAlmostEqual(a[key],b[key],places=11)
        self.assertEqual(a['time_shift_s'],0.)
        self.assertEqual(b['time_shift_s'],0.)
        for key in CHANNELS:
            self.assertAlmostEqual(a['channels'][key]['peak_c'],b['channels'][key]['peak_c'],places=10)
            for front in ('calculated_rise_front','observed_rise_front'):
                self.assertAlmostEqual(b['channels'][key][front]['t50_s']-a['channels'][key][front]['t50_s'],526.5,places=10)

    def test_archived_training_shape_numbers_are_unchanged(self):
        import gzip
        import upv_r123_data as dataio
        root=Path(__file__).resolve().parents[1]/'data/calibration/upv-separated-chemistry-v6/final-training-calibration-20261022-r3/residual-contact-search/trial-000'
        if not(root/'assessment.json').exists():self.skipTest('Optional archived current-core calibration')
        old=json.loads((root/'assessment.json').read_text())['shape']
        with gzip.open(root/'replay/run.json.gz','rt')as stream:run=json.load(stream)
        data,_=dataio.load('no-cooling');new=shape_report(run,data)
        for key in ('shape_objective','front_concentration_objective','shape_with_front_objective','channels'):
            self.assertEqual(old[key],new[key],key)


class AccessGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'data/calibration/upv-separated-chemistry-v6/cooling-fit-20261021-r5/trial-000/scenario.json'
        if not path.exists():raise unittest.SkipTest('Optional local frozen scenario')
        cls.source=json.loads(path.read_text())

    def test_frozen_physics_and_inventory(self):
        before=copy.deepcopy(self.source);s=candidate(self.source,.001)
        self.assertEqual(self.source,before)
        for key in ['cells','materials','shells','interfaces','ports','boundaries','measurement_probes']:
            if key in before:self.assertEqual(s[key],before[key],key)
        for key in before['gas_network']:
            if key!='channels':self.assertEqual(s['gas_network'][key],before['gas_network'][key],key)

    def test_no_diffusion_bypass_through_a_backed_fracture(self):
        s=candidate(self.source,.001);old={c['id']:c for c in self.source['gas_network']['channels']}
        selected=0
        for e in s['gas_network']['channels']:
            o=old[e['id']]
            if e['kind']=='rupture' and e['to'].startswith('gap-TIM2-'):
                selected+=1
                self.assertEqual(e['damage_opening'],o['damage_opening'])
                self.assertEqual(e['opening_pressure_pa'],o['opening_pressure_pa'])
                self.assertAlmostEqual(e['discharge_coefficient']/o['discharge_coefficient'],.001)
                self.assertAlmostEqual(o['mixing_length_m']/e['mixing_length_m'],.001)
        self.assertEqual(selected,9)

    def test_open_routes_and_nonlocal_chamber_untouched(self):
        s=candidate(self.source,.001);old={e['id']:e for e in self.source['gas_network']['channels']}
        for e in s['gas_network']['channels']:
            if e['id'].startswith(('escape-ISS1-','escape-ISS2-','void-face-','pore-')):
                self.assertEqual(e,old[e['id']])

    def test_invalid_and_repeated_application_rejected(self):
        for f in [0.,-1.,float('nan'),1.1]:
            with self.assertRaises(ValueError):candidate(self.source,f)
        for key in ['lip_open_fraction','backing_open_fraction','plume_exit_fraction','witness_edge_open_fraction','jet_wetted_fraction','outer_insulation_open_fraction','iss3_accessible_fraction','open_iss_pad_wetted_fraction']:
            for value in [0.,float('nan'),1.1]:
                with self.assertRaises(ValueError):candidate(self.source,.001,**{key:value})
        with self.assertRaises(ValueError):candidate(candidate(self.source,.1),.1)

    def test_unit_access_has_identical_physics(self):
        new=candidate(self.source,1.)
        self.assertEqual(new['gas_network'],self.source['gas_network'])
        for key in ['cells','shells','interfaces','ports','boundaries','measurement_probes']:
            self.assertEqual(new[key],self.source[key])

    def test_lip_is_geometric_area_not_a_material_or_response_refit(self):
        new=candidate(self.source,.001,lip_open_fraction=.01)
        old=self.source['gas_network']['thermal_contacts']
        for a,b in zip(old,new['gas_network']['thermal_contacts']):
            changed=a!=b
            protected=any(a['port_id'].startswith('opening-contact:'+p+':')for p in ['TIM1','TIM2','TIM3','ISS3'])
            self.assertEqual(changed,protected)
            if not changed:continue
            self.assertEqual(a['gas_surface_connectivity'],b['gas_surface_connectivity'])
            self.assertEqual(a['solid_surface_connectivity'],b['solid_surface_connectivity'])
            for key,value in a['connectivity'].items():
                if key.endswith('conductance_w_per_k'):self.assertAlmostEqual(b['connectivity'][key],value*.01)
                else:self.assertEqual(value,b['connectivity'][key])
        self.assertEqual(new['interfaces'],self.source['interfaces'])

    def test_missing_measurement_does_not_erase_model_peak(self):
        t=np.array([875.,880.,881.,882.,884.,886.,887.,888.,890.,900.,915.,930.,950.])
        observed=np.array([300.,300.,300.,350.,600.,1300.,np.nan,1000.,800.,700.,600.,500.,400.])
        predicted=np.array([300.,300.,300.,340.,590.,900.,1400.,1000.,790.,690.,590.,490.,390.])
        data={'time_s':t};run=dict(time_s=t.tolist(),temperature_k={k:predicted.tolist()for k in CHANNELS})
        def observations(_data,at):return {k:np.interp(at,t,observed)for k in CHANNELS}
        with patch('upv_gas_access.dataio.observed',side_effect=observations),patch('upv_gas_access.dataio.event',return_value={'time_s':886.}):
            report=shape_report(run,data)
        self.assertEqual(report['channels']['C2_ISS']['calculated_rise_k'],1100.)
        self.assertEqual(report['channels']['C2_ISS']['excitation_penalty'],0.)
        changed=copy.deepcopy(run)
        for k in CHANNELS:changed['temperature_k'][k][-1]+=5000.
        with patch('upv_gas_access.dataio.observed',side_effect=observations),patch('upv_gas_access.dataio.event',return_value={'time_s':886.}):
            late=shape_report(changed,data)
            cold=shape_report(dict(time_s=t.tolist(),temperature_k={k:[300.]*len(t)for k in CHANNELS}),data)
        self.assertEqual(report['shape_objective'],late['shape_objective'])
        self.assertTrue(cold['no_heat_rejected'])

    def test_collar_narrows_shared_faces_without_new_hubs_or_inventory(self):
        new=candidate(self.source,.001,plume_exit_fraction=.03)
        changes=new['metadata']['gas_access_geometry']['plume_collar_faces']
        self.assertTrue(changes)
        old={e['id']:e for e in self.source['gas_network']['channels']}
        self.assertEqual(new['gas_network']['compartments'],self.source['gas_network']['compartments'])
        self.assertEqual(len(new['gas_network']['channels']),len(old))
        for row in changes:
            self.assertTrue(row['channel'].startswith('void-face-'))
            self.assertAlmostEqual(row['open_area_m2'],row['nominal_face_area_m2']*.03)
        self.assertEqual(new['ports'],self.source['ports'])

    def test_backing_cover_is_shared_and_does_not_square_existing_lip(self):
        new=candidate(self.source,.001,lip_open_fraction=.01,backing_open_fraction=.01)
        ports={p['id']:p for p in new['ports']};selected=set()
        for a,b in zip(self.source['gas_network']['thermal_contacts'],new['gas_network']['thermal_contacts']):
            owner=ports[a['port_id']]['cell_id']
            if owner not in ('Al1','Al2','Al3'):continue
            selected.add(owner)
            for k in a['connectivity']:
                if k.endswith('conductance_w_per_k'):self.assertAlmostEqual(b['connectivity'][k],a['connectivity'][k]*.01)
                else:self.assertEqual(a['connectivity'][k],b['connectivity'][k])
        self.assertEqual(selected,{'Al1','Al2','Al3'})
        for key in ['cells','shells','interfaces']:self.assertEqual(new[key],self.source[key])

    def test_witness_edges_share_coverage_but_c2_and_broad_faces_stay(self):
        new=candidate(self.source,.001,witness_edge_open_fraction=.1)
        ports={p['id']:p for p in new['ports']};selected=set()
        for a,b in zip(self.source['gas_network']['thermal_contacts'],new['gas_network']['thermal_contacts']):
            owner=ports[a['port_id']]['cell_id']
            covered=owner.startswith(('C1:case:','C3:case:'))and owner[-1]in ('y','z')
            self.assertEqual(a!=b,covered)
            if covered:
                selected.add(owner.split(':')[0])
                for k in AREA_KEYS:
                    self.assertAlmostEqual(b['connectivity'][k],a['connectivity'][k]*.1)
        self.assertEqual(selected,{'C1','C3'})

    def test_hot_footprint_is_shared_by_both_walls_and_both_iss_slots(self):
        new=candidate(self.source,.001,jet_wetted_fraction=.5)
        ports={p['id']:p for p in new['ports']};selected=set()
        for a,b in zip(self.source['gas_network']['thermal_contacts'],new['gas_network']['thermal_contacts']):
            covered=a['compartment'].startswith(('gap-ISS1-','gap-ISS2-'))
            self.assertEqual(a!=b,covered)
            if covered:
                selected.add(ports[a['port_id']]['cell_id'])
                for k in AREA_KEYS:self.assertAlmostEqual(b['connectivity'][k],a['connectivity'][k]*.5)
                self.assertEqual(a['gas_surface_connectivity'],b['gas_surface_connectivity'])
                self.assertEqual(a['solid_surface_connectivity'],b['solid_surface_connectivity'])
        self.assertEqual(selected,{'ISS1','ISS2','C3:case:-x','C2:case:-x'})
        self.assertEqual(new['interfaces'],self.source['interfaces'])

    def test_common_pad_footprint_leaves_facing_casings_and_solid_laws_unchanged(self):
        new=candidate(self.source,1.,open_iss_pad_wetted_fraction=.5)
        ports={p['id']:p for p in new['ports']};selected=set()
        for a,b in zip(self.source['gas_network']['thermal_contacts'],new['gas_network']['thermal_contacts']):
            owner=ports[a['port_id']]['cell_id']
            covered=owner in ('ISS1','ISS2')and a['compartment'].startswith('gap-'+owner+'-')
            self.assertEqual(a!=b,covered)
            if not covered:continue
            selected.add(owner)
            for k in AREA_KEYS:self.assertAlmostEqual(b['connectivity'][k],a['connectivity'][k]*.5)
            self.assertEqual(a['gas_surface_connectivity'],b['gas_surface_connectivity'])
            self.assertEqual(a['solid_surface_connectivity'],b['solid_surface_connectivity'])
        self.assertEqual(selected,{'ISS1','ISS2'})
        for key in ['cells','shells','interfaces','ports','boundaries','measurement_probes']:
            self.assertEqual(new[key],self.source[key])
        for key in ['channels','compartments']:self.assertEqual(new['gas_network'][key],self.source['gas_network'][key])
        neutral=candidate(self.source,.001,open_iss_pad_wetted_fraction=1.)
        self.assertEqual(neutral,candidate(self.source,.001))

    def test_outer_insulation_uses_same_cover_without_squaring_al_exposure(self):
        controls=dict(lip_open_fraction=.001,backing_open_fraction=.01,
            witness_edge_open_fraction=.1)
        old=candidate(self.source,.001,**controls)
        new=candidate(self.source,.001,outer_insulation_open_fraction=.001,**controls)
        ports={p['id']:p for p in new['ports']};selected=set()
        for a,b in zip(old['gas_network']['thermal_contacts'],new['gas_network']['thermal_contacts']):
            owner=ports[a['port_id']]['cell_id']
            covered=a['port_id'].startswith('opening-contact:ISS4:')
            self.assertEqual(a!=b,covered)
            if not covered:continue
            selected.add(owner)
            factor=.1 if owner=='Al3' else .001
            for k in AREA_KEYS:self.assertAlmostEqual(b['connectivity'][k],a['connectivity'][k]*factor)
            self.assertEqual(a['gas_surface_connectivity'],b['gas_surface_connectivity'])
            self.assertEqual(a['solid_surface_connectivity'],b['solid_surface_connectivity'])
        self.assertEqual(selected,{'ISS4','Al3','steel-R'})
        for key in ['cells','materials','shells','interfaces','ports','boundaries']:
            if key in old:self.assertEqual(new[key],old[key])
        for key in ['compartments','channels']:
            self.assertEqual(new['gas_network'][key],old['gas_network'][key])
        self.assertIn('ISS4',new['metadata']['gas_access_geometry']['protected_slots'])

    def test_outer_cover_invalid_or_unit_geometry(self):
        self.assertEqual(candidate(self.source,1.,outer_insulation_open_fraction=1.)['gas_network'],
            self.source['gas_network'])

    def test_outer_rim_can_be_less_covered_than_compressed_internal_slots(self):
        a=candidate(self.source,.001,lip_open_fraction=.001,backing_open_fraction=.01)
        b=candidate(self.source,.001,lip_open_fraction=.001,backing_open_fraction=.01,
            outer_insulation_open_fraction=.5)
        ports={p['id']:p for p in b['ports']}
        for old,new in zip(a['gas_network']['thermal_contacts'],b['gas_network']['thermal_contacts']):
            selected=old['port_id'].startswith('opening-contact:ISS4:')
            changed=selected and ports[old['port_id']]['cell_id']!='Al3'
            self.assertEqual(old!=new,changed)
            if changed:
                for k in AREA_KEYS:self.assertAlmostEqual(new['connectivity'][k],old['connectivity'][k]*.5)

    def test_iss3_adjusts_mass_access_not_thermal_contact_gain(self):
        old=candidate(self.source,.001)
        new=candidate(self.source,.001,iss3_accessible_fraction=.003)
        changed=0
        for a,b in zip(old['gas_network']['channels'],new['gas_network']['channels']):
            affected=a['id'].startswith('escape-ISS3-')or(a['kind']=='rupture'and a['to'].startswith('gap-ISS3-'))
            self.assertEqual(a!=b,affected)
            if a['id'].startswith('escape-ISS3-'):
                self.assertAlmostEqual(b['area_m2'],3*a['area_m2']);changed+=1
            if a['kind']=='rupture'and a['to'].startswith('gap-ISS3-'):
                self.assertAlmostEqual(b['discharge_coefficient'],3*a['discharge_coefficient'])
                self.assertAlmostEqual(b['mixing_length_m'],a['mixing_length_m']/3)
        self.assertEqual(changed,12)
        self.assertEqual(old['gas_network']['thermal_contacts'],new['gas_network']['thermal_contacts'])
        self.assertEqual(old['gas_network']['compartments'],new['gas_network']['compartments'])
        for k in ['cells','shells','interfaces','ports']:self.assertEqual(old[k],new[k])


class FrontDiagnostics(unittest.TestCase):
    def test_real_pulse_is_not_rewarded_for_being_smoothed_away(self):
        smooth=front_concentration_error(5.,20.,12.5,20.)
        pulse=front_concentration_error(13.,20.,12.5,20.)
        self.assertGreater(smooth,pulse)
        self.assertAlmostEqual(smooth,front_concentration_error(10.,40.,12.5,20.))
        self.assertAlmostEqual(pulse,front_concentration_error(13.,20.,25.,40.))

    def test_front_has_no_fitted_time_shift_or_temperature_gain(self):
        row=rise_front([10.,11.,12.,13.],[300.,301.,302.,303.],300.,3.)
        self.assertAlmostEqual(row['t10_s'],10.3)
        self.assertAlmostEqual(row['t50_s'],11.5)
        self.assertAlmostEqual(row['t90_s'],12.7)
        self.assertAlmostEqual(row['rise_10_90_s'],2.4)

    def test_front_does_not_interpolate_through_nan(self):
        row=rise_front([0.,1.,2.,3.],[300.,np.nan,302.,303.],300.,3.)
        self.assertIsNone(row['t10_s'])
        self.assertIsNone(row['t50_s'])
        self.assertAlmostEqual(row['t90_s'],2.7)
        self.assertIsNone(row['rise_10_90_s'])


AREA_KEYS=('background_conductance_w_per_k','active_conductance_w_per_k',
    'destroyed_background_conductance_w_per_k','destroyed_active_conductance_w_per_k')

if __name__=='__main__':unittest.main()
