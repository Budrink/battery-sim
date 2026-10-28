"""Fresh calibration contract: no old profile reads and no source-temperature drive."""
import copy
import gzip
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import upv_r123_data as dataio
import upv_r123_model as model
import calibrate_upv_r123 as calibration


class CalibrationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene,cls.meta=model.build('no-cooling',warmup=True)

    def test_solid_and_boundary_heat_uses_directed_s_links(self):
        for edge in self.scene['interfaces']:
            self.assertEqual(edge['law']['kind'],'directed_geometry')
        for boundary in self.scene['boundaries']:
            self.assertEqual(boundary['connectivity']['kind'],'directed_geometry')
            self.assertIn('remote_surface_connectivity',boundary)
        cooled,_=model.build('active-cooling',warmup=True)
        water=next(b for b in cooled['boundaries'] if b['id']=='water')
        self.assertEqual(water['connectivity']['kind'],'directed_geometry')
        plate=next(e for e in cooled['interfaces'] if e['id'].endswith('--cooling'))
        self.assertEqual(plate['law']['kind'],'directed_geometry')
        self.assertEqual(plate['law']['destroyed_active_conductance_w_per_k'],0.)
        self.assertLess(plate['law']['active_conductance_w_per_k'],10000.*0.09)

    def test_primary_cell_is_solved_and_not_forced(self):
        scene=self.scene;cells={c['id']:c for c in scene['cells']}
        self.assertIn('C2',cells)
        self.assertEqual(cells['C2']['thermal']['divisions'],[3,3,3])
        self.assertIn('electrical',cells['C2'])
        self.assertFalse(scene['primary_failures'])
        self.assertFalse(any(b['id'].startswith('source-') for b in scene['boundaries']))
        self.assertTrue(all('conversion' in cells[name] for name in ('C1','C2','C3')))
        self.assertFalse(any('temperature_profile' in b for b in scene['boundaries']))

    def test_r2_gate_is_late_and_joule_ignores_post_gate_c2(self):
        gate=calibration.r2_gate_s(self.meta['data'],self.meta['event']['time_s'])
        self.assertGreater(gate,450.)
        self.assertLess(gate,520.)
        times=np.arange(0.,self.meta['event']['time_s']-20.,5.)
        obs=dataio.observed(self.meta['data'],times)
        run=dict(time_s=times.tolist(),temperature_k={k:obs[k].tolist() for k in dataio.CHANNELS},
                 gas=dict(chamber=dict(pressure_pa=(obs['pressure_bar']*1e5).tolist())))
        late=copy.deepcopy(run)
        for i,t in enumerate(times):
            if t>gate:
                late['temperature_k']['C2_TIM'][i]+=30.
                late['temperature_k']['C2_ISS'][i]+=30.
        matched=float(calibration.residual(run,self.meta,'heat',protocol='joule')@
                      calibration.residual(run,self.meta,'heat',protocol='joule'))
        joule=float(calibration.residual(late,self.meta,'heat',protocol='joule')@
                    calibration.residual(late,self.meta,'heat',protocol='joule'))
        full=float(calibration.residual(late,self.meta,'heat')@calibration.residual(late,self.meta,'heat'))
        self.assertAlmostEqual(joule,matched,places=6)
        self.assertGreater(full,joule+10.)

    def test_early_cold_is_joule_late_cold_is_r2(self):
        times=np.arange(0.,self.meta['event']['time_s']-20.,5.)
        obs=dataio.observed(self.meta['data'],times)
        gate=calibration.r2_gate_s(self.meta['data'],self.meta['event']['time_s'])
        def run_from(shift):
            return dict(time_s=times.tolist(),
                        temperature_k={k:(obs[k]+shift).tolist() for k in dataio.CHANNELS},
                        gas=dict(chamber=dict(pressure_pa=(obs['pressure_bar']*1e5).tolist())),
                        cell_mean_temperature_k={'C2':(.5*(obs['C2_TIM']+obs['C2_ISS'])+shift).tolist()})
        early=run_from(-12.)
        self.assertEqual(calibration.diagnose_heat(early,self.meta)['action'],'joule')
        late=run_from(0.)
        for i,t in enumerate(times):
            if t>gate:
                for name in ('C2_TIM','C2_ISS'):
                    late['temperature_k'][name][i]-=12.
                late['cell_mean_temperature_k']['C2'][i]-=12.
        self.assertEqual(calibration.diagnose_heat(late,self.meta)['action'],'phi_r2')
        self.assertTrue(set(('r2_rate','r2_activation','energy_j')).issubset(calibration.PROTOCOL['phi_r2']))
        self.assertNotIn('resistance_ohm',calibration.PROTOCOL['phi_r2'])

    def test_electrical_envelope_uses_committed_snapshot_clock(self):
        raw,_=dataio.electrical_record();marker=self.meta['event']['time_s']
        t=raw[100,0]+marker-raw[-1,0];u=raw[100,1];i=raw[100,2]
        run=dict(time_s=[t+5.],electrical={'C2':dict(snapshot_time_s=[t],current_a=[i],resistance_ohm=[(u-4.)/i])})
        np.testing.assert_allclose(calibration.electrical_envelope_residual(run,self.meta),0.)
        run['electrical']['C2']['resistance_ohm']=[.0001]
        self.assertGreater(np.linalg.norm(calibration.electrical_envelope_residual(run,self.meta)),0.)
        run['electrical']['C2']['current_a']=[0.]
        np.testing.assert_allclose(calibration.electrical_envelope_residual(run,self.meta),0.)

    def test_final_resistance_guard_does_not_suppress_measured_spike(self):
        law=next(c['electrical'] for c in self.scene['cells'] if c['id']=='C2')
        self.assertGreater(law['max_resistance_ohm'],1.3)

    def test_first_pressure_jump_is_not_scored_as_quiet_warmup(self):
        event=self.meta['event']['time_s'];t=[0.,self.meta['event']['previous_s'],event,event+1.]
        obs=dataio.observed(self.meta['data'],np.asarray(t))
        run=dict(time_s=t,temperature_k={k:obs[k].tolist() for k in dataio.CHANNELS},
                 gas=dict(chamber=dict(pressure_pa=(obs['pressure_bar']*1e5).tolist())))
        baseline=calibration.residual(run,self.meta,'failure')
        run['gas']['chamber']['pressure_pa'][2]+=1e5
        changed=calibration.residual(run,self.meta,'failure')
        # A one-bar error on the first jump belongs to the .25-bar event
        # window (2 samples), not to the .03-bar quiet pressure window.
        self.assertAlmostEqual(float(np.sum(changed**2)-np.sum(baseline**2)),8.,places=7)

    def test_casing_contact_uses_state_table_without_material_changes(self):
        altered,_=model.build('no-cooling',{'case_warm_response':-2.,'case_hot_response':3.},warmup=True)
        for a,b in zip(self.scene['cells'],altered['cells']):
            self.assertEqual(a['thermal']['heat_capacity_j_per_k'],b['thermal']['heat_capacity_j_per_k'])
            self.assertEqual(a['thermal']['conductivity_w_per_m_k'],b['thermal']['conductivity_w_per_m_k'])
        cell=next(c for c in altered['cells'] if c['id']=='C2')
        table=cell['thermal']['surface_connectivity']['temperature_log_response']
        self.assertEqual(table,[[298.15,0.],[330.,-2.],[390.,3.]])
        self.assertTrue(all(s['surface_connectivity']['temperature_log_response']==table for s in altered['shells']))

    def test_material_properties_not_fitted(self):
        a=self.scene;b,_=model.build('no-cooling',dict(model.DEFAULT,resistance_ohm=.07,iss_h=9000.,tim_h=10.,surface_iss=3.),warmup=True)
        for x,y in zip(a['cells'],b['cells']):
            self.assertEqual(x['geometry'],y['geometry'])
            self.assertEqual(x['thermal']['heat_capacity_j_per_k'],y['thermal']['heat_capacity_j_per_k'])
            self.assertEqual(x['thermal']['conductivity_w_per_m_k'],y['thermal']['conductivity_w_per_m_k'])
        for name in ('C1','C2','C3'):
            c=next(v for v in a['cells'] if v['id']==name);outer=[.011,.087,.187];inner=c['geometry']['dimensions_m']
            case=(math.prod(outer)-math.prod(inner))*1600.*1250.
            self.assertAlmostEqual(c['thermal']['heat_capacity_j_per_k']+case,.410*1007.)

    def test_electrical_record_is_not_ui_heating(self):
        values,meta=dataio.electrical_record()
        self.assertGreater(len(values),1000)
        self.assertAlmostEqual(values[0,2],34.9)
        self.assertEqual(values[-1,2],0.)
        self.assertIn('sha256',meta)
        self.assertFalse(self.scene['metadata']['electrical']['synchronization_measured'])
        self.assertTrue(self.scene['metadata']['electrical']['electrical_power_not_used_as_heat'])
        e=next(v['electrical'] for v in self.scene['cells'] if v['id']=='C2')
        self.assertNotIn('power_w',e)

    def test_gas_reservoir_volume_and_casing(self):
        gas=self.scene['gas_network']
        self.assertAlmostEqual(sum(n['volume_m3'] for n in gas['compartments'] if not n['id'].startswith('inside-')),.120)
        self.assertEqual(gas['mass_backend'],'cuda_network')
        self.assertEqual(sum(n['id'].startswith('inside-') for n in gas['compartments']),81)
        self.assertTrue(all(q['damage_opening']['mode']=='deterministic_aperture' for q in gas['channels'] if q['kind']=='rupture'))

    def test_overcharge_is_explicit_and_local_to_source(self):
        cells={c['id']:c for c in self.scene['cells']}
        self.assertEqual(cells['C2']['electrical']['charge_acceptance_fraction'],1.)
        self.assertEqual(cells['C2']['electrical']['nominal_capacity_coulomb'],35.*3600.)
        for name in ('C1','C3'):
            self.assertNotIn('electrical',cells[name])
        self.assertGreater(cells['C2']['conversion']['charge_conversion_per_nominal_charge'],0.)
        self.assertIn('voltage_profile_v',cells['C2']['electrical'])
        self.assertGreater(len(cells['C2']['electrical']['voltage_profile_v']),10)
        self.assertEqual(cells['C2']['conversion']['precursor_yield'],1.)
        self.assertIn('precursor_yield',calibration.HEAT)
        self.assertNotIn('r1_rate',calibration.HEAT)
        self.assertIn('charge_formation_gain',calibration.HEAT)
        self.assertIn('resistance_charge',calibration.HEAT)
        self.assertIn('resistance_charge',calibration.LINEAR)
        self.assertIn('resistance_charge',calibration.PROTOCOL['joule'])
        self.assertIn('resistance_charge',calibration.PROTOCOL['onset10'])
        self.assertEqual(cells['C2']['electrical']['charge_log_slope'],0.)
        self.assertEqual(model.DEFAULT['resistance_charge'],0.)
        self.assertEqual(model.BOUNDS['resistance_charge'][0],0.)

    def test_gas_yield_temperature_is_identified(self):
        self.assertIn('gas_yield_temperature',calibration.HEAT)
        self.assertIn('gas_yield_temperature',calibration.LINEAR)
        lo,hi=model.BOUNDS['gas_yield_temperature']
        self.assertEqual(lo,0.)
        # Upper bound is condensed-phase chemistry at the R2 reference, not an
        # expanded gas inventory: E_eff = bT R T^2 < 150 kJ/mol at 450 K.
        self.assertLess(hi*8.31446261815324*450.**2,150000.)
        self.assertGreater(hi,.03)
        cells={c['id']:c for c in self.scene['cells']}
        self.assertEqual(cells['C2']['conversion']['gas_yield_temperature_per_k'],model.DEFAULT['gas_yield_temperature'])
        other,_=model.build('no-cooling',dict(model.DEFAULT,gas_yield_temperature=.03),warmup=True)
        self.assertAlmostEqual(next(c for c in other['cells'] if c['id']=='C2')['conversion']['gas_yield_temperature_per_k'],.03)

    def test_initial_voltage_rejects_resistance_compensation(self):
        calibration.check_initial_voltage(self.scene)
        law=next(c for c in self.scene['cells'] if c['id']=='C2')['electrical']
        self.assertEqual(law['reversible_voltage_floor_v'],0.)
        allowed=copy.deepcopy(self.scene)
        next(c for c in allowed['cells'] if c['id']=='C2')['electrical']['resistance_ohm']=.07
        calibration.check_initial_voltage(allowed)
        impossible=copy.deepcopy(self.scene)
        next(c for c in impossible['cells'] if c['id']=='C2')['electrical']['resistance_ohm']=.2
        with self.assertRaisesRegex(ValueError,'Initial I\\*R'):
            calibration.check_initial_voltage(impossible)

    def test_r3_has_independent_arrhenius_everywhere(self):
        for node in self.scene['gas_network']['compartments']:
            self.assertEqual(node['reaction_reference_k'],600.)
            self.assertEqual(node['reaction_activation_j_per_mol'],80000.)
            if node['id'].startswith('inside-'):
                self.assertEqual(node['initial_oxidizer_mass_fraction'],0.)
        self.assertIn('gas_activation',calibration.FAILURE)
        self.assertNotIn('gas_activation',calibration.HEAT)

    def test_validation_current_does_not_heat_precharge_rest(self):
        raw,_=dataio.load('active-cooling');mark=dataio.event(raw)
        drive,voltage,meta=dataio.current_drive('active-cooling',raw,mark)
        self.assertEqual(drive[0],[0.,0.])
        self.assertGreater(meta['onset_s'],466.)
        self.assertLess(meta['onset_s'],467.)
        self.assertEqual(drive[1][1],0.)
        self.assertTrue(meta['no_temperature_time_shift_or_fit'])

    def test_partial_error_bound_never_exceeds_full_score(self):
        path=ROOT/'data/calibration/upv-r123-arrhenius-v3/r3-only-fixed-phi-final/heat-run.json.gz'
        if not path.exists():self.skipTest('Local replay fixture is not present')
        with gzip.open(path,'rt') as stream:run=json.load(stream)
        profile=json.loads((path.parent/'protocol.json').read_text())['parameters']
        # This fixture tests objective arithmetic, not replay of retired R(T,D).
        profile.pop('resistance_damage',None)
        from migrate_upv_chemistry_profile import migrate
        profile=migrate(profile)
        scene,meta=model.build('no-cooling',profile,warmup=True)
        score=float(calibration.residual(run,meta,'heat')@calibration.residual(run,meta,'heat'))
        previous=0.
        for n in (20,60,120,len(run['time_s'])):
            partial=dict(time_s=run['time_s'][:n],temperature_k={k:v[:n] for k,v in run['temperature_k'].items()},
                         gas=dict(chamber=dict(pressure_pa=run['gas']['chamber']['pressure_pa'][:n])))
            bound=calibration.partial_objective_lower_bound(partial,scene,meta,'heat')
            self.assertGreaterEqual(bound,previous)
            self.assertLessEqual(bound,score*(1+1e-9))
            previous=bound
        self.assertAlmostEqual(previous,score,places=8)

    def local_fixture(self,times):
        observed=dataio.observed(self.meta['data'],np.asarray(times))
        return dict(time_s=times,temperature_k={name:observed[name].tolist() for name in dataio.CHANNELS})

    def test_sustained_local_divergence_is_pruned(self):
        reference=self.local_fixture([270.,280.,290.,300.]);run=copy.deepcopy(reference)
        run['temperature_k']['C2_TIM']=[t+60. for t in run['temperature_k']['C2_TIM']]
        reason=calibration.local_trajectory_rejection(run,reference,self.meta,'heat')
        self.assertIn('C2_TIM',reason)
        self.assertIn('300.000 s',reason)
        self.assertIsNone(calibration.local_trajectory_rejection(reference,reference,self.meta,'heat'))

    def test_electrical_products_and_explicit_geometry(self):
        c=next(c for c in self.scene['cells'] if c['id']=='C2')
        self.assertEqual(c['electrical']['terminal_axis'],'z')
        self.assertEqual(c['electrical']['product_conductivity_ratio'],.2)
        self.assertNotIn('damage_log_slope',c['electrical'])
        with self.assertRaisesRegex(ValueError,'Old R'):
            model.build('no-cooling',{'resistance_damage':1.},warmup=True)

    def test_charging_and_thermal_degradation_are_not_a_paid_fraction(self):
        c=next(c for c in self.scene['cells'] if c['id']=='C2')
        self.assertNotIn('overcharge_formation_fraction',c['electrical'])
        self.assertNotIn('overcharge_pay',model.DEFAULT)
        self.assertNotIn('overcharge_pay',calibration.HEAT)
        with self.assertRaisesRegex(ValueError,'Retired overcharge_pay'):
            model.build('no-cooling',{'overcharge_pay':.25},warmup=True)

    def test_joule_hypothesis_is_separate_from_measured_temperature_error(self):
        run=dict(time_s=[0.,1.],chemistry={'C1':{},'C2':{},'C3':{}},
                 released_energy_j_history=[0.,0.,0.,1.,29.,0.],
                 joule_energy_j={'C1':[0.,0.],'C2':[0.,60.],'C3':[0.,0.]},
                 gas={'chamber':{'released_energy_j':[0.,10.]}})
        sources=calibration.heat_sources(run)
        self.assertEqual(sources['joule_share'],.6)
        self.assertEqual(sources['r2_local_j'],30.)
        self.assertEqual(calibration.source_prior_residual(run,.5),0.)
        self.assertAlmostEqual(calibration.source_prior_residual(run,.7),5.)

    def test_proposal_obeys_linearized_source_fraction(self):
        step=calibration.fit_increment(np.eye(2),np.array([-1.,-1.]),np.array([-2.,-2.]),np.array([2.,2.]),
                                       np.array([1.,-1.]),.5)
        self.assertGreaterEqual(step[0]-step[1],.5-1e-8)
        np.testing.assert_allclose(step,[1.25,.75],atol=1e-5)

    def test_local_guard_ignores_one_spike_and_broken_source_sensor(self):
        reference=self.local_fixture([270.,280.,290.,300.]);run=copy.deepcopy(reference)
        run['temperature_k']['C2_TIM'][-1]+=1000.
        self.assertIsNone(calibration.local_trajectory_rejection(run,reference,self.meta,'heat'))
        event=self.meta['event']['time_s'];reference=self.local_fixture([event+50.,event+60.,event+70.])
        run=copy.deepcopy(reference)
        run['temperature_k']['C2_TIM']=[t+1000. for t in run['temperature_k']['C2_TIM']]
        self.assertIsNone(calibration.local_trajectory_rejection(run,reference,self.meta,'failure'))

    def test_failure_prefix_bound_uses_full_phase_denominators(self):
        scene,meta=model.build('no-cooling',duration=self.meta['event']['time_s']+80.)
        times=np.asarray(scene['failure_sample_times_s']);obs=dataio.observed(meta['data'],times)
        run=dict(time_s=times.tolist(),temperature_k={name:(obs[name]+7.).tolist() for name in dataio.CHANNELS},
                 gas=dict(chamber=dict(pressure_pa=((obs['pressure_bar']+.05)*1e5).tolist())))
        r=calibration.residual(run,meta,'failure');score=float(r@r);previous=0.
        for n in (20,len(times)//2,len(times)):
            partial=dict(time_s=run['time_s'][:n],temperature_k={k:v[:n] for k,v in run['temperature_k'].items()},
                         gas=dict(chamber=dict(pressure_pa=run['gas']['chamber']['pressure_pa'][:n])))
            bound=calibration.partial_objective_lower_bound(partial,scene,meta,'failure')
            self.assertGreaterEqual(bound,previous)
            self.assertLessEqual(bound,score*(1+1e-9));previous=bound
        self.assertAlmostEqual(previous,score,places=8)

    def test_pruning_terminates_and_reaps_native_child(self):
        child=Mock();child.poll.return_value=None
        def checkpoint(request):
            destination=json.loads(request)['checkpoint_path']
            Path(destination).write_text(json.dumps(dict(time_s=[0.,100.])))
        child.stdin.write.side_effect=checkpoint
        with tempfile.TemporaryDirectory() as directory,patch.object(calibration.subprocess,'Popen',return_value=child):
            with self.assertRaisesRegex(RuntimeError,'Pruned: test divergence'):
                calibration.replay(self.scene,Path(directory),'test',60.,reject_partial=lambda run:'test divergence')
            saved=json.loads((Path(directory)/'last-pruned.json').read_text())
            self.assertEqual(saved['last_sample_s'],100.)
            self.assertEqual(saved['reason'],'test divergence')
        child.terminate.assert_called_once()
        self.assertEqual(child.wait.call_args.kwargs,dict(timeout=5))

    def test_charge_onset_is_current_not_clock_zero(self):
        onset=dataio.charge_onset_s('no-cooling',self.meta['data'],self.meta['event'])
        self.assertGreater(onset,26.)
        self.assertLess(onset,28.)
        cooling,_=dataio.load('active-cooling');mark=dataio.event(cooling)
        self.assertGreater(dataio.charge_onset_s('active-cooling',cooling,mark),466.)
        self.assertLess(dataio.charge_onset_s('active-cooling',cooling,mark),467.)

    def test_native_failure_retains_completed_observations(self):
        for returncode in (0,1):
            with self.subTest(returncode=returncode),tempfile.TemporaryDirectory() as directory:
                child=Mock();child.poll.return_value=returncode;child.returncode=returncode
                def launch(*args,**kwargs):
                    kwargs['stdout'].write(json.dumps(dict(error='native rejected candidate')))
                    kwargs['stdout'].flush()
                    kwargs['stderr'].write('native rejected candidate');kwargs['stderr'].flush()
                    return child
                def checkpoint(request):
                    destination=json.loads(request)['checkpoint_path']
                    Path(destination).write_text(json.dumps(dict(time_s=[0.,100.],advanced_to_s=101.)))
                child.stdin.write.side_effect=checkpoint
                with patch.object(calibration.subprocess,'Popen',side_effect=launch):
                    with self.assertRaisesRegex(RuntimeError,'native rejected candidate'):
                        calibration.replay(self.scene,Path(directory),'test',60.)
                with gzip.open(Path(directory)/'last-incomplete-run.json.gz','rt') as stream:
                    saved=json.load(stream)
                self.assertTrue(saved['incomplete'])
                self.assertEqual(saved['time_s'],[0.,100.])
                self.assertEqual(saved['advanced_to_s'],101.)

    def test_onset10_scores_only_first_ten_charge_seconds(self):
        onset=dataio.charge_onset_s('no-cooling',self.meta['data'],self.meta['event'])
        times=np.arange(0.,onset+30.,.5)
        obs=dataio.observed(self.meta['data'],times)
        run=dict(time_s=times.tolist(),temperature_k={k:obs[k].tolist() for k in dataio.CHANNELS},
                 gas=dict(chamber=dict(pressure_pa=(obs['pressure_bar']*1e5).tolist())))
        meta=dict(self.meta,experiment='no-cooling')
        matched=float(calibration.residual(run,meta,'heat',protocol='onset10')@
                      calibration.residual(run,meta,'heat',protocol='onset10'))
        late=copy.deepcopy(run)
        for i,t in enumerate(times):
            if t>onset+calibration.ONSET_WINDOW_S:
                late['temperature_k']['C2_ISS'][i]+=30.
                late['gas']['chamber']['pressure_pa'][i]+=1e5
        late_score=float(calibration.residual(late,meta,'heat',protocol='onset10')@
                         calibration.residual(late,meta,'heat',protocol='onset10'))
        self.assertAlmostEqual(late_score,matched,places=6)
        offset=copy.deepcopy(run)
        for i,t in enumerate(times):
            if onset<=t<=onset+calibration.ONSET_WINDOW_S:
                offset['temperature_k']['C2_ISS'][i]+=1.
        offset_score=float(calibration.residual(offset,meta,'heat',protocol='onset10')@
                           calibration.residual(offset,meta,'heat',protocol='onset10'))
        self.assertAlmostEqual(offset_score,matched,places=6)
        ramp=copy.deepcopy(run)
        for i,t in enumerate(times):
            if onset<=t<=onset+calibration.ONSET_WINDOW_S:
                ramp['temperature_k']['C2_ISS'][i]+=(t-onset)/calibration.ONSET_WINDOW_S
        ramp_score=float(calibration.residual(ramp,meta,'heat',protocol='onset10')@
                         calibration.residual(ramp,meta,'heat',protocol='onset10'))
        self.assertGreater(ramp_score,matched+5.)

    def test_onset10_uses_daq_samples_and_does_not_move_chemistry(self):
        self.assertEqual(calibration.PROTOCOL['onset10'][0],'resistance_ohm')
        self.assertIn('iss_h',calibration.PROTOCOL['onset10'])
        self.assertIn('environment_h',calibration.PROTOCOL['onset10'])
        self.assertNotIn('precursor_yield',calibration.PROTOCOL['onset10'])
        self.assertNotIn('r2_activation',calibration.PROTOCOL['onset10'])
        self.assertNotIn('energy_j',calibration.PROTOCOL['onset10'])
        self.assertIn('resistance_ohm',calibration.PROTOCOL['onset10'])
        self.assertTrue(set(('resistance_ohm','iss_h')).issubset(calibration.PROTOCOL['onset10']))
        onset=dataio.charge_onset_s('no-cooling',self.meta['data'],self.meta['event'])
        scene,meta=model.build('no-cooling',warmup=True,duration=onset+10.,dense_after=onset)
        times=np.asarray(scene['failure_sample_times_s'])
        self.assertEqual(meta['experiment'],'no-cooling')
        self.assertGreater(np.count_nonzero((times>=onset)&(times<=onset+10.)),15)
        self.assertLess(scene['simulation']['duration_s'],onset+10.1)

    def test_onset_window_can_extend_to_twenty_seconds(self):
        onset=dataio.charge_onset_s('no-cooling',self.meta['data'],self.meta['event'])
        times=np.arange(0.,onset+40.,.5)
        obs=dataio.observed(self.meta['data'],times)
        run=dict(time_s=times.tolist(),temperature_k={k:obs[k].tolist() for k in dataio.CHANNELS},
                 gas=dict(chamber=dict(pressure_pa=(obs['pressure_bar']*1e5).tolist())))
        meta=dict(self.meta,experiment='no-cooling')
        previous=calibration.ONSET_WINDOW_S
        try:
            calibration.set_onset_window(20.)
            matched=float(calibration.residual(run,meta,'heat',protocol='onset10')@
                          calibration.residual(run,meta,'heat',protocol='onset10'))
            mid=copy.deepcopy(run)
            for i,t in enumerate(times):
                if onset+10.<t<=onset+20.:
                    mid['temperature_k']['C2_ISS'][i]+=(t-onset)/20.
            mid_score=float(calibration.residual(mid,meta,'heat',protocol='onset10')@
                            calibration.residual(mid,meta,'heat',protocol='onset10'))
            self.assertGreater(mid_score,matched+1.)
            after=copy.deepcopy(run)
            for i,t in enumerate(times):
                if t>onset+20.:
                    after['temperature_k']['C2_ISS'][i]+=30.
            after_score=float(calibration.residual(after,meta,'heat',protocol='onset10')@
                              calibration.residual(after,meta,'heat',protocol='onset10'))
            self.assertAlmostEqual(after_score,matched,places=6)
        finally:
            calibration.ONSET_WINDOW_S=previous

    def test_onset_balance_treats_r2_as_off(self):
        report=calibration.onset_balance('no-cooling')
        self.assertTrue(report['r2_treated_as_off'])
        self.assertLess(report['r2_converted_fraction'],1e-3)
        self.assertGreater(report['r1_converted_fraction'],0.)
        self.assertLess(report['r1_converted_fraction'],1e-5)
        self.assertGreater(report['charging_rate_per_s'],report['r1_thermal_rate_per_s']*1000.)
        self.assertGreater(report['well_mixed_resistance_ohm'],.005)
        self.assertLess(report['well_mixed_resistance_ohm'],.0195)
        self.assertLess(report['isolated_c2_resistance_ohm'],report['well_mixed_resistance_ohm'])


if __name__=='__main__':unittest.main()
