import copy
import argparse
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import search_upv_destruction as search
import upv_destruction_objective as objective
import fit_upv_kinetics as fit


def seed():
    p=dict(fit.model.DEFAULT)
    p.update(r1_rate=fit.rate_at(1e-9,240000.,348.15,400.),r1_activation=240000.,
             r2_rate=fit.rate_at(1e-6,120000.,348.15,450.),r2_activation=120000.)
    return p


class SearchTests(unittest.TestCase):
    def test_driver_saves_best_and_compacts_rejected_fields_without_gpu(self):
        with tempfile.TemporaryDirectory(prefix='battery-search-test-')as tmp:
            root=Path(tmp);source=root/'seed.json'
            extension=dict(mixing_flow_fraction=.25)
            controls=dict(coupled_gas_films=True,coupled_transport_interval_s=.01)
            source.write_text(json.dumps(dict(parameters=seed(),model_fingerprint='test',
                gas_extensions=extension,numerical_controls=controls)))
            out=root/'result'
            def evaluate(proposal,base,directory,fp,timeout,**kwargs):
                self.assertTrue(kwargs['explosion']);self.assertIs(kwargs['objective_fn'],objective.score)
                self.assertEqual(kwargs['gas_extensions'],extension)
                self.assertEqual(kwargs['numerical_controls'],controls)
                folder=directory/proposal['label'];folder.mkdir()
                raw,_=fit.dataio.load('no-cooling');event=fit.dataio.event(raw)['time_s'];end=event+300.
                t=np.r_[np.arange(0.,end),end];obs=fit.dataio.observed(raw,t)
                # Missing instrument scans remain missing TARGETS, never NaN
                # simulated states (the native solver rejects those).
                predictions={k:np.interp(t,t[np.isfinite(obs[k])],obs[k][np.isfinite(obs[k])]).tolist()
                             for k in fit.dataio.CHANNELS}
                r=dict(time_s=t.tolist(),temperature_k=predictions,
                       gas={'chamber':{'pressure_pa':(obs['pressure_bar']*1e5).tolist()}})
                with gzip.open(folder/'trial-run.json.gz','wt')as f:json.dump(r,f)
                return dict(label=proposal['label'],parameters=fit.parameters(base,proposal,allow_destruction=True),
                            status='complete',objective=1.+int(proposal['label'][-4:]),last_sample_s=end)
            args=argparse.Namespace(output=out,seed=source,resume=False,levels=7,global_points=1,
                random_seed=123,max_trials=2,max_hours=1.,timeout=1.,max_output_gib=1.)
            with patch.object(fit.cal,'fingerprint',return_value='test'),patch.object(fit,'evaluate',side_effect=evaluate):
                search.run(args)
            state=json.loads((out/'search-state.json').read_text())
            self.assertEqual(state['status'],'trial_budget');self.assertEqual(state['trials'],2)
            self.assertTrue(state['incumbent']['feasible'])
            self.assertTrue((out/'trial-0000/trial-run.json.gz').exists())
            self.assertFalse((out/'trial-0001/trial-run.json.gz').exists())
            self.assertTrue((out/'trial-0001/observations.json.gz').exists())
            self.assertEqual(json.loads((out/'best-feasible.json').read_text())['label'],'trial-0000')

    def test_roundtrip_and_hard_bounds(self):
        for x in [np.zeros(len(search.KEYS)),np.ones(len(search.KEYS)),search.anchor_coordinates(seed())]:
            p=search.decode(x)
            for k,(lo,hi,_) in search.AXES.items():self.assertTrue(lo<=p[k]<=hi)
            np.testing.assert_allclose(search.encode(p),x,atol=1e-14)
        for v in [-.001,1.001,float('nan')]:
            x=np.zeros(len(search.KEYS));x[0]=v
            with self.assertRaises(ValueError):search.decode(x)

    def test_wide_reproducible_lattice_not_cartesian_explosion(self):
        a=search.make_plan(seed());b=search.make_plan(seed())
        self.assertEqual(a,b);self.assertGreater(len(a['coarse']),300)
        for item in a['coarse']:
            self.assertTrue(np.all(np.asarray(item['x'])>=0) and np.all(np.asarray(item['x'])<=1))
            if item['origin']=='space-filling-grid':
                np.testing.assert_allclose(np.asarray(item['x'])*6,np.rint(np.asarray(item['x'])*6),atol=1e-12)

    def test_success_follows_only_proven_coordinate_and_critical_failure_shrinks(self):
        state=search.make_plan(seed());x=search.anchor_coordinates(seed())
        row=dict(label='first',parameters=seed(),feasible=True,objective=10.,status='complete')
        search.feedback(state,dict(x=x.tolist(),axis=None),row,(0,10.,0.))
        i=search.KEYS.index('chamber_gas_h');y=x.copy();y[i]+=.1
        item=dict(x=y.tolist(),axis='chamber_gas_h')
        row=dict(row,label='better',objective=8.)
        self.assertTrue(search.feedback(state,item,row,(0,8.,0.)))
        self.assertEqual(state['gains'][i],1.)
        self.assertEqual(state['local'][0]['origin'],'continue-improving-axis')
        self.assertEqual(state['local'][0]['axis'],'chamber_gas_h')
        radius=state['radii'][i];bad=state['local'][0]
        self.assertFalse(search.feedback(state,bad,dict(row,feasible=False,status='pruned'),(1,-90,1.)))
        self.assertLess(state['radii'][i],radius)
        self.assertEqual(state['incumbent']['label'],'better')
        # A good partial prefix can NEVER replace a feasible full trajectory.
        self.assertFalse(search.feedback(state,item,dict(row,feasible=False),(1,-10000,0.)))

    def test_joint_gain_is_not_falsely_attributed_to_one_axis(self):
        state=search.make_plan(seed());x=search.anchor_coordinates(seed())
        row=dict(label='base',parameters=seed(),feasible=True,objective=10.,status='complete')
        search.feedback(state,dict(x=x.tolist(),axis=None),row,(0,10.,0.))
        x[0]+=.1;x[1]+=.1
        search.feedback(state,dict(x=x.tolist(),axis='k1_75'),dict(row,label='joint'),(0,8.,0.))
        self.assertEqual(state['gains'],[0.]*len(search.KEYS))

    def test_chamber_contact_is_independent_and_no_old_contact_changes(self):
        p=seed();a,_=fit.model.build('no-cooling',p,duration=1186.639)
        p['chamber_gas_h']=80.;b,_=fit.model.build('no-cooling',p,duration=1186.639)
        contacts=b['gas_network']['thermal_contacts']
        walls=[c for c in contacts if c['port_id'].startswith('chamber-wall:gas-partition:')]
        self.assertTrue(walls)
        for wall in walls:
            from upv_connectivity import contact, reference_gas_conductance
            area=next(port['area_m2']for port in b['ports']if port['id']==wall['port_id'])
            self.assertAlmostEqual(reference_gas_conductance(wall,area),80.*area)
            wall['connectivity']=contact(area,fit.model.DEFAULT['chamber_gas_h'],destroyed=None)
        b['metadata']['parameters']['chamber_gas_h']=fit.model.DEFAULT['chamber_gas_h']
        self.assertEqual(a,b)
        q=dict(label='wall',k1_75=1e-9,ea1=240000.,k2_75=1e-6,ea2=120000.,chamber_gas_h=80.)
        with self.assertRaises(ValueError):fit.parameters(seed(),q)
        updated=fit.parameters(seed(),q,allow_destruction=True)
        self.assertEqual(updated['environment_h'],seed()['environment_h'])


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(401.)
        self.meta=dict(event=dict(time_s=100.),fit_end_s=400.,data={'time_s':self.t})
        def observed(_data,t):
            t=np.asarray(t);result={k:np.full(len(t),300.)for k in fit.dataio.CHANNELS}
            result['108']+=1000*np.exp(-.5*((t-103.)/1.2)**2)
            result['pressure_bar']=1+6*np.exp(-.5*((t-103.)/1.2)**2)
            return result
        self.observed=observed;self.mock=patch.object(fit.dataio,'observed',side_effect=observed);self.mock.start()
        o=observed({},self.t)
        self.trajectory=dict(time_s=self.t.tolist(),temperature_k={k:o[k].tolist()for k in fit.dataio.CHANNELS},
                      gas={'chamber':{'pressure_pa':(o['pressure_bar']*1e5).tolist()}})

    def tearDown(self):self.mock.stop()

    def prefix(self,r,end):
        n=end+1
        return dict(time_s=r['time_s'][:n],temperature_k={k:v[:n]for k,v in r['temperature_k'].items()},
                    gas={'chamber':{'pressure_pa':r['gas']['chamber']['pressure_pa'][:n]}})

    def test_short_burst_is_softened_but_missing_peak_costs_nonzero(self):
        missing=copy.deepcopy(self.trajectory);missing['temperature_k']['108']=[300.]*len(self.t)
        self.assertIsNone(objective.reject_reason(self.prefix(missing,110),self.meta))
        self.assertGreater(objective.loss(missing,self.meta)['peak_features']['108'],0.)
        self.assertEqual(objective.loss(self.trajectory,self.meta)['objective'],0.)

    def test_matching_cooling_beats_overfitting_uncertain_temperature_peak(self):
        peak=copy.deepcopy(self.trajectory);tail=copy.deepcopy(self.trajectory)
        target=np.asarray(peak['temperature_k']['108'])
        peak['temperature_k']['108']=(300.+.5*(target-300.)).tolist()
        for i in range(160,401):tail['temperature_k']['108'][i]+=20.
        self.assertLess(objective.loss(peak,self.meta)['objective'],objective.loss(tail,self.meta)['objective'])
        self.assertGreater(objective.loss(peak,self.meta)['peak_features']['108'],0.)

    def test_soft_thermocouple_policy_does_not_soften_pressure(self):
        bad=copy.deepcopy(self.trajectory)
        p=np.asarray(bad['gas']['chamber']['pressure_pa'])
        bad['gas']['chamber']['pressure_pa']=(1e5+.5*(p-1e5)).tolist()
        expected=objective.loss(bad,self.meta)['objective']
        with patch.dict(objective.TEMPERATURE_PHASE_WEIGHTS,{k:0. for k in objective.TEMPERATURE_PHASE_WEIGHTS}), \
             patch.object(objective,'TEMPERATURE_PEAK_WEIGHT',0.):
            self.assertAlmostEqual(objective.loss(bad,self.meta)['objective'],expected)

    def test_persistent_pre_and_tail_departures_are_stopped(self):
        for end,key,lo in [(94,'101',60),(300,'108',200)]:
            bad=copy.deepcopy(self.trajectory)
            for i in range(lo,end+1):bad['temperature_k'][key][i]+=90.
            self.assertIn('temperature critical zone',objective.reject_reason(self.prefix(bad,end),self.meta))

    def test_isolated_outlier_not_a_persistent_critical_zone(self):
        bad=copy.deepcopy(self.trajectory);bad['temperature_k']['101'][90]+=400.
        self.assertIsNone(objective.reject_reason(self.prefix(bad,94),self.meta))

    def test_partial_loss_is_lower_bound_and_never_a_full_fit(self):
        bad=copy.deepcopy(self.trajectory);bad['temperature_k']['101']=[312.]*len(self.t)
        prefix=self.prefix(bad,80)
        self.assertLessEqual(objective.loss(prefix,self.meta)['objective'],objective.loss(bad,self.meta)['objective'])
        self.assertIsNone(objective.score(prefix,self.meta))
        with patch.object(fit.cal,'metrics',return_value={}):
            self.assertIsNotNone(objective.score(bad,self.meta))
            self.assertIsNone(objective.score(dict(bad,partial=True),self.meta))

    def test_final_scan_catches_divergence_missed_between_checkpoints(self):
        bad=copy.deepcopy(self.trajectory)
        for i in range(60,96):bad['temperature_k']['101'][i]+=90.
        self.assertIsNone(objective.reject_reason(bad,self.meta))
        self.assertIsNotNone(search.first_violation(bad,self.meta))

    def test_physical_safety_is_not_softened_at_the_peak(self):
        bad=self.prefix(self.trajectory,103);bad['gas']['chamber']['pressure_pa'][-1]=50e5
        self.assertIn('critical pressure',objective.reject_reason(bad,self.meta))


if __name__=='__main__':unittest.main()
