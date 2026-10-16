"""Post-event protocol guards; no parameter changes to accepted intact model."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from calibrate_upv_failure import (PRIOR,CHANNELS,load_record,event_marker,resample,
        post_scene,bake,check_parameters,audit,refresh_boundaries,observations)
    from calibrate_upv import Worker
finally:sys.path.pop(0)

def fixture():
    base=json.loads((PRIOR/'no-cooling-fitted-scenario.json').read_text());d,_=load_record('no-cooling')
    warm={'final_fields_k':{c['id']:[c['thermal']['initial_temperature_k']]*int(np.prod(c['thermal']['divisions'])) for c in base['cells']}}
    return base,post_scene(base,warm,d,event_marker(d),3.)

class FailureTests(unittest.TestCase):
    def test_event_brackets_and_source_quality(self):
        for name,expected in [('no-cooling',886.639),('active-cooling',1413.248)]:
            d,_=load_record(name);e=event_marker(d)
            self.assertAlmostEqual(e['first_event_s'],expected,places=2)
            self.assertLess(e['bracket_width_s'],1.)
            self.assertTrue(all(70<v<140 for v in e['source_pre_event_c'].values()))
            d['U114_v'][:]=.42
            with self.assertRaises(ValueError):event_marker(d)

    def test_masking_no_gap_filling_or_extrapolation(self):
        t=np.arange(5.);y=np.array([1.,2.,np.nan,4.,5.])
        v=resample(t,y,[.5,1.,1.5,2.5,3.,4.])
        self.assertEqual(v[[0,1,4,5]].tolist(),[1.5,2.,4.,5.])
        self.assertTrue(np.isnan(v[[2,3]]).all())
        with self.assertRaises(ValueError):resample(t,y,[-.01])
        with self.assertRaises(ValueError):resample(t,y,[5.])
        self.assertTrue(np.isnan(resample([0,1,2,20,21],[1,2,3,4,5],[10])[0]))

    def test_reconstructed_source_and_frozen_materials(self):
        base,s=fixture();c=next(c for c in s['cells'] if c['id']=='C2')
        self.assertEqual(c['thermal']['divisions'],[3,3,3]);self.assertEqual(len(c['thermal']['initial_temperature_field_k']),27)
        self.assertEqual(len(s['cells']),len(base['cells'])+1)
        self.assertEqual(len(s['interfaces']),len(base['interfaces'])+2)
        self.assertFalse(any(b['id'].startswith('source-') for b in s['boundaries']))
        self.assertEqual(s['interfaces'][:len(base['interfaces'])],base['interfaces'])
        self.assertEqual({p['id'] for p in s['measurement_probes']},set(CHANNELS))
        for old,new in zip(base['cells'],s['cells']):
            t=copy.deepcopy(new['thermal']);t.pop('initial_temperature_field_k');t['initial_temperature_k']=old['thermal']['initial_temperature_k']
            self.assertEqual(t,old['thermal'])
        p=dict(energy_j=120000.,duration_s=8.,destroyed_iss_h=20.,destroyed_tim_h=500.)
        changed=bake(s,p);changed['cells'][-1]['runaway']=s['cells'][-1]['runaway']
        for old,new in zip(s['interfaces'],changed['interfaces']):
            if old['id'].startswith('failure-C2-'):new['law']['destroyed_background_conductance_w_per_k']=old['law']['destroyed_background_conductance_w_per_k']
        self.assertEqual(s,changed)

    def test_parameter_block(self):
        p=dict(energy_j=120000.,duration_s=8.,destroyed_iss_h=20.,destroyed_tim_h=500.);check_parameters(p)
        for key,value in [('capacity_45',1.),('energy_j',-1.),('duration_s',121.),('destroyed_tim_h',float('nan'))]:
            q=dict(p);q[key]=value
            with self.assertRaises(ValueError):check_parameters(q)

    def test_original_scans_and_invalid_source_readings_preserved(self):
        d,_=load_record('no-cooling');event=event_marker(d);start=event['last_pre_event_s']
        t,y=observations(d,start,300.)
        select=(d['time_s']>=start)&(d['time_s']<=start+300.)
        np.testing.assert_array_equal(t[:-1],d['time_s'][select]-start)
        np.testing.assert_array_equal(y['101'][:-1],d['T101_k'][select])
        expected=(d['T105_k'][select]+d['T106_k'][select])/2
        np.testing.assert_array_equal(y['C2-ISS'][:-1],expected)
        self.assertTrue(np.isnan(y['C2-ISS']).any());self.assertEqual(t[-1],300.)

    @unittest.skipUnless(os.environ.get('BATTERY_SIM_FAILURE_PROBE'),'CUDA failure probe not requested')
    def test_cuda_replay_energy_switch_export_and_guards(self):
        _,s=fixture();p=dict(energy_j=20000.,duration_s=1.,destroyed_iss_h=20.,destroyed_tim_h=500.)
        with tempfile.TemporaryDirectory(prefix='failure-test-') as tmp:
            path=Path(tmp)/'scene.json';path.write_text(json.dumps(s));w=Worker(Path(os.environ['BATTERY_SIM_FAILURE_PROBE']),path)
            try:
                run=w.run(p);audit(run,s,p)
                self.assertAlmostEqual(run['released_energy_j'],20000.,places=6)
                self.assertLess(abs(run['energy_residual_j']),1e-6)
                self.assertEqual(run['temperature_k'],w.run(p)['temperature_k'])
                switched={e['id'] for e in run['events'] if e['kind']=='interface_destroyed'}
                self.assertEqual(switched,{'failure-C2-ISS','failure-C2-TIM'})
                for key in ('capacity_45','battery_iss_scale','surface_linear','initial_temperature_k'):
                    with self.assertRaises(RuntimeError):w.run({key:1.})
                with self.assertRaises(RuntimeError):w.run({'energy_j':1000001.})
            finally:w.close()
            path.write_text(json.dumps(bake(s,p)));w=Worker(Path(os.environ['BATTERY_SIM_FAILURE_PROBE']),path)
            try:self.assertEqual(run['temperature_k'],w.run({})['temperature_k'])
            finally:w.close()
            next(c for c in s['cells'] if c['role']=='passive')['thermal']['heat_capacity_law']={'factor':[[298.15,1],[343.15,1]]}
            path.write_text(json.dumps(s));w=Worker(Path(os.environ['BATTERY_SIM_FAILURE_PROBE']),path)
            try:
                with self.assertRaises(RuntimeError):w.run({})
            finally:w.close()

    @unittest.skipUnless(os.environ.get('BATTERY_SIM_FAILURE_PROBE') and os.environ.get('BATTERY_SIM_EXPERIMENT_PROBE'),'Both CUDA probes needed')
    def test_same_intact_observer_as_accepted_probe(self):
        s=json.loads((PRIOR/'no-cooling-fitted-scenario.json').read_text());s['simulation']['duration_s']=2
        with tempfile.TemporaryDirectory(prefix='failure-intact-') as tmp:
            path=Path(tmp)/'scene.json';path.write_text(json.dumps(s));runs=[]
            for env in ('BATTERY_SIM_FAILURE_PROBE','BATTERY_SIM_EXPERIMENT_PROBE'):
                w=Worker(Path(os.environ[env]),path)
                try:runs.append(w.run({}))
                finally:w.close()
            self.assertEqual(runs[0]['temperature_k'],runs[1]['temperature_k'])

if __name__=='__main__':unittest.main()
