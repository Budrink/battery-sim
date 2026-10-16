"""Transient structure, finite reserve, late-only observations and frozen Cp/k."""
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
    import calibrate_upv_failure_transition as tr
    from calibrate_upv import Worker
finally:sys.path.pop(0)
from test_upv_failure import fixture
from battery_sim.io import scenario_from_dict
from battery_sim.model import ModelError

def parameters():
    return tr.unpack(np.log([70000.,30.,500.,.1]))

class TransientFailureTests(unittest.TestCase):
    def test_only_late_witnesses_are_scored(self):
        t=np.array([0.,1.,300.,301.,1000.,1800.,1801.])+0.5
        observed={k:np.full(t.size,300.) for k in tr.WITNESSES}
        run={'time_s':t.tolist(),'temperature_k':{k:(observed[k]+1).tolist() for k in tr.WITNESSES}}
        event={'bracket_width_s':.5}
        self.assertEqual(tr.metrics(run,observed,event)['rmse_k'],1.)
        changed=copy.deepcopy(run);obs=copy.deepcopy(observed)
        obs['C2-ISS']=np.full(t.size,2000.);changed['temperature_k']['C2-ISS']=[100.]*t.size
        for k in tr.WITNESSES:
            changed['temperature_k'][k][0:2]=[10000.,10000.]
            changed['temperature_k'][k][-1]=10000.
        self.assertEqual(tr.metrics(changed,obs,event)['rmse_k'],1.)
        changed['temperature_k']['101'][3]=302.
        self.assertGreater(tr.metrics(changed,obs,event)['rmse_k'],1.)
        obs['101'][:]=np.nan
        with self.assertRaises(ValueError):tr.metrics(changed,obs,event)

    def test_export_changes_only_authorized_failure_block(self):
        _,s=fixture();s=tr.activate(s);p=parameters();tr.check(p);b=tr.bake(s,p)
        for a,c in zip(s['cells'],b['cells']):
            if a['id']=='C2':
                c['runaway']=a['runaway']
                c['thermal']['bulk_connectivity']['destroyed_background_factor']=a['thermal']['bulk_connectivity']['destroyed_background_factor']
                c['thermal']['bulk_failure_transient']=a['thermal']['bulk_failure_transient']
        for a,c in zip(s['interfaces'],b['interfaces']):
            if a['id'].startswith('failure-C2-'):
                c['law']['destroyed_background_conductance_w_per_k']=a['law']['destroyed_background_conductance_w_per_k']
        self.assertEqual(s,b)
        self.assertEqual(set(tr.PROTOCOL['fitted']),set(tr.FIT_NAMES))
        self.assertFalse(set(tr.STRUCTURE_PRIOR)&set(tr.FIT_NAMES))
        for k,v in [('capacity_45',1.),('bulk_peak_factor',0.),('destroyed_bulk_factor',float('nan'))]:
            with self.assertRaises(ValueError):tr.check(dict(p,**{k:v}))

    def test_python_reference_fails_closed(self):
        raw=json.loads((tr.v1.ROOT/'examples/three-cell-contained.json').read_text())
        for where,extra in [('thermal',{'bulk_failure_transient':{'peak_factor':3.,'relaxation_time_s':5.}}),
                            ('thermal',{'bulk_destruction_on_boundary':True}),('runaway',{'profile':'burst_tail'})]:
            s=copy.deepcopy(raw);s['cells'][0][where].update(extra)
            with self.assertRaisesRegex(ModelError,'native CUDA'):scenario_from_dict(s)

    @unittest.skipUnless(os.environ.get('BATTERY_SIM_FAILURE_PROBE'),'CUDA failure probe not requested')
    def test_cuda_transient_export_clock_energy_and_guards(self):
        _,scene=fixture();s=tr.activate(scene);p=parameters()
        p.update(duration_s=.1,tail_time_s=5.,bulk_peak_factor=30.,bulk_relaxation_time_s=.1)
        with tempfile.TemporaryDirectory(prefix='failure-transient-') as tmp:
            path=Path(tmp)/'scene.json';path.write_text(json.dumps(s));w=Worker(Path(os.environ['BATTERY_SIM_FAILURE_PROBE']),path)
            try:
                r=w.run(p);tr.audit(r,s,p)
                self.assertEqual(r['temperature_k'],w.run(p)['temperature_k'])
                f=np.asarray(r['bulk_factor_min']);self.assertTrue((f>10.).any());self.assertAlmostEqual(f[-1],.1)
                for key in ('capacity_45','internal_log_35','surface_linear','initial_temperature_k'):
                    with self.assertRaises(RuntimeError):w.run({key:1.})
                for key,value in [('bulk_peak_factor',101),('bulk_relaxation_time_s',0.),('fast_fraction',1.1)]:
                    with self.assertRaises(RuntimeError):w.run({key:value})
            finally:w.close()
            path.write_text(json.dumps(tr.bake(s,p)));w=Worker(Path(os.environ['BATTERY_SIM_FAILURE_PROBE']),path)
            try:self.assertEqual(r['temperature_k'],w.run({})['temperature_k'])
            finally:w.close()

    @unittest.skipUnless(os.environ.get('BATTERY_SIM_FAILURE_PROBE') and os.environ.get('BATTERY_SIM_EXPERIMENT_PROBE'),'Both CUDA probes needed')
    def test_both_surface_observers_agree_after_rupture(self):
        _,s=fixture();s=tr.bake(tr.activate(s),parameters());s.pop('failure_sample_times_s')
        with tempfile.TemporaryDirectory(prefix='failure-observer-') as tmp:
            path=Path(tmp)/'scene.json';path.write_text(json.dumps(s));runs=[]
            for env in ('BATTERY_SIM_FAILURE_PROBE','BATTERY_SIM_EXPERIMENT_PROBE'):
                w=Worker(Path(os.environ[env]),path)
                try:runs.append(w.run({}))
                finally:w.close()
            self.assertEqual(runs[0]['temperature_k'],runs[1]['temperature_k'])

if __name__=='__main__':unittest.main()
