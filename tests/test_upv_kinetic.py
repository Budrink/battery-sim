"""Kinetic failure calibration: restricted parameters, GPU audit and export."""
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
    import calibrate_upv_kinetic as kin
    from calibrate_upv import Worker
finally:sys.path.pop(0)
from test_upv_failure import fixture

class KineticCalibrationTests(unittest.TestCase):
    def test_bake_only_six_failure_parameters(self):
        _,s=fixture();s=kin.activate(s);p=dict(zip(kin.NAMES,(20000.,.3,3.,30.,.02,.1)))
        changed=kin.bake(s,p)
        for a,b in zip(s['cells'],changed['cells']):
            if a['id']=='C2':
                b['runaway']['energy_j']=a['runaway']['energy_j']
                b['runaway']['kinetics']['rate_at_reference_per_s']=a['runaway']['kinetics']['rate_at_reference_per_s']
                b['thermal']['reaction_structure']['residue_factor']=a['thermal']['reaction_structure']['residue_factor']
        for a,b in zip(s['interfaces'],changed['interfaces']):
            if a['id'].startswith('failure-C2-'):
                b['law']['destroyed_background_conductance_w_per_k']=a['law']['destroyed_background_conductance_w_per_k']
        self.assertEqual(s,changed)
        self.assertEqual(set(kin.PROTOCOL['fitted']),set(kin.NAMES))
        c=next(c for c in s['cells'] if c['id']=='C2')
        self.assertNotIn('energy_outlet',c['runaway'])
        self.assertNotIn('duration_s',c['runaway'])
        self.assertEqual(c['thermal']['divisions'],[3,3,3])
        for key in ('capacity_45','surface_linear','duration_s','activation_energy_j_per_mol'):
            with self.assertRaises(ValueError):kin.check(dict(p,**{key:1.}))

    def test_unchanged_late_only_objective(self):
        times=np.array([0.,100.,300.,600.,1800.])+.5
        observed={k:np.full(5,300.) for k in kin.prior.WITNESSES}
        run=dict(time_s=times.tolist(),temperature_k={k:[900.,900.,301.,301.,301.] for k in observed})
        self.assertEqual(kin.metrics(run,observed,{'bracket_width_s':.5})['rmse_k'],1.)
        run['temperature_k']['C2-TIM']=[float('nan')]*5
        self.assertEqual(kin.metrics(run,observed,{'bracket_width_s':.5})['rmse_k'],1.)

    @unittest.skipUnless(os.environ.get('BATTERY_SIM_FAILURE_PROBE'),'CUDA failure probe not requested')
    def test_gpu_parameter_guard_audit_reset_and_export(self):
        _,s=fixture();s=kin.activate(s);p=dict(zip(kin.NAMES,(20000.,.3,3.,30.,.02,.1)))
        with tempfile.TemporaryDirectory(prefix='kinetic-fit-test-') as tmp:
            path=Path(tmp)/'s.json';path.write_text(json.dumps(s));w=Worker(Path(os.environ['BATTERY_SIM_FAILURE_PROBE']),path)
            try:
                run=w.run(p);kin.audit(run,s,p)
                self.assertEqual(run['temperature_k'],w.run(p)['temperature_k'])
                self.assertEqual(run['source_damage'][0],0.)
                self.assertEqual(run['released_energy_j_history'][0],0.)
                for key in ('duration_s','initial_temperature_k','heat_capacity_j_per_k','internal_log_35','activation_damage','vent_heat_fraction','bulk_peak_factor'):
                    with self.assertRaises(RuntimeError):w.run({key:1.})
                for key,val in [('rate_400_per_s',0.),('residue_inplane',11.),('destroyed_iss_h',.001),('max_fraction_per_step',.1)]:
                    with self.assertRaises(RuntimeError):w.run({key:val})
                fine=w.run(dict(p,max_fraction_per_step=.0025,max_temperature_rise_per_step_k=.25,max_dt_s=.1))
                kin.audit(fine,s,p)
            finally:w.close()
            path.write_text(json.dumps(kin.bake(s,p)));w=Worker(Path(os.environ['BATTERY_SIM_FAILURE_PROBE']),path)
            try:
                export=w.run({});kin.audit(export,kin.bake(s,p),p)
                self.assertEqual(run['temperature_k'],export['temperature_k'])
                self.assertEqual(run['released_energy_j_history'],export['released_energy_j_history'])
            finally:w.close()

    @unittest.skipUnless(os.environ.get('BATTERY_SIM_FAILURE_PROBE') and os.environ.get('BATTERY_SIM_EXPERIMENT_PROBE'),'Both GPU observers needed')
    def test_surface_observer_parity(self):
        _,s=fixture();p=dict(zip(kin.NAMES,(20000.,.3,3.,30.,.02,.1)));s=kin.bake(kin.activate(s),p)
        s.pop('failure_sample_times_s')
        with tempfile.TemporaryDirectory(prefix='kinetic-observer-') as tmp:
            path=Path(tmp)/'s.json';path.write_text(json.dumps(s));runs=[]
            for env in ('BATTERY_SIM_FAILURE_PROBE','BATTERY_SIM_EXPERIMENT_PROBE'):
                w=Worker(Path(os.environ[env]),path)
                try:runs.append(w.run({}))
                finally:w.close()
            self.assertEqual(runs[0]['temperature_k'],runs[1]['temperature_k'])

if __name__=='__main__':unittest.main()
