#!/usr/bin/env python3
"""Conditioned vent sensitivity; not a fresh calibration or TC-position fit."""
import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import signal

import numpy as np

from calibrate_fsri30_bayes import atomic_save
from check_fsri30_radiation_experiment import CORE, PROBE, read_run
from check_upv_restart import native
import calibrate_upv_r123 as settings
from fsri30_data import ROOT, load, save, sha
from fsri30_objective import model_fields, targets


def truncate(scene, end):
    s=deepcopy(scene);s['simulation']['duration_s']=end
    s['failure_sample_times_s']=[t for t in s['failure_sample_times_s'] if t<=end+1e-8]
    if s['failure_sample_times_s'][-1]!=end:s['failure_sample_times_s'].append(end)
    return s


def summary(run, data):
    t,_,v=model_fields(run);truth=targets(data)
    grid=truth['time'][(truth['time']>=3600.)&(truth['time']<=t[-1]+1e-7)]
    q=np.quantile(v,[0.,.5,.9,1.],axis=1).T
    if not len(grid):return dict(observed_end_s=float(t[-1]))
    pred=np.stack([np.interp(grid,t,q[:,j])for j in range(4)],axis=1)
    mask=(truth['time']>=grid[0])&(truth['time']<=grid[-1]);error=pred-truth['q'][mask]
    return dict(observed_start_s=float(t[0]),observed_end_s=float(t[-1]),
        cooling_interval_s=[float(grid[0]),float(grid[-1])],
        envelope_rmse_k=float(np.sqrt(np.mean(error**2))),
        quantile_rmse_k=np.sqrt(np.mean(error**2,axis=0)).tolist(),
        quantile_bias_k=np.mean(error,axis=0).tolist(),last_quantiles_c=(pred[-1]-273.15).tolist())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--vent-scale',type=float,default=4.)
    ap.add_argument('--end',type=float,default=4200.)
    args=ap.parse_args()
    if not .05<=args.vent_scale<=4.:raise ValueError('Outside existing native diagnostic prior')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'data/calibration/fsri30-balance-bracket-20261026-r2/trial-0026'
    base=json.loads((source/'scenario.json').read_text());settings.PROBE=PROBE
    if sha(PROBE)!=CORE:raise ValueError('Frozen native binary changed')
    checkpoint=source/'post-cascade-3000.bin';data,_=load()
    state=dict(status='running',pid=os.getpid(),native_sha256=CORE,
        source_scenario=str(source/'scenario.json'),source_scenario_sha256=sha(source/'scenario.json'),
        checkpoint_sha256=sha(checkpoint),vent_scale=args.vent_scale,
        conditioned_suffix=True,fresh_calibration=False,validation_passed=False,calibrated=False)
    save(out/'contract.json',dict(state,controller_sha256=sha(Path(__file__))))
    atomic_save(out/'status.json',state)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        def replay(scene,folder,changed=False):
            folder.mkdir();save(folder/'scenario.json',scene)
            request=dict(restart_contract=CORE,restart_from=str(checkpoint),stop_after_first_rupture=False)
            if changed:request['restart_transport_scenario']=str(source/'scenario.json')
            return native(folder/'scenario.json',folder/'replay',request,7200.,minimum_speed=.2,rolling_minimum=.2)
        control=replay(truncate(base,3010.),out/'identity-control')
        reference=read_run(source/'replay/run.json.gz')
        rt,_,rv=model_fields(reference);ct,_,cv=model_fields(control)
        expected=np.stack([np.interp(ct,rt,rv[:,j])for j in range(30)],axis=1)
        difference=float(np.max(np.abs(cv-expected)))
        save(out/'identity-check.json',dict(maximum_component_temperature_difference_k=difference,
            exact_observation_times=bool(np.all(np.isin(ct,rt))),accepted=difference<1e-8))
        if difference>=1e-8:raise ValueError('Unchanged suffix differs from archived reference')
        scene=truncate(base,args.end);changed=[]
        for edge in scene['gas_network']['channels']:
            if edge['id'].startswith('vent-'):
                edge['area_m2']*=args.vent_scale;changed.append(edge['id'])
        if len(changed)!=2:raise ValueError('Expected the two actual module exits')
        state.update(phase='vent-suffix',changed_edges=changed)
        atomic_save(out/'status.json',state)
        run=replay(scene,out/'vent-suffix',True)
        # Strict ledgers and dt floor remain acceptance conditions, not fit knobs.
        if run.get('max_oxidizer_residual_kg',0.)>1e-10:raise ValueError('Oxygen ledger failed')
        if run['solver_profile']['minimum_regular_step_s']<.01-1e-9:raise ValueError('Minimum step violated')
        reference['time_s']=rt[rt<=args.end+1e-7].tolist()
        reference['cell_mean_temperature_k']={k:v[:len(reference['time_s'])]for k,v in reference['cell_mean_temperature_k'].items()}
        result=dict(reference=summary(reference,data),conditioned_trial=summary(run,data),
            balances={k:v for k,v in run.items()if k.startswith('max_')and'residual'in k},
            fresh_calibration=False,validation_passed=False,calibrated=False)
        save(out/'summary.json',result);state.update(status='complete',result=result)
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:atomic_save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()
