#!/usr/bin/env python3
"""Frozen-parameter comparison: neutral mixing versus the shared gas S(T)."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import numpy as np
import calibrate_upv_r123 as cal
import upv_r123_data as dataio
from upv_gas_contact_fit import gas_mixing_profile
from upv_gas_heat_budget import inspect
from check_upv_monotone_gas import plot
from check_upv_restart import native


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--timeout',type=float,default=1800.);a=ap.parse_args()
    source=dataio.ROOT/'data/calibration/upv-separated-chemistry-v6/monotone-gas-audit-20261020-r1/monotone'
    original=json.loads((source/'scenario.json').read_text());coupled=copy.deepcopy(original)
    gas_mixing_profile(coupled,original['metadata']['gas_contact_trial'])
    expected=copy.deepcopy(original)
    for key in ('surface_connectivity','mixing_connectivity'):
        expected['gas_network'][key]=copy.deepcopy(coupled['gas_network'][key])
    if coupled!=expected:raise ValueError('Frozen material, chemistry, geometry or thermal film changed')
    core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest();out=a.output.resolve();out.mkdir(exist_ok=False)
    state=dict(status='running',pid=os.getpid(),native_sha256=core,source=str(source),cases={},
        calibrated=False,validation_run=False,changed='Counterexchange now evaluates shared gas S(T); existing profile and all physical parameters frozen')
    cal.save(out/'status.json',state);data,_=dataio.load('no-cooling');runs={}
    def stop(_sig,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        for label,scene in [('control',original),('temperature',coupled)]:
            if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()!=core:raise ValueError('Native binary changed during audit')
            folder=out/label;folder.mkdir();cal.save(folder/'scenario.json',scene)
            state['active_case']=label;cal.save(out/'status.json',state);print('START',label,flush=True)
            run=native(folder/'scenario.json',folder/'replay',dict(stop_after_first_rupture=False),a.timeout)
            report=inspect(scene,run)
            gap={n for n in run['gas']if n.startswith('gap-ISS1-')}
            exits=[e for e in scene['gas_network']['channels']if e['id'].startswith('escape-ISS1-')]
            inlet={e['to']for e in exits};times=run['time_s'];at=930.
            def temperature(names):
                masses={n:float(np.interp(at,times,run['gas'][n]['mass_kg']))for n in names}
                return sum(masses[n]*float(np.interp(at,times,run['gas'][n]['temperature_k']))for n in names)/sum(masses.values())-273.15
            report['C3_ISS_at930']=dict(surface_c=float(np.interp(at,times,run['temperature_k']['108']))-273.15,
                gap_gas_c=temperature(gap),inlet_gas_c=temperature(inlet),
                mixed_perimeter_mass_kg=sum(float(np.interp(at,times,run['channels'][e['id']]['mixed_mass_kg']))for e in exits))
            if label=='control':
                prior=json.load(gzip.open(source/'replay/run.json.gz','rt'))
                if prior['time_s']!=run['time_s']:raise ValueError('Control sampling changed')
                difference=max(float(np.max(np.abs(np.asarray(v)-prior['temperature_k'][k])))for k,v in run['temperature_k'].items())
                report['difference_from_pre_fix_neutral_mixing_k']=difference
                del prior
                if difference>1e-6:raise ValueError('Neutral S changed control beyond roundoff: '+str(difference))
            if report['minimum_regular_step_s']<.005-1e-12:raise ValueError('Regular step below 0.005 s')
            cal.save(folder/'report.json',report);runs[label]=run;state['cases'][label]=report
            cal.save(out/'status.json',state)
            plot(out,runs,data,labels={'control':'Контроль: нейтральный газовый обмен','temperature':'Обмен газ–газ через S(T)'},
                title='Температурный газообмен; геометрия, химия и параметры не подбираются')
            print('CASE',label,'speed',report['speed'],'C3_ISS_final_C',run['temperature_k']['108'][-1]-273.15,flush=True)
        state['status']='complete'
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:
        state.pop('active_case',None);cal.save(out/'status.json',state);print('THERMAL_MIXING_STATUS',state['status'],flush=True)
    return 0 if state['status']=='complete'else 1


if __name__=='__main__':raise SystemExit(main())
