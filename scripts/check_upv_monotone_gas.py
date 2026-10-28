#!/usr/bin/env python3
"""Controlled full replay of a falling gas law versus its minimal monotone repair."""
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
import upv_gas_contact_fit as fit
from check_upv_restart import native
from upv_gas_heat_budget import inspect


def plot(out,runs,data,labels=None,title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=labels or {'archived':'Прежний закон с провалом','monotone':'Связность без падения'}
    fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for ax,key in zip(axes.flat,dataio.CHANNELS):
        for name,run in runs.items():ax.plot(run['time_s'],np.asarray(run['temperature_k'][key])-273.15,label=labels[name])
        t=np.asarray(next(iter(runs.values()))['time_s'])
        ax.plot(t,dataio.observed(data,t)[key]-273.15,c='#222222',label='Эксперимент',lw=1.)
        ax.set(title=key,ylabel='°C',xlabel='Время, с');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(title or 'Открытые края 2 мм; исправляется только падение газовой связности')
    fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for name,run in runs.items():
        for solid,ax in [('chamber-wall',axes[0]),('C3 casing',axes[1])]:
            rows=[row for row in run['gas_film_trace'].values()if
                  (row['solid']=='chamber-wall'if solid=='chamber-wall'else row['solid'].startswith('C3:case:'))]
            heat=np.sum([row['heat_to_solid_j']for row in rows],axis=0)/1000.
            ax.plot(run['time_s'],heat,label=labels[name])
    for ax,title in zip(axes,('Газ → стенка камеры','Газ → все поверхности оболочки C3')):
        ax.set(title=title,xlabel='Время, с',ylabel='Переданная энергия, кДж');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'heat-distribution.png',dpi=140);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--timeout',type=float,default=1800.);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(exist_ok=False)
    source=dataio.ROOT/'data/calibration/upv-separated-chemistry-v6/pad-openings-20261020-r1'
    original=json.loads((source/'open/scenario.json').read_text());prior=json.loads((source/'status.json').read_text())
    core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    if core!=prior['native_sha256']:raise ValueError('Diagnostic requires the SAME native kernel as the source run')
    old=original['metadata']['gas_contact_trial'];p=dict(old,metric_450=0.)
    fit.validate(p)
    # Preserve even the last bit of archived normalization: this controlled
    # contrast changes ONE knot, not a reconstructed or refitted contact law.
    repaired=copy.deepcopy(original);repaired['metadata']['gas_contact_trial']=p
    for c in repaired['gas_network']['thermal_contacts']:
        table=c['gas_surface_connectivity']['temperature_log_response']
        if table!=[[298.15,0.],[350.,0.],[450.,old['metric_450']],[900.,old['metric_900']]]:
            raise ValueError('Unexpected source temperature law')
        table[2][1]=0.
    for key in ('cells','ports','interfaces','shells','boundaries','measurement_probes'):
        if repaired[key]!=original[key]:raise ValueError('Frozen physics changed: '+key)
    state=dict(status='running',pid=os.getpid(),native_sha256=core,source=str(source),cases={},
        changed='ONLY gas-source metric at 450 K: -4 -> 0; identical kernel, geometry, cold G, ceiling, wall, chemistry',
        old_parameters=old,new_parameters=p,calibrated=False,validation_run=False)
    cal.save(out/'status.json',state);runs={};data,_=dataio.load('no-cooling')
    def stop(_sig,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        for name,scene in [('archived',original),('monotone',repaired)]:
            if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()!=core:raise ValueError('Native binary changed')
            folder=out/name;folder.mkdir();scene=copy.deepcopy(scene);scene['gas_film_trace']=True
            cal.save(folder/'scenario.json',scene);state['active_case']=name;cal.save(out/'status.json',state)
            print('START',name,flush=True)
            run=native(folder/'scenario.json',folder/'replay',dict(stop_after_first_rupture=False),a.timeout)
            report=inspect(scene,run)
            if name=='archived':
                reference=json.load(gzip.open(source/'open/replay/run.json.gz','rt'))
                if run['time_s']!=reference['time_s']:raise ValueError('Tracing changed sampling times')
                difference=max(float(np.max(np.abs(np.asarray(v)-reference['temperature_k'][k])))for k,v in run['temperature_k'].items())
                report['tracing_temperature_difference_k']=difference
                if difference!=0.:raise ValueError('Read-only tracing changed the reference trajectory')
                del reference
            cal.save(folder/'report.json',report);state['cases'][name]=report;runs[name]=run
            cal.save(out/'status.json',state);plot(out,runs,data)
            print('CASE',json.dumps(dict(name=name,speed=report['speed'],final=report['snapshots'][-1])),flush=True)
        state['status']='complete'
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:
        state.pop('active_case',None);cal.save(out/'status.json',state);print('MONOTONE_STATUS',state['status'],flush=True)
    return 0 if state['status']=='complete'else 1


if __name__=='__main__':raise SystemExit(main())
