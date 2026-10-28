#!/usr/bin/env python3
"""Paired full-trajectory check: closed pad footprints versus open edge bands."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import numpy as np
import calibrate_upv_r123 as cal
import upv_r123_model as model
import upv_r123_data as dataio
from upv_gas_contact_fit import candidate
from check_upv_restart import native
from check_upv_gas_routes import distribution


def plot(out,runs,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for ax,key in zip(axes.flat,dataio.CHANNELS):
        for label,r in runs.items():
            ax.plot(r['time_s'],np.asarray(r['temperature_k'][key])-273.15,label=label)
        raw=data['time_s'];lo=min(r['time_s'][0]for r in runs.values());hi=max(r['time_s'][-1]for r in runs.values())
        t=raw[(raw>=lo)&(raw<=hi)];ax.plot(t,dataio.observed(data,t)[key]-273.15,c='#222222',label='Эксперимент',lw=1.)
        ax.set(title=key,xlabel='Время, с',ylabel='°C');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Проверка геометрии, без новой калибровки: закрытый край / отступ прокладки 2 мм')
    fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gas-parameters',type=Path,required=True,help='Explicit admissible gas law; falling archived recovery laws are rejected')
    ap.add_argument('--timeout',type=float,default=1800.);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(exist_ok=False)
    root=dataio.ROOT/'data/calibration/upv-separated-chemistry-v6'
    seed_path=root/'restart-gas-baseline-20261019-r1/parameters.json';seed=json.loads(seed_path.read_text())
    fit_path=a.gas_parameters.resolve();gas=json.loads(fit_path.read_text())['parameters']
    data,_=dataio.load('no-cooling');event=dataio.event(data);end=event['time_s']+300.
    core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    state=dict(status='running',pid=os.getpid(),native_sha256=core,cases={},gas_contact_parameters=gas,
        source_parameters=str(seed_path),source_gas_trial=str(fit_path),calibrated=False,
        reason='Geometry diagnostic with the SAME last recovery gas parameters in both cases',
        full_from_ambient=True,horizon_s=end)
    cal.save(out/'status.json',state)
    def stop(_sig,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);runs={}
    try:
        for label,clearance in [('Контроль: край закрыт',0.),('Открытые края: 2 мм',.002)]:
            folder=out/('closed'if clearance==0. else'open');folder.mkdir()
            scene,_=model.build('no-cooling',seed['parameters'],duration=end,gas_extensions=seed['gas_extensions'],
                gas_geometry={'pad_edge_clearance_m':clearance})
            controls=seed['numerical_controls'];scene['simulation']['minimum_dt_s']=controls['minimum_dt_s']
            scene['gas_network'].update({k:v for k,v in controls.items()if k!='minimum_dt_s'})
            scene=candidate(scene,gas);scene['metadata']['zone_diagnostics']=False
            scene['metadata']['numerical_controls']=controls;scene['thermal_path_trace']=True
            cal.save(folder/'scenario.json',scene);state['active_case']=label;cal.save(out/'status.json',state)
            print('START',label,flush=True)
            run=native(folder/'scenario.json',folder/'replay',dict(stop_after_first_rupture=False),a.timeout)
            report=distribution(scene,run)
            if any(abs(v.get('oxidizer_balance_residual_kg',0.))>1e-10 for v in report['groups'].values()):
                raise RuntimeError('Local oxygen budget does not close')
            report.update(minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'],
                gas_volumes=len(scene['gas_network']['compartments']),gas_channels=len(scene['gas_network']['channels']))
            cal.save(folder/'report.json',report);state['cases'][label]=report;runs[label]=run
            cal.save(out/'status.json',state);plot(out,runs,data)
            print('CASE',json.dumps(dict(label=label,speed=report['speed'],C3_ISS_final_c=report['final_temperature_c']['108'],
                oxygen_ISS1=report['groups']['ISS1'],wall_s=run['wall_s'])),flush=True)
        state['status']='complete'
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:
        state.pop('active_case',None);cal.save(out/'status.json',state);print('OPENINGS_STATUS',state['status'],flush=True)
    return 0 if state['status']=='complete'else 1


if __name__=='__main__':raise SystemExit(main())
