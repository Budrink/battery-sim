#!/usr/bin/env python3
"""Remove all pads, restore one at a time; frozen pre-vent state and chemistry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import numpy as np
import upv_r123_model as model
from upv_pad_ablation import PADS
from upv_gas_remap import unvented_remap
from check_upv_restart import native
from check_upv_gas_routes import distribution,load
import calibrate_upv_r123 as cal


def scene_for(seed,duration,removed):
    scene,_=model.build('no-cooling',seed['parameters'],duration=duration,
        gas_extensions=seed['gas_extensions'],diagnostic_removed_pads=removed)
    controls=seed['numerical_controls']
    scene['simulation']['minimum_dt_s']=controls['minimum_dt_s']
    scene['gas_network'].update({k:v for k,v in controls.items()if k!='minimum_dt_s'})
    scene['metadata']['numerical_controls']=controls
    scene['thermal_path_trace']=True
    return scene


def metrics(scene,run,removed):
    result=distribution(scene,run);errors={}
    objects={c['id']:c for c in scene['cells']}
    for name in removed:
        temperature=np.array(run['thermal_paths']['objects'][name]['reference_capacity_weighted_temperature_k'])
        errors[name]=float(np.max(abs(temperature-temperature[0]))*objects[name]['thermal']['heat_capacity_j_per_k'])
    if max(errors.values(),default=0.)>1e-4:raise RuntimeError('Removed pad is still exchanging energy')
    port_owner={p['id']:p['cell_id']for p in scene['ports']}
    for e in scene['interfaces']:
        if any(port_owner[e[k]]in removed for k in ('port_a','port_b')):
            flux=run['thermal_paths']['interfaces'][e['id']]['a_to_b_power_w']
            if max(abs(x)for x in flux)>1e-12:raise RuntimeError('Heat crosses removed pad')
    result.update(removed=list(removed),retained=[n for n in PADS if n not in removed],
        escrow_energy_drift_j=errors,c3_iss_peak_c=max(run['temperature_k']['108'])-273.15,
        c2_iss_peak_c=max(run['temperature_k']['C2_ISS'])-273.15,
        max_c3_iss_gap_c=max(sum(run['gas'][f'gap-ISS1-{j}']['temperature_k'][i]*run['gas'][f'gap-ISS1-{j}']['mass_kg'][i]for j in range(8))/
            sum(run['gas'][f'gap-ISS1-{j}']['mass_kg'][i]for j in range(8))-273.15 for i in range(len(run['time_s']))))
    return result


def plot(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for row in rows:
        if row['status']!='complete':continue
        run=load(out/row['case']/'replay/run.json.gz');t=run['time_s']
        label='Без прокладок'if not row['retained']else'Вернули '+','.join(row['retained'])
        for ax,key in zip(axes.flat,('108','C2_ISS','101','107')):
            ax.plot(t,np.asarray(run['temperature_k'][key])-273.15,label=label,lw=1.5)
    for ax,title in zip(axes.flat,('C3 / ISS','C2 / ISS','C1 / ISS','C3 / TIM')):
        ax.set(title=title,xlabel='Время, с',ylabel='Виртуальная температура поверхности, °C');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Диагностическое удаление слоёв после прогрева; химия не меняется')
    fig.savefig(out/'pad-ablation-temperatures.png',dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--timeout',type=float,default=400.);ap.add_argument('--max-cases',type=int,default=8)
    ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    out=a.output.resolve();out.mkdir(exist_ok=a.resume);base=a.baseline.resolve()
    seed=json.loads((base/'parameters.json').read_text());source=json.loads((base/'scenario.json').read_text())
    fingerprint=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest();rows=[]
    if a.resume:
        prior=json.loads((out/'status.json').read_text())
        if prior['native_sha256']!=fingerprint:raise ValueError('Native changed during ablation series')
        rows=prior['cases']
    state=dict(status='running',pid=os.getpid(),native_sha256=fingerprint,baseline=str(base),
        chemistry_frozen=True,calibrated=False,order=list(PADS),cases=rows)
    def stop(_sig,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        for k in range(min(a.max_cases,8)):
            name=f'{k:02d}-retained-{k}'
            if any(r['case']==name and r['status']=='complete'for r in rows):continue
            if (out/name).exists():raise RuntimeError('Incomplete case already exists: '+name)
            folder=out/name;folder.mkdir();removed=list(PADS[k:]);scene=scene_for(seed,950.,removed)
            cal.save(folder/'scenario.json',scene);mapping=unvented_remap(source,scene);cal.save(folder/'remap.json',mapping)
            state.update(active_case=name);cal.save(out/'status.json',state)
            print('START',name,'removed',removed,flush=True)
            try:
                run=native(folder/'scenario.json',folder/'replay',dict(restart_from=str(base/'sealed-state.bin'),
                    restart_source_scenario=str(base/'scenario.json'),unvented_gas_remap=mapping,
                    restart_contract=json.loads((base/'status.json').read_text())['native_sha256'],diagnostic_pad_ablation=True),a.timeout)
                summary=metrics(scene,run,removed);cal.save(folder/'summary.json',summary)
                row=dict(case=name,status='complete',retained=list(PADS[:k]),**{q:summary[q]for q in
                    ('c3_iss_peak_c','c2_iss_peak_c','max_c3_iss_gap_c','speed')},
                    c3_iss_transport_j=summary['groups']['ISS1']['net_inward_transport_energy_j'],
                    c3_iss_r3_j=summary['groups']['ISS1']['r3_during_suffix_j'])
                rows.append(row);print('DONE',json.dumps(row),flush=True)
                plot(out,rows)
            except Exception as exc:
                rows.append(dict(case=name,status='failed',retained=list(PADS[:k]),error=str(exc)));raise
            finally:cal.save(out/'status.json',state)
        state.update(status='complete'if len(rows)==8 else'case_budget');plot(out,rows)
    except KeyboardInterrupt:state.update(status='stopped')
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:cal.save(out/'status.json',state);print('SERIES',state['status'],flush=True)


if __name__=='__main__':main()
