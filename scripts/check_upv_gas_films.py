#!/usr/bin/env python3
"""Check shared directed gas connectivity with frozen parameters against scalar archives."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import numpy as np
import calibrate_upv_r123 as cal
from check_upv_restart import native
from check_upv_gas_routes import load,distribution
from upv_gas_remap import unvented_remap
from upv_connectivity import explicit_gas_contacts


def inspect(scene,run,reference):
    groups={}
    for key,c in run['gas_film_trace'].items():
        if not c['gas'].startswith('gap-ISS1-'):continue
        row=groups.setdefault(c['solid'],dict(conductance_w_k=0.,heat_to_solid_j=0.,positive_step_heat_j=0.,negative_step_heat_j=0.))
        row['conductance_w_k']+=c['reference_conductance_w_per_k']
        for q in ('heat_to_solid_j','positive_step_heat_j','negative_step_heat_j'):row[q]+=c[q][-1]
    balance=distribution(scene,run)['groups']['ISS1']
    direct=sum(c['heat_to_solid_j']for c in groups.values())
    closure=direct-balance['inferred_heat_to_surfaces_j']
    if abs(closure)>1e-6:raise RuntimeError('Actual gas-film scatter does not match gas energy balance')
    difference=max(float(np.max(abs(np.asarray(v)-np.asarray(reference['temperature_k'][k]))))for k,v in run['temperature_k'].items())
    return dict(contacts=groups,gas_energy_balance=balance,closure_j=closure,
        difference_from_archived_scalar_physics_k=difference,wall_s=run['wall_s'],speed=200./run['wall_s'],
        minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'])


def plot(out,runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for label,r in runs.items():
        t=r['time_s'];names=[n for n in r['gas']if n.startswith('gap-ISS1-')]
        temperature=np.sum([np.asarray(r['gas'][n]['temperature_k'])*np.asarray(r['gas'][n]['mass_kg'])for n in names],axis=0)/np.sum([r['gas'][n]['mass_kg']for n in names],axis=0)-273.15
        axes[0,0].plot(t,temperature,label=label)
        axes[0,1].plot(t,np.asarray(r['temperature_k']['108'])-273.15,label=label)
        for solid,style in [('C3:case:-x','-'),('ISS1','--')]:
            rows=[c for c in r['gas_film_trace'].values()if c['gas']in names and c['solid']==solid]
            if not rows:continue
            q=np.sum([c['heat_to_solid_j']for c in rows],axis=0)
            power=np.sum([c['instantaneous_power_w']for c in rows],axis=0)
            axes[1,0].plot(t,q,ls=style,label=label+' → '+solid)
            axes[1,1].plot(t,power,ls=style,label=label+' → '+solid)
    for ax,title,y in zip(axes.flat,('Газ перед C3/ISS','Поверхность оболочки C3/ISS','Фактически переданная энергия','Мгновенная мощность контакта'),('°C','°C','Дж','Вт')):
        ax.set(title=title,xlabel='Время, с',ylabel=y);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Газ → поверхность: общий направленный закон связности, параметры фиксированы')
    fig.savefig(out/'gas-film-audit.png',dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--timeout',type=float,default=400.);a=ap.parse_args()
    root=cal.ROOT if hasattr(cal,'ROOT')else Path(__file__).resolve().parents[1]
    data=root/'data/calibration/upv-separated-chemistry-v6'
    base=data/'restart-gas-baseline-20261019-r1';original=json.loads((base/'scenario.json').read_text())
    cases={'Без прокладок':data/'pad-ablation-20261020-r1/00-retained-0','Все прокладки':data/'gas-routes-20261019-r2'}
    out=a.output.resolve();out.mkdir(exist_ok=False)
    state=dict(status='running',pid=os.getpid(),native_sha256=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest(),cases={},
        physics_unchanged=False,change='All gas/surface heat uses shared directed connectivity; parameters frozen')
    cal.save(out/'status.json',state);runs={}
    def stop(_sig,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        test=cal.PROBE.parent/'battery_sim_gas_film_audit'
        analytical=subprocess.run([str(test),str(cases['Все прокладки']/'scenario.json')],capture_output=True,text=True,timeout=60.)
        if analytical.returncode:raise RuntimeError(analytical.stderr)
        cal.save(out/'analytical.json',json.loads(analytical.stdout))
        for i,(label,path)in enumerate(cases.items()):
            folder=out/f'case-{i}';folder.mkdir();scene=json.loads((path/'scenario.json').read_text());scene['gas_film_trace']=True
            explicit_gas_contacts(scene)
            cal.save(folder/'scenario.json',scene)
            request=dict(restart_from=str(base/'sealed-state.bin'),restart_source_scenario=str(base/'scenario.json'),
                unvented_gas_remap=unvented_remap(original,scene),restart_contract=json.loads((base/'status.json').read_text())['native_sha256'])
            if scene['metadata'].get('pad_ablation')is not None:request['diagnostic_pad_ablation']=True
            state['active_case']=label;cal.save(out/'status.json',state);print('START',label,flush=True)
            run=native(folder/'scenario.json',folder/'replay',request,a.timeout)
            report=inspect(scene,run,load(path/'replay/run.json.gz'));cal.save(folder/'audit.json',report)
            state['cases'][label]=report;runs[label]=run
            cal.save(out/'status.json',state);plot(out,runs);print('AUDIT',json.dumps(report),flush=True)
        state['status']='complete'
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:cal.save(out/'status.json',state);print('AUDIT_STATUS',state['status'],flush=True)
    return 0 if state['status']=='complete'else 1


if __name__=='__main__':raise SystemExit(main())
