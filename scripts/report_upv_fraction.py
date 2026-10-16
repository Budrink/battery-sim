#!/usr/bin/env python3
"""Plot completed fixed-fraction diagnostics; never declare a fit accepted."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import calibrate_upv_fraction as protocol
from calibrate_upv import save


def load(path):
    return json.loads(path.read_text())


def witness_gas_budgets(scene,run):
    """Resolve through the actual sensor contact, never through cell numbering.

    In this frozen geometry C3/ISS is ISS1, while ISS3 belongs to C1/ISS.
    Gas budgets are not the complete solid heat balance (conduction is separate).
    """
    ports={p['id']:p for p in scene['ports']};gas=scene['gas_network'];result={}
    for probe in scene['measurement_probes']:
        if 'neighbor_port_id' not in probe:continue
        pad=ports[probe['neighbor_port_id']]['cell_id']
        ids={n['id'] for n in gas['compartments'] if n.get('local_equilibrium',{}).get('cell_id')==pad}
        net=sum((int(c['to'] in ids)-int(c['from'] in ids))*run['channels'][c['id']]['delivered_gas_energy_j'][-1]
                for c in gas['channels'])
        result[probe['id']]=dict(material_cell_id=pad,gas_compartments=sorted(ids),net_advected_energy_j=net,
                                gas_reaction_j=sum(run['gas'][k]['released_energy_j'][-1] for k in ids))
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('folder',type=Path);args=p.parse_args();folder=args.folder
    rows=[]
    for name,_ in protocol.CASES:
        for suffix in ('-check','-check-fine','','-duration3'):
            path=folder/(name+suffix+'.json')
            if path.exists():rows.append((name+suffix,load(path)))
    if not rows:raise ValueError('No completed reports')
    fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for name,r in rows:
        run=r['result'];t=run['time_s'];channels=[v for k,v in run['channels'].items() if k.startswith('C2-progressive')]
        area=np.sum([c['effective_area_m2'] for c in channels],axis=0)*1e6
        work=np.sum([c['fracture_work_j'] for c in channels],axis=0)
        mass=np.sum([c['transferred_mass_kg'] for c in channels],axis=0)*1000
        pressure=np.max([v['pressure_pa'] for k,v in run['gas'].items() if k.startswith('inside-C2')],axis=0)/1e5
        style='--' if name.endswith('fine') else '-'
        for a,y in zip(ax.flat,(pressure,area,work,mass)):a.plot(t,y,style,label=name)
    for a,title in zip(ax.flat,('Max sampled internal pressure, bar','Total open area, mm²','Fracture work, J','Net gas export, g')):
        a.set_title(title);a.set_xlabel('Time after initial state, s');a.grid(alpha=.2)
    ax[0,0].legend(fontsize=8);fig.savefig(folder/'fraction-numerics.png',dpi=150);plt.close(fig)
    trajectories=[(n,r) for n,r in rows if r['duration']>=3]
    if trajectories:
        fig,ax=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
        _,prior=protocol.area.old.geometry.inputs('no-cooling')
        t=np.asarray(prior['result']['time_s']);mask=t<=max(r['duration'] for _,r in trajectories)
        for a,ch in zip(ax.flat,('101','102','107','108')):
            a.plot(t[mask],np.asarray(prior['observed_temperature_k'][ch])[mask]-273.15,'k.',label='Experiment')
            for name,r in trajectories:a.plot(r['result']['time_s'],np.asarray(r['result']['temperature_k'][ch])-273.15,label=name)
            a.set_title({'101':'C1 / ISS','102':'C1 / TIM','107':'C3 / TIM','108':'C3 / ISS'}[ch]+' — °C')
        for a in (ax[1,1],ax[1,2]):
            a.plot(t[mask],np.asarray(prior['observed_pressure_bar'])[mask],'k.',label='Experiment')
            for name,r in trajectories:a.plot(r['result']['time_s'],np.asarray(r['result']['gas']['chamber']['pressure_pa'])/1e5,label=name)
            a.set_title('Chamber pressure, bar')
        ax[1,2].set_xlim(0,3)
        for a in ax.flat:a.grid(alpha=.2);a.set_xlabel('Time, s')
        ax[0,0].legend(fontsize=8);fig.savefig(folder/'fraction-experiment.png',dpi=150);plt.close(fig)
    summary=[]
    for name,r in rows:
        run=r['result'];s=load(folder/(name+'-scenario.json'))
        summary.append(dict(id=name,parameters=r['parameters'],duration=r['duration'],metrics=r['metrics'],features=r['features'],
            steps=run['steps'],wall_s=run['wall_s'],witness_gas_budgets=witness_gas_budgets(s,run),
            chamber_gas_reaction_j=run['gas']['chamber']['released_energy_j'][-1],
            total_gas_reaction_j=sum(v['released_energy_j'][-1] for v in run['gas'].values())))
    save(folder/'assessment.json',dict(cases=summary,calibration_accepted=False))


if __name__=='__main__':main()
