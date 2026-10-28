#!/usr/bin/env python3
"""Frozen-chemistry spatial gas comparison from a verified sealed checkpoint."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import numpy as np
import calibrate_upv_r123 as cal
import fit_upv_kinetics as fit
from check_upv_restart import native
from upv_gas_remap import unvented_remap
from upv_connectivity import reference_gas_conductance
import upv_r123_data as dataio


def load(path):
    with gzip.open(path,'rt')as f:return json.load(f)


def distribution(scene,run):
    groups={}
    def group_of(name):
        return name.split('-')[1] if name.startswith(('inside-','gap-')) else 'external'
    for name,g in run['gas'].items():
        group=group_of(name)
        row=groups.setdefault(group,{})
        for key in ('mass_kg','chemical_j','released_energy_j','oxidizer_kg','vapor_kg'):
            row[key]=row.get(key,0.)+g[key][-1]
        if not name.startswith('inside-'):
            cv=scene['gas_network']['cv_j_per_kg_k'];tref=scene['gas_network']['reference_temperature_k']
            delta=cv*(g['mass_kg'][-1]*(g['temperature_k'][-1]-tref)-g['mass_kg'][0]*(g['temperature_k'][0]-tref))
            delta+=g['latent_j'][-1]-g['latent_j'][0]
            row['gas_energy_change_j']=row.get('gas_energy_change_j',0.)+delta
            row['r3_during_suffix_j']=row.get('r3_during_suffix_j',0.)+g['released_energy_j'][-1]-g['released_energy_j'][0]
            row.setdefault('net_inward_transport_energy_j',0.)
            row['initial_oxidizer_kg']=row.get('initial_oxidizer_kg',0.)+g['oxidizer_kg'][0]
            row['r3_consumed_oxidizer_kg']=row.get('r3_consumed_oxidizer_kg',0.)+g['consumed_oxidizer_kg'][-1]-g['consumed_oxidizer_kg'][0]
            row.setdefault('net_inward_oxidizer_kg',0.)
    arrivals={}
    for edge in scene['gas_network']['channels']:
        values=run['channels'][edge['id']]
        # Boundaries of each unbound gas group: exact cumulative transport
        # ledgers, not a sparse-time integration of instantaneous flow rates.
        for endpoint,sign in [('from',-1.),('to',1.)]:
            name=edge[endpoint]
            if name not in run['gas'] or name.startswith('inside-'):continue
            row=groups[group_of(name)]
            q=values['delivered_gas_energy_j'][-1]-values['delivered_gas_energy_j'][0]
            q+=values['transferred_latent_j'][-1]-values['transferred_latent_j'][0]
            row['net_inward_transport_energy_j']+=sign*q
            row['net_inward_oxidizer_kg']+=sign*(values['transferred_oxidizer_kg'][-1]-values['transferred_oxidizer_kg'][0])
        if edge['id'].startswith('escape-'):
            pad=edge['from'].split('-')[1]
            row=arrivals.setdefault(pad,dict(net_inward_mass_kg=0.,net_inward_chemical_j=0.,exchanged_each_way_kg=0.))
            for key,value in [('net_inward_mass_kg','transferred_mass_kg'),('net_inward_chemical_j','transferred_chemical_j')]:
                row[key]-=values[value][-1]-values[value][0]
            row['exchanged_each_way_kg']+=values['mixed_mass_kg'][-1]-values['mixed_mass_kg'][0]
    for row in groups.values():
        if 'gas_energy_change_j'in row:
            row['inferred_heat_to_surfaces_j']=row['net_inward_transport_energy_j']+row['r3_during_suffix_j']-row['gas_energy_change_j']
            row['oxidizer_balance_residual_kg']=row['initial_oxidizer_kg']+row['net_inward_oxidizer_kg']-row['r3_consumed_oxidizer_kg']-row['oxidizer_kg']
    ports={p['id']:p for p in scene['ports']};objects={c['id']:c for c in scene['cells']}
    slits={};times=np.asarray(run['time_s'])
    for pad in arrivals:
        names=[n for n in run['gas']if n.startswith('gap-'+pad+'-')]
        edges=[e for e in scene['gas_network']['channels']if e['id'].startswith('escape-'+pad+'-')]
        neighbors={e['to']for e in edges};ha=0.
        for c in scene['gas_network']['thermal_contacts']:
            if c['compartment']not in names:continue
            port=ports[c['port_id']];area=port['area_m2']
            if 'thermal_volume'in c:
                dims=objects[port['cell_id']]['thermal']['divisions'];axis='xyz'.index(port['surface'][1])
                area/=dims[(axis+1)%3]*dims[(axis+2)%3]
            ha+=reference_gas_conductance(c,port['area_m2'])*area/port['area_m2']
        rows=[]
        for at in (860.,885.,900.,950.):
            i=int(np.argmin(abs(times-at)))
            def temperature(ids):
                return sum(run['gas'][n]['temperature_k'][i]*run['gas'][n]['mass_kg'][i]for n in ids)/sum(run['gas'][n]['mass_kg'][i]for n in ids)-273.15
            rows.append(dict(time_s=float(times[i]),adjacent_external_mass_weighted_temperature_c=temperature(neighbors),
                slit_mass_weighted_temperature_c=temperature(names),
                slit_oxidizer_fraction=sum(run['gas'][n]['oxidizer_kg'][i]for n in names)/sum(run['gas'][n]['mass_kg'][i]for n in names)))
        initial_capacity=sum(run['gas'][n]['mass_kg'][0]for n in names)*scene['gas_network']['cv_j_per_kg_k']
        slits[pad]=dict(perimeter_area_m2=sum(e['area_m2']for e in edges),total_film_conductance_w_k=ha,
            initial_gas_film_relaxation_s=initial_capacity/ha,snapshots=rows)
    return dict(end_s=run['time_s'][-1],groups=groups,slit_perimeter_net_arrivals=arrivals,slit_diagnostics=slits,
        chamber_peak_bar=max(dataio.chamber_pressure_pa(run))/1e5,
        final_temperature_c={k:v[-1]-273.15 for k,v in run['temperature_k'].items()},
        residuals={k:v for k,v in run.items()if k.startswith('max_')and 'residual'in k},
        speed= (run['advanced_to_s']-run['started_at_s'])/run['wall_s'])


def plot(out,scene,old,new):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    raw,_=dataio.load('no-cooling')
    labels={'101':'C1 / ISS','102':'C1 / TIM','C2_TIM':'C2 / TIM','C2_ISS':'C2 / ISS','107':'C3 / TIM','108':'C3 / ISS'}
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True,constrained_layout=True)
    for ax,(key,label)in zip(axes.flat,labels.items()):
        for r,color,ls,title in [(old,'#707070','--','Прежняя карта'),(new,'#0072b2','-','Полная камера')]:
            ax.plot(r['time_s'],np.asarray(r['temperature_k'][key])-273.15,color=color,ls=ls,label=title)
        t=np.asarray(new['time_s']);dense=raw['time_s'][(raw['time_s']>=t[0])&(raw['time_s']<=t[-1])]
        ax.plot(dense,dataio.observed(raw,dense)[key]-273.15,color='#d55e00',alpha=.8,label='Эксперимент')
        ax.set(title=label,ylabel='°C',xlabel='Время, с');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Без подбора параметров: меняется только газовая карта; старт 750 с')
    fig.savefig(out/'temperatures.png',dpi=150);plt.close(fig)
    meta=scene['metadata']['gas_geometry'];boxes=meta['void_boxes']
    # One real y-slice, not a projection that overlays unrelated volumes.
    steel=next(c for c in scene['cells']if c['id']=='steel-L')
    y=steel['pose']['position_m'][1]+steel['geometry']['dimensions_m'][1]/2+.01
    selected={n:b for n,b in boxes.items()if b['lo'][1]<y<b['hi'][1]}
    fig,axes=plt.subplots(1,3,figsize=(15,6),constrained_layout=True)
    indices=[int(np.argmin(abs(np.asarray(new['time_s'])-t)))for t in (780.,860.,950.)]
    vmax=max(new['gas'][n]['temperature_k'][i]-273.15 for n in selected for i in indices)
    norm=matplotlib.colors.Normalize(vmin=20.,vmax=max(100.,vmax));cmap=plt.get_cmap('inferno')
    for ax,i in zip(axes,indices):
        for name,b in selected.items():
            temp=new['gas'][name]['temperature_k'][i]-273.15
            ax.add_patch(Rectangle((b['lo'][0]*1000,b['lo'][2]*1000),(b['hi'][0]-b['lo'][0])*1000,
                (b['hi'][2]-b['lo'][2])*1000,facecolor=cmap(norm(temp)),edgecolor='#888888',lw=.25))
        for c in scene['cells']:
            if c['id']in ('C1','C2','C3'):
                x,z=c['pose']['position_m'][0],c['pose']['position_m'][2]
                ax.text(x*1000,z*1000,c['id'],ha='center',va='center',color='cyan',fontsize=8)
        ax.set(xlim=(-35,105),ylim=(-145,145),aspect='equal',xlabel='x, мм',ylabel='z, мм',title=f"Газ, {new['time_s'][i]:g} с")
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes,label='Газ, °C')
    fig.suptitle('Пространственный срез газа за боковой гранью модуля (y = край + 10 мм)')
    fig.savefig(out/'gas-map.png',dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--timeout',type=float,default=400.);a=ap.parse_args()
    base=a.baseline.resolve();out=a.output.resolve();out.mkdir(exist_ok=False)
    state=dict(status='running',pid=os.getpid(),chemistry_frozen=True,calibrated=False,
        native_sha256=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest(),baseline=str(base))
    def stop(_sig,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        seed=json.loads((base/'parameters.json').read_text());source=json.loads((base/'scenario.json').read_text())
        scene,_=fit.trial_scene(seed['parameters'],source['simulation']['duration_s'],seed['gas_extensions'],seed['numerical_controls'])
        scene['metadata']['zone_diagnostics']=True;scene['thermal_path_trace']=True
        mapping=unvented_remap(source,scene);cal.save(out/'scenario.json',scene);cal.save(out/'remap.json',mapping)
        cal.save(out/'parameters.json',seed);cal.save(out/'status.json',state)
        contract=json.loads((base/'status.json').read_text())['native_sha256']
        result=native(out/'scenario.json',out/'replay',dict(restart_from=str(base/'sealed-state.bin'),
            restart_contract=contract,restart_source_scenario=str(base/'scenario.json'),unvented_gas_remap=mapping),a.timeout)
        reference=load(base/'resumed/run.json.gz')
        report=dict(previous=distribution(source,reference),spatial=distribution(scene,result),
            rebase=result['rebase_initial_residuals'],changed='gas topology and wall-contact partition only',
            checkpoint_s=result['started_at_s'],chemistry_frozen=True,calibrated=False)
        cal.save(out/'comparison.json',report);plot(out,scene,reference,result)
        state.update(status='complete',comparison=report)
    except KeyboardInterrupt:state.update(status='stopped')
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:cal.save(out/'status.json',state);print(json.dumps(state),flush=True)


if __name__=='__main__':main()
