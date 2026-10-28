"""Observed-prefix helpers: no live-state extrapolation, time shifts or fitting."""
from pathlib import Path
import gzip
import json

import numpy as np

from check_fsri30_resolved_gas import compact


def conversion_events(scene,raw):
    t=np.asarray(raw['time_s']);events=[]
    for cell in scene['cells']:
        if cell['role']!='battery':continue
        law=cell['conversion']
        if law.get('gas_yield_temperature_per_k',0.) or law.get('gas_yield_density_exponent',0.) or law['gas_yield']<=0.:
            raise ValueError('Conversion reporting requires the constant gas-yield ledger')
        converted=np.asarray(raw['chemistry'][cell['id']]['r2_generated_gas_kg'])/law['gas_yield']
        hit=np.flatnonzero(converted>=.1*law['mass_kg'])
        if len(hit):
            i=int(hit[0]);when=t[i] if i==0 else np.interp(.1*law['mass_kg'],converted[i-1:i+1],t[i-1:i+1])
            events.append(dict(component=cell['id'],time_s=float(when)))
    return sorted(events,key=lambda e:(e['time_s'],e['component']))


def load_available(folder):
    folder=Path(folder);scene=json.loads((folder/'scenario.json').read_text())
    path=folder/'replay/run.json.gz'
    if not path.exists():path=folder/'replay/incomplete-run.json.gz'
    with gzip.open(path,'rt') as f:raw=json.load(f)
    if raw['started_at_s']!=0.:raise ValueError('Expected a fresh trajectory, not a conditioned suffix')
    meta=dict(path=str(path.resolve()),partial=raw.get('partial',False),
        last_observation_s=float(raw['time_s'][-1]),advanced_to_s=float(raw['advanced_to_s']),
        live_state=raw.get('live_state'),
        balances={k:v for k,v in raw.items() if k.startswith('max_') and 'residual' in k})
    return compact(raw),conversion_events(scene,raw),meta,scene


def trim_observed(history,end):
    t=np.asarray(history['time_s']);count=int(np.searchsorted(t,end+1e-7,side='right'))
    if count<2:raise ValueError('Fewer than two completed observations')
    def visit(value):
        if isinstance(value,dict):return {k:visit(v) for k,v in value.items()}
        if isinstance(value,list):
            if len(value)!=len(t):raise ValueError('Not an aligned observation history')
            return value[:count]
        return value
    # Other metadata (including live_state) must never be sliced into a fake
    # observation. These three sections are explicitly sample-aligned arrays.
    result=dict(time_s=history['time_s'][:count])
    for key in ('cell_mean_temperature_k','process_totals','gas_groups'):result[key]=visit(history[key])
    return result


def summary_at(history,events):
    end=history['time_s'][-1];observed=[e for e in events if e['time_s']<=end+1e-7]
    return dict(all_gas_r3_j=float(history['process_totals']['gas_released_energy_j'][-1]),
        r2_converted_components=len(observed),model_conversion_events=observed)


def plot_prefix(out,runs,end,first,second):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for mm,style in ((20,'-'),(10,'--')):
        r=runs[mm];t=np.asarray(r['time_s']);label=f'{mm} mm'
        axes[0,0].plot(t,np.asarray(r['cell_mean_temperature_k']['B10'])-273.15,style,label=label)
        axes[0,1].plot(t,np.asarray(r['process_totals']['gas_released_energy_j'])/1000.,style,label=label)
        axes[1,0].plot(t,np.asarray(r['gas_groups']['outside']['maximum_temperature_k'])-273.15,style,label=label)
        axes[1,1].plot(t,np.asarray(r['cell_mean_temperature_k']['B09'])-273.15,style,label=label)
        for name,color in (('B11','tab:blue'),('B02','tab:orange')):
            axes[2,0].plot(t,np.asarray(r['cell_mean_temperature_k'][name])-273.15,style,color=color,label=f'{name}, {mm} mm')
        axes[2,1].plot(t,np.asarray(r['gas_groups']['outside']['maximum_temperature_k'])-273.15,style,label=label)
    titles=('B10 mean, °C','Cumulative R3, kJ','First burst: hottest gas, °C',
            'Cascade onset: B09 mean, °C','Other neighbours: means, °C','Cascade onset: hottest gas, °C')
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title);ax.set_xlabel('s');ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlim(0.,end)
    axes[1,0].set_xlim(max(0.,first-50.),min(end,first+150.))
    for ax in (axes[1,1],axes[2,0],axes[2,1]):ax.set_xlim(max(0.,second-60.),end)
    fig.suptitle(f'Common completed snapshots only: 0–{end:g} s; no time shifts\nLater 10 mm cascade/cooling NOT observed')
    path=Path(out)/'comparison.png';fig.savefig(path,dpi=160);plt.close(fig)
    return path
