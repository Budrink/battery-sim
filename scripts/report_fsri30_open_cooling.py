#!/usr/bin/env python3
"""Compare completed open-boundary trials, without invented TC assignments."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np

from fsri30_data import load,save,sha
from fsri30_objective import model_fields
from check_fsri30_open_boundary import CLOSED,BASE,cooling_metrics


def report(out,folders,allow_prefix=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=False);data,_=load()
    fig,axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
    observed=np.array([data[f'T{i}_k'] for i in range(1,10)]).T
    for ax in axes:
        ax.fill_between(data['time_s'],observed.min(axis=1)-273.15,observed.max(axis=1)-273.15,
            color='.65',alpha=.22,label='Experiment: 9 TC range')
        for i in range(9):ax.plot(data['time_s'],observed[:,i]-273.15,color='.5',alpha=.4,lw=.5)
        ax.plot(data['time_s'],np.median(observed,axis=1)-273.15,color='black',lw=1.8,label='Experiment: median')
    entries=[('Closed, archived core',CLOSED,None,'#777777')]
    colors=['#df7732','#2279a8','#3a9852']
    for i,folder in enumerate(folders):
        s=json.loads((folder/'scenario.json').read_text())
        scale=s['metadata'].get('exhaust_aperture_diagnostic',{}).get('scale',1.)
        path=folder/'replay/run.json.gz'
        if not path.exists() and allow_prefix:path=folder/'replay/incomplete-run.json.gz'
        entries.append((f'Open; module vent area x{scale:g}',path,s,colors[i%len(colors)]))
    results=[]
    for label,path,scene,color in entries:
        with gzip.open(path,'rt') as f:raw=json.load(f)
        complete=bool(not raw.get('partial') and raw['time_s'][-1]>=5945.-1e-7)
        if raw['started_at_s']!=0. or (not complete and not allow_prefix):
            raise ValueError('Completed fresh full trajectory required, not an extrapolated tail')
        if not complete:label+=f" [PREFIX to {raw['time_s'][-1]:g} s]"
        t,_,v=model_fields(raw);q=np.quantile(v,[0.,.5,1.],axis=1).T-273.15
        for ax in axes:
            ax.plot(t,q[:,1],c=color,lw=1.8,label=label+' median')
            ax.plot(t,q[:,0],c=color,lw=1.,ls='--',alpha=.8)
            ax.plot(t,q[:,2],c=color,lw=1.,ls='--',alpha=.8)
        row=dict(label=label,path=str(path.resolve()),sha256=sha(path),cooling=cooling_metrics(raw,data),
            known_until_s=float(t[-1]),eligible_for_full_horizon_selection=complete)
        if scene:
            row.update(module_vent_areas_m2=[c['area_m2'] for c in scene['gas_network']['channels'] if c['id'].startswith('vent-')],
                boundary=scene['metadata']['ambient_boundary'],wall_s=raw['wall_s'],
                realtime_ratio=(raw['advanced_to_s']-raw['started_at_s'])/raw['wall_s'],
                balances={k:v for k,v in raw.items()if k.startswith('max_')and'residual'in k})
        results.append(row);del raw
    for ax,title in zip(axes,('Full history','Cooling tail')):
        ax.set(title=title,xlabel='Time, s',ylabel='Temperature, °C');ax.grid(alpha=.2)
        ax.legend(fontsize=7,loc='upper left')
    axes[1].set_xlim(3000.,5945.)
    fig.suptitle('20 mm; chemistry, materials and internal thermal contacts unchanged\nSolid: medians; dashed: model min/max. TC mapping unresolved — NOT independent validation')
    fig.savefig(out/'cooling-comparison.png',dpi=160);plt.close(fig)
    value=dict(results=results,calibrated=False,validation_passed=False,scope='Boundary sensitivity; exploratory envelopes only. Prefixes end at their actual observations and are not full-horizon candidates.')
    save(out/'comparison.json',value)
    return value


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--allow-prefix',action='store_true',help='Explicit diagnostic prefixes, NOT full scores or future cooling')
    ap.add_argument('folders',type=Path,nargs='+');args=ap.parse_args();report(args.output,args.folders,args.allow_prefix)


if __name__=='__main__':main()
