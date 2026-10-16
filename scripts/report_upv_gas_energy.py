#!/usr/bin/env python3
"""Plot frozen joint trial against original observations, including early pressure."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import calibrate_upv_gas_energy as g
from calibrate_upv import save


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,default=g.OUT)
    p.add_argument('--comparison',type=Path,default=g.t.ROOT/'data/calibration/upv-gas-energy-v6');a=p.parse_args()
    pressure_fig,pa=plt.subplots(2,2,figsize=(12,7));temp_fig,ta=plt.subplots(4,2,figsize=(12,10))
    assessment=dict(protocol=g.PROTOCOL,default_scene_changed=False,experiments={})
    for col,(mode,name) in enumerate([('fit','no-cooling'),('validate','active-cooling')]):
        path=a.output/(mode+'.json')
        if not path.exists():continue
        r=json.loads(path.read_text());v=r['result'];at=np.asarray(v['time_s']);pressure=np.asarray(v['gas']['chamber']['pressure_pa'])/1e5
        old=json.loads((a.comparison/(mode+'.json')).read_text())
        dense_path=a.output/'dense-check'/(name+'.json')
        dense=json.loads(dense_path.read_text()) if dense_path.exists() else None
        old_dense_path=a.comparison/'dense-check'/(name+'.json')
        old_dense=json.loads(old_dense_path.read_text()) if old_dense_path.exists() else None
        for row in (0,1):
            ax=pa[row,col];ax.plot(at,r['observed_pressure_bar'],color='#444444',label='Experiment')
            if row==0 and dense:
                d=dense['refined'];ax.plot(d['time_s'],np.asarray(d['gas']['chamber']['pressure_pa'])/1e5,color='#d1495b',label='Closed gas energy (dense check)')
            else:ax.plot(at,pressure,color='#d1495b',label='Closed gas energy')
            # Compare equally resolved early curves; sparse old output must not
            # make a sampling difference look like a model improvement.
            previous=old_dense['refined'] if row==0 and old_dense else old['result']
            ax.plot(previous['time_s'],np.asarray(previous['gas']['chamber']['pressure_pa'])/1e5,'--',color='#287c9b',label='Previous frozen fit')
            ax.set_xlim(0,10 if row==0 else 1800);ax.set_title(name);ax.set_ylabel('absolute pressure [bar]');ax.set_xlabel('time from last pre-event scan [s]');ax.grid(alpha=.2)
        pa[0,col].legend(fontsize=8)
        obs=r.get('observed_temperature_k')
        if obs is None:_,obs,_,_=g.prepare(name)
        for row,(key,label) in enumerate([('101','C1 / ISS'),('102','C1 / TIM'),('107','C3 / TIM'),('108','C3 / ISS')]):
            ax=ta[row,col];mask=at>=300
            ax.plot(at[mask],np.asarray(obs[key],dtype=float)[mask]-273.15,color='#444444',label='Experiment')
            ax.plot(at[mask],np.asarray(v['temperature_k'][key])[mask]-273.15,color='#d1495b',label='Closed gas energy')
            old_at=np.asarray(old['result']['time_s']);old_mask=old_at>=300
            ax.plot(old_at[old_mask],np.asarray(old['result']['temperature_k'][key])[old_mask]-273.15,'--',color='#287c9b',label='Previous frozen fit')
            ax.set_title(name+' — '+label);ax.set_ylabel('°C');ax.grid(alpha=.2)
        ta[0,col].legend(fontsize=8);ta[-1,col].set_xlabel('time from last pre-event scan [s]')
        assessment['experiments'][name]=dict(metrics=r['metrics'],refinement=r.get('refinement'),refined_metrics=r.get('refined_metrics'),
            previous_late_witness_rmse_k=old['metrics'].get('witness',old['metrics'])['rmse_k'],comparison_directory=str(a.comparison),ledgers={k:x for k,x in v.items() if k.startswith('max_') and 'residual' in k},
            rupture_events=[e for e in v['events'] if e['kind']=='shell_rupture'])
        if dense:assessment['experiments'][name]['dense_check']=dense['summary']
    for fig,name,title in [(pressure_fig,'pressure-comparison.png','Gas pressure — measured scans and modeled gas, no prescribed hot ambient'),
                           (temp_fig,'witness-comparison.png','Late witness response — current and previous frozen fits')]:
        fig.suptitle(title,fontsize=12);fig.tight_layout();fig.savefig(a.output/name,dpi=150);plt.close(fig)
    profile=a.output/'profile.json'
    if profile.exists():assessment['profile']=json.loads(profile.read_text())
    save(a.output/'assessment.json',assessment);print(json.dumps(assessment,indent=2))


if __name__=='__main__':main()
