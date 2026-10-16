#!/usr/bin/env python3
"""Experimental curves and individual seeds; the seed range is not a confidence interval."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import calibrate_upv_stochastic as model


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,default=model.OUT);folder=p.parse_args().output
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    _,prior=model.geometry.inputs('no-cooling');obs_t=np.asarray(prior['result']['time_s']);mask=obs_t<=5
    for ax,(label,_) in zip(axes.flat,model.CASES):
        curves=[]
        for seed in model.SEEDS:
            path=folder/f'{label}-seed{seed}.json'
            if not path.exists():continue
            report=model.geometry.read(path);r=report['result'];t=np.asarray(r['time_s']);y=np.asarray(r['gas']['chamber']['pressure_pa'])/1e5
            ax.plot(t,y,lw=.8,alpha=.6,label=f'seed {seed}');curves.append(y)
        if curves:ax.plot(t,np.mean(curves,axis=0),color='#c43c39',lw=1.8,label='Seed mean')
        ax.plot(obs_t[mask],np.asarray(prior['observed_pressure_bar'])[mask],'k.-',label='Experiment')
        ax.set(title=label,xlim=(0,5),xlabel='Time after event [s]',ylabel='Chamber pressure [bar abs]');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Training experiment: stochastic barrier trials, frozen thermal inputs and chemistry')
    fig.savefig(folder/'pressure-search.png',dpi=140);plt.close(fig)
    selection=folder/'selection.json'
    if not selection.exists():return
    chosen=model.geometry.read(selection)['id']
    fig,axes=plt.subplots(5,2,figsize=(13,16),constrained_layout=True)
    for col,(mode,name,seeds) in enumerate([(chosen,'no-cooling',model.SEEDS),('validation','active-cooling',model.AUDIT_SEEDS)]):
        _,prior=model.geometry.inputs(name);obs_t=np.asarray(prior['result']['time_s']);mask=obs_t<=30
        for row,key in enumerate(['pressure','101','102','107','108']):
            ax=axes[row,col];ys=[]
            for seed in seeds:
                path=folder/f'{mode}-seed{seed}.json'
                if not path.exists():continue
                r=model.geometry.read(path)['result'];t=r['time_s']
                y=np.asarray(r['gas']['chamber']['pressure_pa'])/1e5 if key=='pressure' else np.asarray(r['temperature_k'][key])-273.15
                ax.plot(t,y,lw=.8,alpha=.6,label=f'seed {seed}');ys.append(y)
            if ys:ax.plot(t,np.mean(ys,axis=0),color='#c43c39',lw=1.6,label='Seed mean')
            y=np.asarray(prior['observed_pressure_bar']) if key=='pressure' else np.asarray(prior['observed_temperature_k'][key],dtype=float)-273.15
            ax.plot(obs_t[mask],y[mask],'k.-',lw=1,label='Experiment')
            labels={'pressure':'Chamber pressure [bar abs]','101':'C1 / ISS [°C]','102':'C1 / TIM [°C]','107':'C3 / TIM [°C]','108':'C3 / ISS [°C]'}
            ax.set(title=name+' — '+labels[key],xlim=(0,5 if row==0 else 30),xlabel='Time after event [s]');ax.grid(alpha=.2)
            if row==0:ax.legend(fontsize=7)
    fig.suptitle('Frozen stochastic candidate: training / second-experiment diagnostics, NOT an accepted fit')
    fig.savefig(folder/'experiment-comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
