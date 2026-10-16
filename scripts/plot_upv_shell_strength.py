#!/usr/bin/env python3
"""Pressure accumulation and blowdown versus casing-strength factor."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import upv_mass_paths as model


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=model.ROOT/'data/calibration/upv-shell-strength-v16')
    args=parser.parse_args();folder=args.output
    fig,axes=plt.subplots(3,2,figsize=(13,12),constrained_layout=True)
    for factor,color in [(1,'#777777'),(2,'#1b9e77'),(4,'#d95f02'),(8,'#7570b3')]:
        path=folder/f'strength-{factor}.json'
        if not path.exists():
            continue
        run=model.read(path)['result'];t=np.asarray(run['time_s']);label=f'{factor}× reference strength'
        rupture=[k for k in run['channels'] if k.startswith('C2-progressive')]
        mass=np.sum([run['channels'][k]['transferred_mass_kg'] for k in rupture],axis=0)
        quantities=[np.asarray(run['gas']['chamber']['pressure_pa'])/1e5,
                    np.max([v['pressure_pa'] for k,v in run['gas'].items() if k.startswith('inside-C2')],axis=0)/1e5,
                    mass*1000,np.asarray(run['cell_mean_temperature_k']['C2'])-273.15,
                    np.r_[0,np.diff(mass)/np.diff(t)]*1000,np.asarray(run['temperature_k']['108'])-273.15]
        for ax,y in zip(axes.flat,quantities):
            ax.plot(t,y,color=color,label=label,lw=1.6)
    _,prior=model.inputs('no-cooling');t=np.asarray(prior['result']['time_s']);mask=t<=30
    axes[0,0].plot(t[mask],np.asarray(prior['observed_pressure_bar'])[mask],'k.-',label='Experiment',lw=1)
    axes[2,1].plot(t[mask],np.asarray(prior['observed_temperature_k']['108'],dtype=float)[mask]-273.15,'k.-',label='Experiment',lw=1)
    labels=['Chamber pressure [bar abs]','Maximum source-pore pressure [bar abs]',
            'Net exported mass [g]','C2 mean solid temperature [°C]',
            'Interval-mean net outflow [g/s]','C3 / ISS surface (ch108) [°C]']
    for i,(ax,label) in enumerate(zip(axes.flat,labels)):
        ax.set(xlim=(0,30 if i==5 else 5),xlabel='Time from event [s]',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Only casing reference pressure strength changes\nThermal properties, energy, reaction rate, gap geometry, seam ratio and softening are frozen')
    fig.savefig(folder/'strength-comparison.png',dpi=140);fig.savefig(folder/'strength-comparison.svg')


if __name__=='__main__':
    main()
