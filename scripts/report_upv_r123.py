#!/usr/bin/env python3
"""Compare current CUDA trajectories with immutable observations, no time shift."""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import upv_r123_data as dataio

LABELS={'101':'C1 / ISS','102':'C1 / TIM','107':'C3 / TIM','108':'C3 / ISS',
        'C2_TIM':'C2 / TIM','C2_ISS':'C2 / ISS'}


def main():
    cli=argparse.ArgumentParser(__doc__);cli.add_argument('directory',type=Path)
    cli.add_argument('--stage',choices=['heat','failure','validate','trial'],default='heat');args=cli.parse_args()
    with gzip.open(args.directory/(args.stage+'-run.json.gz'),'rt') as stream:run=json.load(stream)
    name='active-cooling' if args.stage=='validate' else 'no-cooling'
    data,_=dataio.load(name);marker=dataio.event(data);at=np.asarray(run['time_s']);obs=dataio.observed(data,at)
    figure,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
    for ax,key in zip(axes.flat,dataio.CHANNELS):
        ax.plot(at,obs[key]-273.15,color='#d95f02',lw=1.6,label='Experiment')
        ax.plot(at,np.asarray(run['temperature_k'][key])-273.15,color='#0072b2',lw=1.6,label='R123 + Joule CUDA')
        if marker['time_s']<=at[-1]:ax.axvline(marker['time_s'],color='0.5',ls=':',label='Measured vent marker')
        ax.set_title(LABELS[key]);ax.grid(alpha=.2);ax.set_ylabel('Temperature, °C')
    axes[0,0].legend(fontsize=8);axes[1,0].set_xlabel('Physical time from experiment start, s')
    figure.suptitle(name+' — '+args.stage+' (no fitted time shift)');figure.tight_layout()
    figure.savefig(args.directory/(args.stage+'-temperatures.png'),dpi=150)
    if marker['time_s']<=at[-1]:
        for ax in axes.flat:ax.set_xlim(marker['time_s']-20,marker['time_s']+300)
        figure.savefig(args.directory/(args.stage+'-event-temperatures.png'),dpi=150)
    plt.close(figure)
    figure,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    axes[0].plot(at,obs['pressure_bar'],color='#d95f02',label='Experiment chamber')
    axes[0].plot(at,np.asarray(run['gas']['chamber']['pressure_pa'])/1e5,color='#0072b2',label='Calculated chamber')
    axes[0].set_ylabel('Absolute pressure, bar');axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].plot(at,run['joule_energy_j']['C2'],label='C2 Joule heat',color='#009e73')
    gas_energy=np.zeros_like(at)
    for gas in run['gas'].values():gas_energy+=np.asarray(gas['released_energy_j'])
    axes[1].plot(at,gas_energy,label='Total R3 gas heat',color='#cc79a7')
    axes[1].set_ylabel('Cumulative energy, J');axes[1].set_xlabel('Physical time, s');axes[1].grid(alpha=.2);axes[1].legend()
    figure.tight_layout();figure.savefig(args.directory/(args.stage+'-pressure-energy.png'),dpi=150)
    print(args.directory/(args.stage+'-temperatures.png'))


if __name__=='__main__':main()
