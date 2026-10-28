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
    cli.add_argument('--stage',choices=['heat','failure','validate','trial'],default='heat')
    cli.add_argument('--partial',action='store_true',help='Plot an explicitly incomplete checkpoint, never as a full replay')
    args=cli.parse_args();prefix=args.stage+('-partial' if args.partial else '')
    filename='last-incomplete-run.json.gz' if args.partial else args.stage+'-run.json.gz'
    with gzip.open(args.directory/filename,'rt') as stream:run=json.load(stream)
    name='active-cooling' if args.stage=='validate' else 'no-cooling'
    data,_=dataio.load(name);marker=dataio.event(data);at=np.asarray(run['time_s']);obs=dataio.observed(data,at)
    figure,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
    for ax,key in zip(axes.flat,dataio.CHANNELS):
        ax.plot(at,obs[key]-273.15,color='#d95f02',lw=1.6,label='Experiment')
        ax.plot(at,np.asarray(run['temperature_k'][key])-273.15,color='#0072b2',lw=1.6,label='R123 + Joule CUDA')
        if marker['time_s']<=at[-1]:ax.axvline(marker['time_s'],color='0.5',ls=':',label='Measured vent marker')
        ax.set_title(LABELS[key]);ax.grid(alpha=.2);ax.set_ylabel('Temperature, °C')
    axes[0,0].legend(fontsize=8);axes[1,0].set_xlabel('Physical time from experiment start, s')
    figure.suptitle(name+' — '+prefix+' (no fitted time shift)');figure.tight_layout()
    figure.savefig(args.directory/(prefix+'-temperatures.png'),dpi=150)
    if marker['time_s']<=at[-1]:
        for ax in axes.flat:ax.set_xlim(marker['time_s']-20,marker['time_s']+300)
        figure.savefig(args.directory/(prefix+'-event-temperatures.png'),dpi=150)
    plt.close(figure)
    figure,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    axes[0].plot(at,obs['pressure_bar'],color='#d95f02',label='Experiment chamber')
    axes[0].plot(at,np.asarray(dataio.chamber_pressure_pa(run))/1e5,color='#0072b2',label='Calculated chamber')
    axes[0].set_ylabel('Absolute pressure, bar');axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].plot(at,run['joule_energy_j']['C2'],label='C2 Joule heat',color='#009e73')
    r2=np.asarray(run['released_energy_j_history']).reshape(len(at),len(run['chemistry'])).sum(axis=1)
    axes[1].plot(at,r2,label='Total R2 released thermal energy',color='#e69f00')
    gas_energy=np.zeros_like(at)
    for gas in run['gas'].values():gas_energy+=np.asarray(gas['released_energy_j'])
    axes[1].plot(at,gas_energy,label='Total R3 gas heat',color='#cc79a7')
    axes[1].set_ylabel('Cumulative energy, J');axes[1].set_xlabel('Physical time, s');axes[1].grid(alpha=.2);axes[1].legend()
    figure.tight_layout();figure.savefig(args.directory/(prefix+'-pressure-energy.png'),dpi=150)
    plt.close(figure)
    figure,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    c=run['chemistry']['C2']
    for key,label,color in [('virgin_kg','A: initial material','#0072b2'),
                            ('r2_precursor_kg','B: available R2 precursor','#e69f00'),
                            ('residue_kg','S/C: condensed products','#009e73')]:
        axes[0].plot(at,np.asarray(c[key])*1000,label=label,color=color)
    axes[0].set_ylabel('C2 condensed inventory, g');axes[0].legend();axes[0].grid(alpha=.2)
    core=np.asarray(run['cell_mean_temperature_k']['C2'])-273.15
    axes[1].plot(at,core,label='Calculated C2 bulk mean',color='#cc79a7')
    for key,color in [('C2_TIM','#0072b2'),('C2_ISS','#d95f02')]:
        axes[1].plot(at,np.asarray(run['temperature_k'][key])-273.15,label='Calculated '+key,color=color,ls='--')
        axes[1].plot(at,obs[key]-273.15,label='Observed '+key,color=color)
    for ax in axes:
        if at[-1]>=marker['time_s']:ax.axvline(marker['time_s'],color='0.5',ls=':')
    axes[1].set_ylabel('Temperature, °C');axes[1].set_xlabel('Physical time, s');axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
    figure.tight_layout();figure.savefig(args.directory/(prefix+'-chemistry-core.png'),dpi=150)
    plt.close(figure)
    if 'electrical_chemical_input_j' in c:
        figure,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
        mid=.5*(at[1:]+at[:-1]);dt=np.diff(at)
        for values,label,color in [(run['joule_energy_j']['C2'],'C2 Joule heat','#009e73'),
                                    (r2,'Total R2 local heat','#e69f00'),
                                    (gas_energy,'Total R3 heat','#cc79a7')]:
            axes[0].plot(mid,np.diff(np.asarray(values))/dt,label=label,color=color)
        axes[0].set_ylabel('Average thermal power, W')
        axes[1].plot(at,c['electrical_chemical_input_j'],label='Electrical work added to chemistry (not heat)',color='#0072b2')
        axes[1].plot(at,c['precursor_charge_energy_j'],label='Added electrical energy still stored in B',color='#e69f00')
        axes[1].set_ylabel('Chemical energy / work, J')
        for key,label,color,style in [('r2_precursor_kg','Available B','#e69f00','-'),
                                     ('precursor_from_charge_kg','B formed by charging (cumulative)','#0072b2','--'),
                                     ('precursor_from_degradation_kg','B formed by degradation (cumulative)','#009e73',':')]:
            axes[2].plot(at,np.asarray(c[key])*1000,label=label,color=color,ls=style)
        axes[2].set_ylabel('C2 precursor inventory / feed, g');axes[2].set_xlabel('Physical time, s')
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
        figure.tight_layout();figure.savefig(args.directory/(prefix+'-source-storage.png'),dpi=150)
        plt.close(figure)
    figure,axes=plt.subplots(1,3,figsize=(14,4),sharex=True)
    for ax,(label,a,b) in zip(axes,[('C1: ISS − TIM','101','102'),
                                  ('C2: ISS − TIM','C2_ISS','C2_TIM'),
                                  ('C3: TIM − ISS','107','108')]):
        ax.plot(at,obs[a]-obs[b],label='Experiment',color='#d95f02')
        ax.plot(at,np.asarray(run['temperature_k'][a])-np.asarray(run['temperature_k'][b]),
                label='Calculated',color='#0072b2',ls='--')
        ax.set_title(label);ax.set_ylabel('Surface contrast, K');ax.set_xlabel('Physical time, s');ax.grid(alpha=.2)
        if marker['time_s']<=at[-1]:ax.axvline(marker['time_s'],color='0.5',ls=':')
    axes[0].legend();figure.tight_layout();figure.savefig(args.directory/(prefix+'-surface-contrasts.png'),dpi=150)
    plt.close(figure)
    print(args.directory/(prefix+'-temperatures.png'))


if __name__=='__main__':main()
