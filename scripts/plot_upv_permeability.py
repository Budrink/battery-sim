#!/usr/bin/env python3
"""Pressure-only barrier search, with temperature/energy diagnostics kept visible."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import calibrate_upv_permeability as model
from plot_upv_mass_paths import plot_case, plot_process


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,default=model.OUT)
    folder=parser.parse_args().output
    fig,axes=plt.subplots(3,2,figsize=(13,12),constrained_layout=True)
    colors=['#777777','#1b9e77','#d95f02','#7570b3','#e7298a']
    traces=[(label,color) for (label,_),color in zip(model.CASES,colors)]
    traces += [('stiff-coarse','#d95f02'),('stiff-fine','#1b9e77')]
    for label,color in traces:
        path=folder/(label+'.json')
        if not path.exists():
            continue
        run=model.geometry.read(path)['result'];t=np.asarray(run['time_s'])
        rupture=[v for k,v in run['channels'].items() if k.startswith('C2-progressive')]
        mass=np.sum([c['transferred_mass_kg'] for c in rupture],axis=0)
        # Acquisition and dense-output clocks can differ by nanoseconds.
        # Use declared 20 ms bins, not derivatives across those tiny intervals.
        bins=np.arange(0.,min(5.,t[-1])+.01,.02)
        bin_flow=np.diff(np.interp(bins,t,mass))/np.diff(bins)*1000
        series=[np.asarray(run['gas']['chamber']['pressure_pa'])/1e5,
            np.max([v['pressure_pa'] for k,v in run['gas'].items() if k.startswith('inside-C2')],axis=0)/1e5,
            mass*1000,np.r_[0,np.diff(mass)/np.diff(t)]*1000,
            np.asarray(run['cell_mean_temperature_k']['C2'])-273.15,
            np.asarray(run['temperature_k']['108'])-273.15]
        for i,(ax,y) in enumerate(zip(axes.flat,series)):
            ax.plot((bins[1:]+bins[:-1])/2 if i==3 else t,
                    bin_flow if i==3 else y,color=color,label=label,lw=1.5)
    _,prior=model.geometry.inputs('no-cooling');t=np.asarray(prior['result']['time_s']);mask=t<=30
    axes[0,0].plot(t[mask],np.asarray(prior['observed_pressure_bar'])[mask],'k.-',label='Experiment')
    axes[2,1].plot(t[mask],np.asarray(prior['observed_temperature_k']['108'],dtype=float)[mask]-273.15,'k.-',label='Experiment')
    labels=['Chamber pressure [bar abs]','Maximum local source pressure [bar abs]',
            'Net exported mass [g]','20 ms mean net mass flow [g/s]',
            'C2 mean solid temperature [°C]','C3 / ISS surface (ch108) [°C]']
    for i,(ax,label) in enumerate(zip(axes.flat,labels)):
        ax.set(xlim=(0,30 if i==5 else 5),xlabel='Time from event [s]',ylabel=label)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0,1].set_yscale('log')
    fig.suptitle('Barrier permeability diagnostics — thermal properties and chemistry frozen\nLocal source pressure is a model prediction, NOT chamber measurement')
    fig.savefig(folder/'permeability-comparison.png',dpi=140)
    fig.savefig(folder/'permeability-comparison.svg');plt.close(fig)
    audits=[folder/(name+'.json') for name in ('stiff-coarse','stiff-fine')]
    if all(p.exists() for p in audits):
        reports=[model.geometry.read(p)['result'] for p in audits]
        ids=sorted(k for k in reports[0]['channels'] if k.startswith('C2-progressive'))
        fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
        for ax,key in zip(axes.flat,ids):
            for run,label,color,style in zip(reports,['Original controls','Halved controls'],['#d95f02','#1b9e77'],['-','--']):
                ax.plot(run['time_s'],100*np.asarray(run['channels'][key]['opening_fraction']),
                        color=color,ls=style,label=label)
            ax.set(title=key,xlabel='Time from event [s]',ylabel='Open face area [%]')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Numerical aperture refinement — not an experimental fit')
        fig.savefig(folder/'aperture-refinement.png',dpi=140);plt.close(fig)
    for name,stem in [('no-cooling','fit'),('active-cooling','validation')]:
        if not (folder/(stem+'.json')).exists():
            continue
        report=model.geometry.read(folder/(stem+'.json'))
        scene=model.geometry.read(folder/(stem+'-scenario.json'));_,prior=model.geometry.inputs(name)
        plot_case(folder,name,report,prior,scene);plot_process(folder,name,report,scene)


if __name__=='__main__':
    main()
