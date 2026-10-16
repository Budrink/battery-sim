#!/usr/bin/env python3
"""Render measured versus CUDA-predicted temperatures from a saved report."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from generate_upv import ROOT, TARGETS, load_case

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('report',type=Path)
p.add_argument('--data',type=Path,default=ROOT/'data/experiments/upv')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();report=json.loads(a.report.read_text());run=report['result']
if report.get('status')=='rejected':raise ValueError('Cannot present a rejected calibration report as a valid fit')
data,*_=load_case(a.data,report['experiment'])
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,constrained_layout=True)
labels={'101':'C1 — ISS face','102':'C1 — TIM face','107':'C3 — TIM face','108':'C3 — ISS face'}
for ax,c in zip(axes.flat,TARGETS):
    stop=next((i for i,t in enumerate(data['time_s']) if t>run['time_s'][-1]),len(data['time_s']))
    ax.plot(data['time_s'][:stop],[v-273.15 for v in data[f'T{c}_k'][:stop]],label='Measured',color='#273fa5',lw=1.6)
    ax.plot(run['time_s'],[v-273.15 for v in run['temperature_k'][c]],label='CUDA prediction',color='#da551d',lw=1.6,ls='--')
    ax.set_title(f"{labels[c]} (channel {c}); RMSE {report['metrics']['per_channel'][c]['rmse_k']:.2f} K")
    ax.set_xlabel('Time since recording start, s');ax.set_ylabel('Temperature, °C');ax.grid(alpha=.2);ax.legend()
stage='Calibration' if report['kind']=='fit' else 'Frozen-parameter validation'
topology=' | C1/C3 numbering hypothesis (unconfirmed)' if report.get('topology')=='swapped-witnesses' else ''
fig.suptitle(f"UPV {report['experiment']} — {stage}, RMSE {report['metrics']['rmse_k']:.2f} K\nConditional pre-event heat transfer{topology}",fontsize=12)
a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=150)
print(a.output)
