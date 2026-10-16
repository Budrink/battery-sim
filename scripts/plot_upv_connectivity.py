#!/usr/bin/env python3
"""Display the fitted contact law and instantaneous CUDA interface conductance."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calibrate_upv import CONTACTS

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--profile',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
profile=json.loads(a.profile.read_text());report=json.loads(a.report.read_text());params=profile['parameters']
if profile['parameter_set']!='state_geometry' or report['parameters']!=params:raise ValueError('Matching state geometry fit required')
fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
temperature=np.linspace(20,150,400);theta=(temperature-25)/100
for key in CONTACTS:
    h0=params[key]*100;z=np.log((h0-1)/(10000-h0))+params['surface_linear']*theta+params['surface_quadratic']*theta**2
    axes[0].plot(temperature,1+9999/(1+np.exp(-z)),label=key.removesuffix('_scale').replace('_',' / '))
axes[0].set(xlabel='Both endpoint temperatures, °C (D = 0)',ylabel='Contact h, W/(m² K)',title='Fitted structural response, equal-temperature section')
run=report['result'];selected=('ISS2--Al1','Al1--ISS2','Al2--ISS3','Al3--ISS4','ISS3--C1','C3--TIM1')
for name,track in run['interface_snapshot'].items():
    if name in selected:
        axes[1].plot(run['time_s'],track['conductance_w_per_k'],label=name)
axes[1].set(xlabel='Time, s',ylabel='Total G, W/K (incl. series bulk resistance)',title='Local law evaluated on simulated states')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Conditional pre-event fit; destruction and high-temperature extrapolation are NOT calibrated',fontsize=11)
a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=150);print(a.output)
