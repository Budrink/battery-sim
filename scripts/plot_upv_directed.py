#!/usr/bin/env python3
"""Plot both ordered conductances at identical instantaneous node states.

Reverse G is a diagnostic capacity, not a simultaneous uphill heat current.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.report.read_text());run=r['result']
if r['protocol']['revision']!='directed-geometry-v6':raise ValueError('Directed v6 report required')
names=('ISS1--C3','C3--TIM1','Al2--ISS3','ISS3--C1')
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,constrained_layout=True)
for ax,name in zip(axes.flat,names):
    v=run['interface_snapshot'][name];a_node,b_node=name.split('--')
    ax.plot(run['time_s'],v['a_to_b_conductance_w_per_k'],label=f'{a_node} → {b_node}',color='#d25425')
    ax.plot(run['time_s'],v['b_to_a_conductance_w_per_k'],label=f'{b_node} → {a_node}',color='#276bad',ls='--')
    ax.plot(run['time_s'],v['conductance_w_per_k'],label='Selected by local temperature gradient',color='#222222',ls=':',lw=1.)
    ax.set(title=name,xlabel='Time, s',ylabel='G including bulk resistance, W/K');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Directed contact law: two capacities, one conservative downhill heat exchange\nReverse G evaluated at the SAME states; not a measured reverse-heating experiment',fontsize=11)
a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=150);print(a.output)
