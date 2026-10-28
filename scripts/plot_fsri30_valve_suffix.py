"""Render completed conditioned valve suffixes; physical time is not shifted."""
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/development/fsri30-valve-suffix-sweep-20261027-r1'
CONTROL=ROOT/'data/development/fsri30-valve-control-1600-1800-20261027-r1'


def load(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def main():
    fig,axes=plt.subplots(3,2,figsize=(15,11),sharex=True,constrained_layout=True)
    records=[('80 kPa + shell damage',CONTROL)]+[(f'{v} kPa, valve only',OUT/f'valve-{v}kpa') for v in (40,80,160)]
    for label,folder in records:
        file=folder/'replay/run.json.gz'
        if not file.exists():continue
        d=load(file);scene=json.loads((folder/'scenario.json').read_text());t=np.asarray(d['time_s'])
        for col,cell in enumerate(('B10','B17')):
            axes[0,col].plot(t,np.asarray(d['cell_mean_temperature_k'][cell])-273.15,label=label)
            key=f'rupture-{cell}-1';edge=next(e for e in scene['gas_network']['channels'] if e['id']==key)
            pin=np.asarray(d['gas'][edge['from']]['pressure_pa']);pout=np.asarray(d['gas'][edge['to']]['pressure_pa'])
            axes[1,col].plot(t,(pin-pout)/1000.,label=label)
            axes[2,col].plot(t,np.asarray(d['channels'][key]['opening_fraction'])*100,label=label)
    for col,cell in enumerate(('B10','B17')):
        axes[0,col].set_title(cell);axes[0,col].set_ylabel('Mean temperature, C')
        axes[1,col].set_ylabel('Valve pressure difference, kPa')
        axes[2,col].set_ylabel('Open patch area, %');axes[2,col].set_xlabel('Simulation time, s')
        for row in range(3):axes[row,col].grid(alpha=.25)
    axes[0,0].legend(fontsize=8);fig.suptitle('Conditioned restart at 1600 s; TCs not mapped')
    target=OUT/'valve-suffix-comparison.png';fig.savefig(target,dpi=160);print(target)


if __name__=='__main__':main()
