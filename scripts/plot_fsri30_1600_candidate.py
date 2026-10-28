"""Render the full conditioned candidate against raw T5 and unconfirmed T4."""
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fsri30_data import load

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'data/development/fsri30-b17-inner15-from1600-hot075-20261027-r1'

def main():
    with gzip.open(RUN/'replay/run.json.gz','rt') as f:d=json.load(f)
    e,_=load();t=np.array(d['time_s']);mask=(e['time_s']>=1600)&(e['time_s']<=2200)
    fig,ax=plt.subplots(2,1,figsize=(13,9),sharex=True,constrained_layout=True)
    ax[0].plot(t,np.array(d['cell_mean_temperature_k']['B10'])-273.15,label='model B10 mean')
    ax[0].plot(e['time_s'][mask],e['T5_k'][mask]-273.15,'--',label='experiment T5')
    for name in ('B17','B18'):ax[1].plot(t,np.array(d['cell_mean_temperature_k'][name])-273.15,label='model '+name+' mean')
    ax[1].plot(e['time_s'][mask],e['T4_k'][mask]-273.15,'--',label='experiment T4; mapping unverified')
    for a in ax:a.set_ylabel('Temperature, C');a.grid(alpha=.25);a.legend()
    ax[1].set_xlabel('Time, s, unshifted');fig.suptitle('Conditioned from 1600 s; local B17/B18 conductivity x1.5')
    target=RUN/'candidate-vs-experiment.png';fig.savefig(target,dpi=160);print(target)

if __name__=='__main__':main()
