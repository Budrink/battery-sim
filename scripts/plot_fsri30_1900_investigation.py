"""Compare conditioned 1900 s B17 diagnostics and unpaired experimental T4."""
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fsri30_data import load

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/development'
RUNS=[('original x1',D/'fsri30-b17-hot-film-075-2200-20261027-r1/replay/incomplete-run.json.gz'),
      ('gas .625 after 1900',D/'fsri30-b17-hot-film-0625-from1900-20261027-r1/replay/run.json.gz'),
      ('internal x1.5 after 1900',D/'fsri30-b17-internal-x15-from1900-20261027-r1/replay/run.json.gz'),
      ('internal x2 after 1900',D/'fsri30-b17-internal-x2-from1900-20261027-r1/replay/run.json.gz')]

def main():
    fig,ax=plt.subplots(2,2,figsize=(15,9),sharex=True,constrained_layout=True)
    for name,path in RUNS:
        with gzip.open(path,'rt') as f:d=json.load(f)
        t=np.array(d['time_s']);mask=(t>=1900)&(t<=2200);tt=t[mask]
        ax[0,0].plot(tt,(np.array(d['cell_mean_temperature_k']['B17'])-273.15)[mask],label=name)
        ax[0,1].plot(tt,(np.array(d['thermal_paths']['objects']['B17']['maximum_temperature_k'])-273.15)[mask],label=name)
        ax[1,0].plot(tt,np.array(d['chemistry']['B17']['liquid_kg'])[mask]*1000,label=name)
        ax[1,1].plot(tt,np.array(d['chemistry']['B17']['r2_power_w'])[mask],label=name)
    ex,_=load();mask=(ex['time_s']>=1900)&(ex['time_s']<=2200)
    ax[0,0].plot(ex['time_s'][mask],ex['T4_k'][mask]-273.15,'k--',alpha=.65,label='experiment T4 (mapping unverified)')
    ax[0,0].set_ylim(50,1200);ax[0,0].set_ylabel('B17 mean / T4, C');ax[0,1].set_ylabel('B17 hottest zone, C')
    ax[1,0].set_ylabel('liquid, g');ax[1,1].set_ylabel('R2 power, W');ax[1,1].set_yscale('symlog',linthresh=.1)
    for a in ax.flat:a.grid(alpha=.25);a.set_xlabel('physical time, s')
    ax[0,0].legend(fontsize=8);fig.suptitle('Shared checkpoint 1900 s: conditional interventions, not a fresh calibration')
    target=D/'fsri30-1900-mechanism-investigation-20261027.png';fig.savefig(target,dpi=160);print(target)

if __name__=='__main__':main()
