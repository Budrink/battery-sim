"""Plot observed control and conditioned one-factor B17 interventions."""
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fsri30_data import load

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/development'
RUNS=[('Control',DATA/'fsri30-valve80kpa-only-20261027-r1/cascade/replay/incomplete-run.json.gz'),
      ('Hot gas response reduced',DATA/'fsri30-b17-hot-film-half-20261027-r1/replay/run.json.gz'),
      ('Evaporation relaxation x10',DATA/'fsri30-b17-evap-x10-20261027-r1/replay/run.json.gz'),
      ('Internal conductivity x2',DATA/'fsri30-b17-internal-x2-20261027-r1/replay/run.json.gz')]

def main():
    fig,ax=plt.subplots(2,2,figsize=(15,9),sharex=True,constrained_layout=True)
    for label,path in RUNS:
        if not path.exists():continue
        with gzip.open(path,'rt') as f:d=json.load(f)
        t=np.array(d['time_s']);idx=(t>=1600)&(t<=2000);tt=t[idx]
        mean=np.array(d['cell_mean_temperature_k']['B17'])-273.15
        local=np.array(d['thermal_paths']['objects']['B17']['maximum_temperature_k'])-273.15
        ax[0,0].plot(tt,mean[idx],label=label);ax[0,1].plot(tt,local[idx],label=label)
        ax[1,0].plot(tt,np.array(d['chemistry']['B17']['liquid_kg'])[idx]*1000,label=label)
        ax[1,1].plot(tt,np.array(d['chemistry']['B17']['r2_power_w'])[idx],label=label)
    exp,_=load();idx=(exp['time_s']>=1600)&(exp['time_s']<=2000)
    ax[0,0].plot(exp['time_s'][idx],exp['T4_k'][idx]-273.15,'k--',label='T4 (mapping unverified)',alpha=.7)
    ax[0,0].set_ylabel('B17 mean / T4, C');ax[0,1].set_ylabel('B17 hottest zone, C')
    ax[1,0].set_ylabel('B17 liquid mass, g');ax[1,1].set_ylabel('B17 R2 power, W');ax[1,1].set_yscale('symlog',linthresh=.1)
    for a in ax.flat:a.grid(alpha=.25);a.set_xlabel('Time, s')
    ax[0,0].legend(fontsize=8);fig.suptitle('B17: conditioned local diagnostic from the same state at 1600 s')
    target=DATA/'fsri30-b17-mechanism-comparison-20261027.png';fig.savefig(target,dpi=160);print(target)

if __name__=='__main__':main()
