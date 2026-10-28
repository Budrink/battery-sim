#!/usr/bin/env python3
"""Plot an explicit saved trial against immutable no-cooling measurements.

Observation only: no simulator, fitted shift, gain, smoothing or extrapolation.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import upv_r123_data as dataio
from calibrate_upv_r123 import save


CHANNELS = [('101','C1 / ISS'),('102','C1 / TIM'),
            ('C2_ISS','C2 / ISS'),('C2_TIM','C2 / TIM'),
            ('108','C3 / ISS'),('107','C3 / TIM')]
EXPERIMENT = '#333333'
SIMULATION = '#0072B2'


def temperatures(out,run,data,window,title,filename):
    left,right=window
    observed=dataio.observed(data,data['time_s'])
    measured=(data['time_s']>=left)&(data['time_s']<=right)
    t=np.asarray(run['time_s']);known=(t>=left)&(t<=right)
    fig,axes=plt.subplots(3,2,figsize=(13,9),constrained_layout=True)
    for ax,(key,label)in zip(axes.flat,CHANNELS):
        # Draw the original scans, including NaNs: do not bridge missing data.
        ax.plot(data['time_s'][measured],observed[key][measured]-273.15,
                color=EXPERIMENT,lw=1.5,label='Эксперимент')
        ax.plot(t[known],np.asarray(run['temperature_k'][key])[known]-273.15,
                color=SIMULATION,lw=1.8,label='Расчёт: последний прогон r5')
        ax.set(title=label,xlabel='Время от начала опыта, с',ylabel='Температура, °C',xlim=window)
        ax.grid(alpha=.23)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=2)if matplotlib.__version__>='3.7'else axes[0,0].legend(fontsize=9)
    fig.suptitle(title+'\nКалибровочный опыт без водяного охлаждения; без сдвига времени',fontsize=13)
    fig.savefig(out/filename,dpi=160)
    plt.close(fig)


def pressure(out,run,data,end):
    t=np.asarray(run['time_s']);p=np.asarray(dataio.chamber_pressure_pa(run))/1e5
    fig,axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
    for ax,window,title in zip(axes,[(880.,910.),(1187.,end)],['Выброс','Позднее охлаждение']):
        a=(data['time_s']>=window[0])&(data['time_s']<=window[1]);b=(t>=window[0])&(t<=window[1])
        ax.plot(data['time_s'][a],2.5*data['U114_v'][a],color=EXPERIMENT,lw=1.5,label='Эксперимент')
        ax.plot(t[b],p[b],color=SIMULATION,lw=1.8,label='Расчёт: последний прогон r5')
        ax.set(title=title,xlabel='Время от начала опыта, с',ylabel='Абсолютное давление, бар',xlim=window)
        ax.grid(alpha=.23);ax.legend(fontsize=9)
    fig.suptitle('Давление в камере, не внутри батареи')
    fig.savefig(out/'pressure.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();trial=args.trial.resolve();out=args.output.resolve();out.mkdir(exist_ok=False)
    path=trial/'replay/run.json.gz'
    with gzip.open(path,'rt')as stream:run=json.load(stream)
    if run.get('partial'):raise ValueError('Requested trial is incomplete')
    data,record=dataio.load('no-cooling');end=float(run['time_s'][-1])
    if end<1500.-1e-7:raise ValueError('This review requires the completed 1500 s trial')
    save(out/'provenance.json',dict(trial=str(trial),run=str(path),
        run_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scenario_sha256=hashlib.sha256((trial/'scenario.json').read_bytes()).hexdigest(),
        experimental_sha256=record['canonical_sha256'],end_s=end,
        parameters=json.loads((trial/'parameters.json').read_text()),
        observations='Raw experimental scans; C2 channels are means 105/106 and 103/104, matching the existing comparison. Missing scans remain missing.',
        simulation_rerun=False,fitted_time_shift_s=0.,temperature_gain=1.,smoothing=False,extrapolation=False))
    temperatures(out,run,data,(880.,915.),'Выброс: резкость и высота пиков','burst.png')
    temperatures(out,run,data,(900.,end),'После выброса: вся рассчитанная часть охлаждения','cooling.png')
    temperatures(out,run,data,(1187.,end),'Хвост охлаждения: увеличенный масштаб','tail.png')
    pressure(out,run,data,end)
    print(json.dumps(dict(output=str(out),run=str(path),end_s=end,rerun=False)),flush=True)


if __name__=='__main__':main()
