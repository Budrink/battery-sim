"""Recorded temperatures vs experiment with an explicitly assumed Kelvin tolerance."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    run=args.run.resolve()
    m=json.loads((run/'visual/manifest.json').read_text())
    if m['partial']:raise ValueError('Recording is incomplete')
    t=np.asarray(m['time_s'])
    width=4*len(m['objects'])+m['channel_fields']*len(m['channels'])+m['gas_fields']*len(m['gas'])
    frames=np.memmap(run/'visual/frames.bin',dtype='<f4',mode='r',shape=(len(t),width))
    def temperature(name):return np.asarray(frames[:,4*m['objects'].index(name)],dtype=float)-273.15
    exp=json.loads((run/'experiment.json').read_text())
    et=np.asarray(exp['time_s']);mask=(et>=0)&(et<=t[-1]+1e-6);et=et[mask]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,1,figsize=(13,9),sharex=True)
    fig.subplots_adjust(top=.88,bottom=.13,left=.08,right=.98,hspace=.31)
    fig.suptitle('BatterySim · сравнение с экспериментом',x=.08,ha='left',fontsize=22,fontweight='bold',color='#172b3a')
    fig.text(.08,.923,'Полная сохранённая траектория 0–3000 с · без временного сдвига и сглаживания',color='#536875',fontsize=11)
    panels=[('T5',['B10'],'Первичный разгон · нагреваемая батарея'),
            ('T4',['B17','B18'],'Вторичный разгон · предполагаемое соответствие T4 ↔ B17/B18')]
    colors=['#087f8c','#7058a3']
    for ax,(sensor,cells,title) in zip(axes,panels):
        y=np.asarray(exp[sensor])[mask]
        tol=.05*(y+273.15)
        ax.fill_between(et,y-tol,y+tol,color='#e9ac61',alpha=.30,
            label='Рабочий допуск ±5% от T, K',linewidth=0)
        ax.plot(et,y,color='#b66620',lw=1.3,label=f'Эксперимент · {sensor}')
        for i,cell in enumerate(cells):
            ax.plot(t,temperature(cell),color=colors[i],lw=1.8,ls='-' if i==0 else '--',label=f'Модель · {cell}')
        ax.set_title(title,loc='left',fontsize=13,fontweight='bold',pad=12)
        ax.set_ylabel('Температура, °C');ax.grid(color='#dce4e8',alpha=.7,lw=.6)
        ax.set_xlim(0,3000)
        maximum=max(float(y.max()),*(float(temperature(c).max()) for c in cells))
        ax.set_ylim(-5,max(1200,100*np.ceil((maximum+40)/100)))
        ax.legend(loc='upper left',frameon=True,facecolor='white',edgecolor='#dce4e8',fontsize=10)
    axes[-1].set_xlabel('Время от начала опыта, с')
    fig.text(.08,.057,'Полоса — условный инженерный допуск: ΔT = 0,05 × (Tэксп, °C + 273,15). Не доверительный интервал измерений.',fontsize=10,color='#536875')
    fig.text(.08,.032,'Модель: средние температуры батарей; эксперимент: локальные термопары. Привязка T4 к B17/B18 не подтверждена.',fontsize=10,color='#536875')
    out=run/'comparison-tolerance';out.mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(out/f'comparison.{ext}',dpi=160,facecolor='white')
    plt.close(fig)
    print(out/'comparison.png')


if __name__=='__main__':main()
