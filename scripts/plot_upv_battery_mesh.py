#!/usr/bin/env python3
"""Compare frozen battery depth grids; do not choose a grid by holdout fit."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from generate_upv import ROOT, load_case


def iss_to_tim_sign(scene,cell):
    """Bind the sign by physical ports, not battery numbering or list order."""
    ports={p['id']:p for p in scene['ports']};objects={c['id']:c for c in scene['cells']}
    faces={}
    for edge in scene['interfaces']:
        a,b=ports[edge['port_a']],ports[edge['port_b']]
        if a['cell_id']==cell:own,other=a,b
        elif b['cell_id']==cell:own,other=b,a
        else:continue
        material=objects[other['cell_id']]['material_id']
        if material in ('ISS','TIM'):
            if material in faces:raise ValueError('Ambiguous material face')
            faces[material]=own['surface']
    if faces=={'ISS':'-x','TIM':'+x'}:return 1.
    if faces=={'ISS':'+x','TIM':'-x'}:return -1.
    raise ValueError('Expected opposite ISS/TIM depth faces')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    summary=json.loads((a.directory/'summary.json').read_text())
    colors={'nx2':'#cc6928','nx4':'#209d83','nx8':'#3468cc','nx16':'#7f56ad','isothermal':'#ba357d'}
    styles={'nx2':'--','nx4':'-.','nx8':'-','nx16':':','isothermal':':'}
    labels={'nx2':'2 depth slabs','nx4':'4 depth slabs','nx8':'8 depth slabs','nx16':'16 depth slabs','isothermal':'Isothermal battery'}
    for experiment,s in summary['experiments'].items():
        cases={name:json.loads((a.directory/f'{experiment}-{name}.json').read_text()) for name in s['cases']}
        scene_path=Path(cases['nx2']['scenario_path'])
        if not scene_path.is_absolute():scene_path=ROOT/scene_path
        scene=json.loads(scene_path.read_text())
        observed,*_=load_case(ROOT/'data/experiments/upv',experiment)
        times=np.array(cases['nx2']['result']['time_s'])
        obs={c:np.interp(times,observed['time_s'],observed['T'+c+'_k']) for c in ('101','102','107','108')}
        fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,constrained_layout=True)
        for ax,channel,title in zip(axes.flat,('101','102','107','108'),('C1 ISS face','C1 TIM face','C3 TIM face','C3 ISS face')):
            ax.plot(times,obs[channel]-273.15,color='#151515',label='Measured',lw=2.1)
            for name,r in cases.items():
                ax.plot(times,np.array(r['result']['temperature_k'][channel])-273.15,color=colors[name],ls=styles[name],label=labels[name],lw=1.5)
            ax.set(title=title,xlabel='Time, s',ylabel='Temperature, °C');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'UPV {experiment}: frozen contacts and materials, battery-depth sensitivity\nOnly x subdivisions change; isothermal is a separate physical closure. C1/C3 numbering unconfirmed.',fontsize=11)
        fig.savefig(a.directory/f'{experiment}-temperatures.png',dpi=150);plt.close(fig)

        fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,constrained_layout=True)
        for col,(cell,iss,tim) in enumerate((('C1','101','102'),('C3','108','107'))):
            ax=axes[0,col];ax.plot(times,obs[iss]-obs[tim],color='#151515',label='Measured',lw=2.1)
            for name,r in cases.items():
                run=r['result'];depth=r['depth_profile']['objects'][cell]
                ax.plot(times,np.array(run['temperature_k'][iss])-run['temperature_k'][tim],color=colors[name],ls=styles[name],label=labels[name])
                if name!='isothermal':
                    sign=iss_to_tim_sign(scene,cell)
                    axes[1,col].plot(times,sign*np.array(depth['midplane_left_to_right_power_w']),color=colors[name],ls=styles[name],label=labels[name])
            ax.set(title=f'{cell}: surface difference, ISS minus TIM',ylabel='Temperature difference, K');ax.grid(alpha=.2);ax.legend(fontsize=8)
            axes[1,col].set(title=f'{cell}: internal midplane power, positive ISS → TIM',xlabel='Time, s',ylabel='Power, W')
            axes[1,col].grid(alpha=.2);axes[1,col].legend(fontsize=8)
        fig.suptitle(f'UPV {experiment}: surface gradients and resolved internal heat flow\nIsothermal redistributes energy algebraically: no finite-gradient midplane flux is plotted.',fontsize=11)
        fig.savefig(a.directory/f'{experiment}-gradients-flow.png',dpi=150);plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for ax,cell in zip(axes,('C1','C3')):
            for name,r in cases.items():
                depth=r['depth_profile']['objects'][cell]
                x=1000*np.array(depth['slab_center_x_m'])
                ax.plot(x,np.array(depth['slab_temperature_k'][-1])-273.15,marker='o',ms=3,color=colors[name],ls=styles[name],label=labels[name])
            ax.set(title=cell,xlabel='Local depth x, mm',ylabel='Slab mean temperature, °C');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'Final depth profiles at t={times[-1]:.0f} s, {experiment}; points are volume centers, not surfaces',fontsize=11)
        fig.savefig(a.directory/f'{experiment}-depth-profiles.png',dpi=150);plt.close(fig)
    print(a.directory)


if __name__=='__main__':main()
