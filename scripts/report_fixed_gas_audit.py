#!/usr/bin/env python3
"""Collect fixed-network diagnostics and plot the same physical initial state."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calibrate_upv import save
from audit_fixed_gas_network import summarize


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('folder',type=Path);args=p.parse_args();root=args.folder
    def read(path):return json.loads(path.read_text())
    chain=[read(f) for f in sorted((root/'chain').glob('cap-*.json'))]
    levels={};physical=None
    for folder in ('snapshot','snapshot-fine','snapshot-finest'):
        scene=read(root/folder/'scenario.json');current=scene['gas_network']
        if physical is not None and current!=physical:raise ValueError('Different gas initial states/topologies')
        physical=current
        for f in sorted((root/folder).glob('cap-*.json')):
            r=read(f);levels[r['max_dt_s']]=r
    runs=[levels[k] for k in sorted(levels,reverse=True)];reference=runs[-1]
    nodes=['void-6-3-1','gap-TIM2-3','chamber']
    rows=[dict(summarize(r,reference),final_pressure_bar={k:r['gas'][k]['pressure_pa'][-1]/1e5 for k in nodes}) for r in runs]
    paths=sorted(runs[0]['channels'].items(),key=lambda kv:kv[1]['proposed_abs_mass_kg']-kv[1]['accepted_abs_mass_kg'],reverse=True)
    clipping=[dict(id=k,proposed_abs_mass_kg=v['proposed_abs_mass_kg'],accepted_abs_mass_kg=v['accepted_abs_mass_kg'],
                   accepted_over_proposed=v['accepted_abs_mass_kg']/v['proposed_abs_mass_kg']) for k,v in paths[:5]]
    result=dict(scope='Numerical diagnosis, no calibration; gas-only ablation is not a full multiphase restart',
        source=read(root/'snapshot/scenario.json')['metadata'],
        nodes=len(physical['compartments']),open_paths=len(physical['channels']),
        reference_dt_s=reference['max_dt_s'],reference_is_exact=False,rows=rows,
        chain=[dict(max_dt_s=r['max_dt_s'],steps=r['steps'],
            pressure_pa={k:v['pressure_pa'][-1] for k,v in r['gas'].items()},
            far_path_transferred_kg=r['channels']['path1']['transferred_mass_kg'][-1]) for r in chain],
        coarse_clipping_examples=clipping,
        warning='Clipped proposals are not lost mass; they are frozen-state demands, not realizable integrated flows')
    save(root/'summary.json',result)
    fig,(left,right)=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=np.arange(3);width=.24
    for j,k in enumerate((0,1,4)):
        r=chain[k];pressures=[r['gas'][f'v{i}']['pressure_pa'][-1]-100000 for i in range(3)]
        left.bar(x+(j-1)*width,pressures,width,label=f"dt ≤ {r['max_dt_s']*1e6:g} мкс")
    left.axhline(100/3,color='black',linestyle=':',label='Равновесный уровень')
    left.set(xticks=x,xticklabels=['Полость 1','Полость 2','Полость 3'],ylabel='Избыток над 1 бар, Па',
             title='Три полости: одно время 20 мкс')
    left.legend(fontsize=8);left.grid(axis='y',alpha=.2)
    for r in (runs[0],runs[1],levels[7.8125e-7],reference):
        right.plot(np.asarray(r['time_s'])*1e3,np.asarray(r['gas'][nodes[0]]['pressure_pa'])/1e5,
                   '.-',label=f"dt ≤ {r['max_dt_s']*1e6:g} мкс")
    right.set(xlabel='Время после общего снимка, мс',ylabel='Давление, бар',
              title='UPV: одна внешняя полость, отверстия неизменны')
    right.legend(fontsize=8);right.grid(alpha=.2)
    fig.suptitle('Только газовый перенос: без разрушения, реакций и теплообмена с твёрдыми телами',fontsize=11)
    fig.savefig(root/'fixed-network.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(reference_dt_s=result['reference_dt_s'],rows=rows),indent=2))


if __name__=='__main__':main()
