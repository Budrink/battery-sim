#!/usr/bin/env python3
"""Export the actual passive-wall aperture support, before any dynamic opening."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('scenario',type=Path);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();s=json.loads(args.scenario.read_text());meta=s['metadata']['abs_barrier_damage']
    available={(p['wall'],p['thermal_volume'])for p in meta['connected']}
    walls=[c for c in s['cells']if c['id'].startswith('ABS:')]
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    for ax,wall in zip(axes.flat,walls):
        face=wall['id'].split(':')[-1];axis='xyz'.index(face[1]);u,v=(axis+1)%3,(axis+2)%3
        n=wall['thermal']['divisions'];d=wall['geometry']['dimensions_m']
        pixels=np.zeros((n[v],n[u]))
        for iv in range(n[v]):
            for iu in range(n[u]):
                pos=[0,0,0];pos[u]=iu;pos[v]=iv
                local=pos[0]+n[0]*(pos[1]+n[1]*pos[2])
                pixels[iv,iu]=(wall['id'],local)in available
        ax.imshow(pixels,origin='lower',extent=(-d[u]*500,d[u]*500,-d[v]*500,d[v]*500),
            interpolation='nearest',vmin=0,vmax=1,cmap=ListedColormap(['#d7dce2','#159b8a']))
        ax.set(title=f"{wall['id']} · {int(pixels.sum())} участков",
            xlabel='xyz'[u]+', мм',ylabel='xyz'[v]+', мм')
    fraction=meta['connected_area_m2']/meta['total_abs_face_area_m2']
    fig.suptitle(f'ABS: где разрешён локальный прорыв · {100*fraction:.1f}% площади\n'
        'Все пути изначально закрыты; раскрытие рассчитывается по давлению и повреждению',fontsize=13)
    fig.legend(handles=[Patch(color='#159b8a',label='Газовые соседи и вся площадь определены'),
        Patch(color='#d7dce2',label='Смешанная/частично открытая площадка: пока закрыта')],
        loc='outside lower center',ncol=2,fontsize=9)
    args.output.mkdir(parents=True,exist_ok=False)
    for suffix in ('png','pdf','svg'):fig.savefig(args.output/f'abs-geometry.{suffix}',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
