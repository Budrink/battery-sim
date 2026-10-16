#!/usr/bin/env python3
"""Report numerical gas timestep trials; no experimental fit or validation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_fixed_gas_network import ROOT, summarize
from calibrate_upv import save


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args();root=args.folder
    coarse=read(root/'strict-short/cap-0.json')
    fine=read(root/'strict-short/cap-1.json')
    tighter=read(root/'strict-half/cap-0.json')
    relaxed=read(root/'relaxed-long/cap-0.json')
    old_root=ROOT/'data/calibration/upv-fixed-gas-audit-v32'
    old_fine=read(old_root/'snapshot-finest/cap-0.json')
    old_coarse=read(old_root/'snapshot/cap-0.json')
    # Only numerical pressure controls differ across these new trials.
    physical=None
    for folder in ('strict-short','strict-half','relaxed-long'):
        network=read(root/folder/'scenario.json')['gas_network']
        for key in ('pressure_relaxation_fraction','pressure_absolute_tolerance_pa','pressure_relative_tolerance'):
            network.pop(key,None)
        if physical is not None and physical!=network:
            raise ValueError('Different physical initial states/topologies')
        physical=network
    node='void-6-3-1'
    chain=[read(p) for p in sorted((root/'chain').glob('cap-*.json'))]
    result=dict(scope='Fixed gas-only network; not a full reactive/multiphase replay or calibration',
        calibration_accepted=False,reference_is_exact=False,
        short_external_cap_comparison=summarize(coarse,fine),
        short_pressure_control_comparison=summarize(coarse,tighter),
        short_fine_vs_tighter=summarize(fine,tighter),
        chain_comparisons=[summarize(r,chain[-1]) for r in chain],
        long_observation=dict(duration_s=relaxed['duration_s'],steps=relaxed['steps'],
            node=node,old_coarse_pressure_bar=old_coarse['gas'][node]['pressure_pa'][-1]/1e5,
            new_pressure_bar=relaxed['gas'][node]['pressure_pa'][-1]/1e5,
            old_finest_pressure_bar=old_fine['gas'][node]['pressure_pa'][-1]/1e5,
            comparison_to_old_finest=summarize(relaxed,old_fine),
            warning='Different numerical cap formula and looser pressure tolerance; old finest is NOT an exact reference. '
                    'Do not infer temperature/route convergence from close pressures.'),
        limitations=['Only 30 microseconds tested with strict controls.',
            'Six closed paths omitted from the original multiphase graph as in V32.',
            'Shared global timestep remains expensive for stiff heterogeneous networks.',
            'Wall timings overlap other tests and are not a controlled performance benchmark.'])
    save(root/'summary.json',result)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    time=np.array(coarse['time_s'])*1e6
    for run,label in ((coarse,'Обычный допуск'),(tighter,'Допуски вдвое строже')):
        for ax,key,title,unit in zip(axes,('pressure_pa','temperature_k'),
                ('Максимальное расхождение давления','Максимальное расхождение температуры'),('Па','K')):
            delta=np.array([np.abs(np.array(run['gas'][k][key])-np.array(fine['gas'][k][key])) for k in fine['gas']])
            ax.plot(time,delta.max(axis=0),'.-',label=label)
            ax.set(xlabel='Время после снимка, мкс',ylabel=unit,title=title)
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Фиксированная газовая сеть: сравнение с внешним шагом ≤ 0,5 нс\nЧисленная проверка, не калибровка эксперимента',fontsize=11)
    fig.savefig(root/'adaptive-errors.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
