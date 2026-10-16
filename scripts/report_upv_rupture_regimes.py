#!/usr/bin/env python3
"""Compare completed training-only regime screens, with signed residuals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from calibrate_upv import save

LABELS = {'101': 'C1 / ISS', '102': 'C1 / TIM', '107': 'C3 / TIM', '108': 'C3 / ISS'}


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('directories', nargs='+', type=Path)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, reports = [], {}
    for directory in args.directories:
        for item in json.loads((directory/'history.json').read_text()):
            if item['id'] == 'seed' and 'seed' in reports:
                continue
            name = item['id']
            if name in reports:
                raise ValueError('Duplicate case id')
            reports[name] = json.loads((directory/(name+'.json')).read_text())
            rows.append(item)
    first = next(iter(reports.values()))
    t = np.asarray(first['result']['time_s'])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    pfig, pax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, report in reports.items():
        run = report['result']
        if not np.allclose(run['time_s'], t, rtol=0, atol=1e-8):
            raise ValueError('Different observation times')
        for ax, (key, label) in zip(axes.flat, LABELS.items()):
            error = np.asarray(run['temperature_k'][key]) - np.asarray(report['observed_temperature_k'][key], dtype=float)
            mask = t >= 300
            ax.plot(t[mask], error[mask], label=name, linewidth=1.3)
            ax.set(title=label, xlabel='Time after event [s]', ylabel='Model minus observation [K]')
        pressure = np.asarray(run['gas']['chamber']['pressure_pa'])/1e5
        for ax in pax:
            ax.plot(t, pressure, label=name, linewidth=1.3)
    for ax in axes.flat:
        ax.axhline(0, color='black', linewidth=.8)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    for ax, end in zip(pax, (6, 30)):
        ax.plot(t, first['observed_pressure_bar'], 'k.-', label='Experiment', linewidth=2, markersize=4)
        ax.set(xlim=(0, end), xlabel='Time after event [s]', ylabel='Chamber pressure [bar abs]')
        ax.grid(alpha=.2)
    pax[1].legend(fontsize=7, ncol=2)
    fig.suptitle('Fixed chemical inventory: signed late temperature errors (training only)')
    pfig.suptitle('Pressure on original acquisition times; no fitted time shift')
    fig.savefig(args.output/'regime-temperature-errors.png', dpi=150)
    pfig.savefig(args.output/'regime-pressure.png', dpi=150)
    save(args.output/'comparison.json', rows)
    for row in rows:
        m = row['metrics']
        print(row['id'], 'T_RMSE', round(m['witness']['rmse_k'], 3),
              'cost', round(m['objective_sum_squares'], 3),
              'p_early_RMSE', round(m['pressure_early_rmse_bar'], 3),
              'peak', round(m['predicted_peak_bar'], 3),
              'bias', {k: round(v['bias_k'], 2) for k, v in m['witness']['per_channel'].items()})


if __name__ == '__main__':
    main()
