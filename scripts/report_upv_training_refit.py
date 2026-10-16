#!/usr/bin/env python3
"""Plot a frozen training refit and all declared cooling cases, without ranking."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from calibrate_upv import save
from calibrate_upv_failure import load_record

LABELS = {'101': 'C1 / ISS', '102': 'C1 / TIM', '107': 'C3 / TIM', '108': 'C3 / ISS'}


def read(path):
    return json.loads(path.read_text())


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('output', type=Path)
    out = cli.parse_args().output
    training = read(out/'fit.json'); baseline = read(out/'baseline.json')
    summary = read(out/'cooling-check/summary.json')
    if len(summary['cases']) != 3 or summary['selected_case'] is not None:
        raise ValueError('Expected the complete unranked cooling sensitivity series')
    reports = [read(out/'cooling-check'/case['id']/'report.json') for case in summary['cases']]
    colors = ['#e69f00', '#0072b2', '#009e73']
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), constrained_layout=True)
    pressure, paxes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for column, cases in enumerate([[training], reports]):
        first = cases[0]; t = np.asarray(first['result']['time_s']); late = t >= 300
        title = 'No cooling: training' if column == 0 else 'Active cooling: held-out sensitivity'
        for row, (key, label) in enumerate(LABELS.items()):
            ax = axes[row, column]
            obs = np.asarray(first['observed_temperature_k'][key], dtype=float)
            ax.plot(t[late], obs[late]-273.15, color='#20252c', lw=2, label='Experiment')
            if column == 0:
                ax.plot(t[late], np.asarray(baseline['result']['temperature_k'][key])[late]-273.15,
                        color='#999999', ls='--', label='Previous parameters')
            for i, report in enumerate(cases):
                name = 'Refit' if column == 0 else f"G = {report['cooling_priors']['sink_conductance_w_per_k']:g} W/K"
                ax.plot(t[late], np.asarray(report['result']['temperature_k'][key])[late]-273.15,
                        color=colors[i], label=name)
            ax.set(title=title+' — '+label, xlabel='Time after event [s]', ylabel='Temperature [°C]')
            ax.grid(alpha=.2)
        axes[0, column].legend(fontsize=8)
        for row, xmax in enumerate([5, 1800]):
            ax = paxes[row, column]
            ax.plot(t, first['observed_pressure_bar'], color='#20252c', lw=2, label='Experiment')
            for i, report in enumerate(cases):
                name = 'Refit' if column == 0 else f"G = {report['cooling_priors']['sink_conductance_w_per_k']:g} W/K"
                ax.plot(t, np.asarray(report['result']['gas']['chamber']['pressure_pa'])/1e5,
                        color=colors[i], label=name)
            ax.set(xlim=(0, xmax), title=title, xlabel='Time after event [s]', ylabel='Chamber pressure [bar abs]')
            ax.grid(alpha=.2)
        paxes[0, column].legend(fontsize=8)
    fig.suptitle('Frozen training profile; cooling cases are assumptions, not confidence bounds')
    fig.savefig(out/'witness-comparison.png', dpi=150); plt.close(fig)
    pressure.savefig(out/'pressure-comparison.png', dpi=150); plt.close(pressure)
    # This observation is never part of the training objective or a prescribed
    # contact temperature. Its side within the physical contact jump is unclear.
    data, _ = load_record('active-cooling')
    contact_fig, contact_ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    diagnostics = {}
    for color, case in zip(colors, summary['cases']):
        warm = read(out/'cooling-check'/case['id']/'warmup.json')
        at = np.asarray(warm['time_s']); actual = np.asarray(warm['temperature_k']['110'])
        observed = np.interp(at, data['time_s'], data['T110_k'])
        error = actual[at>=60]-observed[at>=60]
        diagnostics[case['id']] = dict(model_final_c=float(actual[-1]-273.15),
            observed_final_c=float(observed[-1]-273.15), rmse_k=float(np.sqrt(np.mean(error**2))),
            bias_k=float(error.mean()),
            caveat='Modeled Al2-side surface; exact experimental side of contact jump unconfirmed')
        contact_ax.plot(at, actual-273.15, color=color,
                        label=f"G = {case['cooling_priors']['sink_conductance_w_per_k']:g} W/K")
    contact_ax.plot(at, observed-273.15, color='#20252c', lw=2, label='Thermocouple 110')
    contact_ax.set(title='Held-out contact diagnostic — not a forcing or fitting target',
                   xlabel='Time from start [s]', ylabel='Temperature [°C]')
    contact_ax.grid(alpha=.2); contact_ax.legend()
    contact_fig.savefig(out/'contact-comparison.png', dpi=150); plt.close(contact_fig)
    save(out/'assessment.json', dict(training=training['metrics'],
         training_baseline=baseline['metrics'], heating=read(out/'heat/fit.json')['metrics'],
         heating_numerics=read(out/'heat/numerics-effective.json'),
         numerics=read(out/'numerics.json')['changes'], cooling=summary, contact_110=diagnostics,
         balances={name: {k: v for k, v in report['result'].items() if 'residual' in k}
                   for name, report in [('training', training)]+[(r['cooling_priors']['id'], r) for r in reports]},
         accepted=False, limitation='Conditional event fit; unknown cooling UA/contact/footprint. No held-out parameter selection.'))


if __name__ == '__main__':
    main()
