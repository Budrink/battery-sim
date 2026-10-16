#!/usr/bin/env python3
"""Compare frozen thermal / destruction-only candidates on identical samples."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import upv_mass_paths as model


def plot_case(folder, name, report, prior, scene):
    fig, axes = plt.subplots(5, 3, figsize=(15, 15), constrained_layout=True)
    times = np.asarray(report['result']['time_s'])
    old_times = np.asarray(prior['result']['time_s'])
    # The retained sensor mapping reverses the C3 face order. Derive labels
    # from actual probe bindings instead of assuming identical channel order.
    rows = [(p['id'], p['cell_id'].split(':')[0]+' / '+p['neighbor_material']+' (ch'+p['id']+')')
            for p in scene['measurement_probes'] if p['id'] in ('101', '102', '107', '108')]
    for row, (channel, label) in enumerate(rows):
        observed = np.asarray(prior['observed_temperature_k'][channel], dtype=float)-273.15
        old = np.asarray(prior['result']['temperature_k'][channel])-273.15
        new = np.asarray(report['result']['temperature_k'][channel])-273.15
        for col, (lo, hi) in enumerate([(0, 3), (3, 300), (300, 1800)]):
            ax = axes[row, col]; mask = (times >= lo)&(times <= hi); before = (old_times >= lo)&(old_times <= hi)
            ax.plot(old_times[before], observed[before], color='#252525', lw=1.6, label='Experiment')
            ax.plot(old_times[before], old[before], color='#969696', ls='--', lw=1.3, label='Previous heat-only aperture')
            ax.plot(times[mask], new[mask], color='#d95f02', lw=1.6, label='Explicit mass paths')
            ax.set(xlim=(lo, hi), ylabel=label+' [°C]');ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(f'{lo}–{hi} s'+(' (diagnostic only)' if col == 0 else ''))
    for col, (lo, hi) in enumerate([(0, 3), (3, 300), (300, 1800)]):
        ax = axes[4, col];mask = (times>=lo)&(times<=hi);before = (old_times>=lo)&(old_times<=hi)
        ax.plot(old_times[before], np.asarray(prior['observed_pressure_bar'])[before], color='#252525', lw=1.6)
        ax.plot(old_times[before], np.asarray(prior['result']['gas']['chamber']['pressure_pa'])[before]/1e5, color='#969696', ls='--')
        ax.plot(times[mask], np.asarray(report['result']['gas']['chamber']['pressure_pa'])[mask]/1e5, color='#d95f02')
        ax.set(xlim=(lo, hi), ylabel='Chamber pressure [bar abs]', xlabel='Time from event [s]');ax.grid(alpha=.2)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(name+' — destruction replay; thermal parameters frozen\nSensor-numbering hypothesis retained; early gas-jet sensor response unresolved', fontsize=13)
    fig.savefig(folder/(name+'.png'), dpi=140);fig.savefig(folder/(name+'.svg'));plt.close(fig)


def route_summary(report, scene):
    result = report['result'];times = np.asarray(result['time_s']);rows = []
    for path in scene['gas_network']['channels']:
        if path['kind'] != 'rupture':
            continue
        trace = result['channels'][path['id']];opened = np.flatnonzero(np.asarray(trace['opening_fraction']) > .01)
        rows.append(dict(id=path['id'], source=path['from'], destination=path['to'],
                         first_sample_above_one_percent_s=float(times[opened[0]]) if len(opened) else None,
                         net_transferred_mass_kg=trace['transferred_mass_kg'][-1],
                         net_transferred_chemical_j=trace['transferred_chemical_j'][-1],
                         peak_absolute_flow_kg_s=float(np.max(np.abs(trace['mass_flow_kg_s']))),
                         final_open_fraction=trace['opening_fraction'][-1],
                         fracture_work_j=trace['fracture_work_j'][-1]))
    return rows


def plot_process(folder, name, report, scene):
    run = report['result'];t = np.asarray(run['time_s']);use = t<=30
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].plot(t[use], np.asarray(run['cell_mean_temperature_k']['C2'])[use]-273.15, label='Mean')
    axes[0, 0].plot(t[use], np.asarray(run['source_max_temperature_k'])[use]-273.15, label='Maximum voxel')
    axes[0, 0].set_ylabel('C2 solid temperature [°C]')
    internal = [v['pressure_pa'] for k,v in run['gas'].items() if k.startswith('inside-C2')]
    axes[0, 2].plot(t[use], np.max(internal, axis=0)[use]/1e5, label='Maximum source pore')
    axes[0, 2].plot(t[use], np.asarray(run['gas']['chamber']['pressure_pa'])[use]/1e5, label='Chamber')
    axes[0, 2].set_ylabel('Sampled pressure [bar abs]')
    for path in scene['gas_network']['channels']:
        if path['kind'] != 'rupture':
            continue
        trace = run['channels'][path['id']];label=path['port_id'].split(':')[2]
        axes[0, 1].plot(t[use], np.asarray(trace['transferred_mass_kg'])[use]*1000, label=label)
        axes[1, 0].plot(t[use], np.asarray(trace['opening_fraction'])[use], label=label)
    axes[0, 1].set_ylabel('Net mass through rupture [g]');axes[1, 0].set_ylabel('Opened casing fraction')
    for prefix in ('gap-ISS', 'gap-TIM', 'near', 'chamber'):
        heat = np.sum([v['released_energy_j'] for k,v in run['gas'].items() if k.startswith(prefix)], axis=0)
        axes[1, 1].plot(t[use], heat[use]/1000, label=prefix)
    axes[1, 1].set_ylabel('Gas reaction heat [kJ]')
    axes[1, 2].plot(t[use], np.asarray(run['remaining_energy_j'])[use]/1000, label='Remaining in source')
    axes[1, 2].set_ylabel('Chemical reserve [kJ]')
    for ax in axes.flat:
        ax.set(xlim=(0, 30), xlabel='Time from event [s]');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(name+' — destruction process and actual mass routes (sampled diagnostics)')
    fig.savefig(folder/(name+'-process.png'), dpi=140);fig.savefig(folder/(name+'-process.svg'));plt.close(fig)


def diagram(folder):
    fig, ax = plt.subplots(figsize=(12, 7));ax.set(xlim=(0, 12), ylim=(0, 7));ax.axis('off')
    boxes = {
        'core': (1.5, 3.5, 'C2 contents\n27 local gas / solid volumes'),
        'iss': (5, 5.6, 'Opened broad face\nISS2 interface gap'),
        'tim': (5, 3.5, 'Opened broad face\nTIM2 interface gap'),
        'edge': (5, 1.4, 'Opened edge seams\nNo intact-pad crossing'),
        'near': (8, 3.5, 'Four perimeter gas regions\n±y, ±z'),
        'neighbours': (9, 5.7, 'C1 / C3 surface gaps\nBoth ISS and TIM sides'),
        'chamber': (10.8, 3.5, 'Sealed chamber\n120 L total')}
    labels = {}
    for name, (x, y, label) in boxes.items():
        labels[name] = ax.text(x, y, label, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=.6', fc='#eaf2f8' if name=='core' else '#f5f5f5', ec='#426a80'))
    fig.canvas.draw()
    for a, b in [('core', 'iss'), ('core', 'tim'), ('core', 'edge'), ('iss', 'near'), ('tim', 'near'), ('edge', 'near'), ('near', 'chamber'), ('near', 'neighbours')]:
        x1,y1,_=boxes[a];x2,y2,_=boxes[b]
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='<->' if b=='neighbours' else '->', color='#d95f02', lw=2,
                    patchA=labels[a].get_bbox_patch(), patchB=labels[b].get_bbox_patch(), shrinkA=6, shrinkB=6))
    ax.text(6, .25, 'Arrows: dominant escape routes, not one-way valves. Pressure reversal permits backflow.\nWith cooling plate: bottom gas exits around lateral plate edges; no path through metal.\nGap dimensions / rupture locations are unmeasured hypotheses. No outlet from the sealed chamber.', ha='center', fontsize=10)
    ax.set_title('Reduced mass-transfer topology — thermal graph unchanged', fontsize=14)
    fig.savefig(folder/'gas-paths.svg', bbox_inches='tight');fig.savefig(folder/'gas-paths.png', dpi=150, bbox_inches='tight');plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=model.ROOT/'data/calibration/upv-mass-paths-v15')
    args = parser.parse_args();folder=args.output;summary={}
    for name, stem in [('no-cooling', 'fit'), ('active-cooling', 'validation')]:
        if not (folder/(stem+'.json')).exists():
            continue
        report = model.read(folder/(stem+'.json'));scene=model.read(folder/(stem+'-scenario.json'))
        _, prior = model.inputs(name);plot_case(folder, name, report, prior, scene);plot_process(folder, name, report, scene)
        summary[name] = dict(metrics=report['metrics'], routes=route_summary(report, scene),
                             thermal_unchanged=model.thermal_signature(scene)==model.thermal_signature(model.inputs(name)[0]),
                             steps=report['result']['steps'], wall_s=report['result']['wall_s'],
                             conservation={k:v for k,v in report['result'].items() if 'residual' in k})
        summary[name]['gas_reaction_heat_kj'] = {
            prefix:sum(v['released_energy_j'][-1] for key,v in report['result']['gas'].items()
                       if key.startswith(prefix))/1000
            for prefix in ('inside', 'gap-ISS', 'gap-TIM', 'near', 'chamber')}
    diagram(folder)
    (folder/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
