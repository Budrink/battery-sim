#!/usr/bin/env python3
"""Deliberately nonphysical, LOCAL C3/ISS transport saturation diagnostic.

Never promotes the extreme scenario to calibration. Pressure conductance and
counterexchange are boosted on existing passages only; no shell is punctured,
no route is added, and no mass, oxygen or thermal energy is injected.
"""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal

import numpy as np
import calibrate_upv_r123 as cal
import upv_r123_data as dataio
from check_upv_monotone_gas import plot
from check_upv_restart import native
from upv_gas_heat_budget import inspect


def local_maximum(original, heat_factor=1000., gas_factor=1000.):
    if not (1. <= heat_factor <= 1e6 and 1. <= gas_factor <= 1e6):
        raise ValueError('Diagnostic multipliers must be finite and >= 1')
    scene = copy.deepcopy(original)
    network = scene['gas_network']
    gaps = {n['id'] for n in network['compartments'] if n['id'].startswith('gap-ISS1-')}
    contacts = [c for c in network['thermal_contacts'] if c['compartment'] in gaps
                or c['port_id'].startswith('opening-contact:ISS1:')]
    galleries = {c['compartment'] for c in contacts} - gaps
    region = gaps | galleries
    if not gaps or not galleries:
        raise ValueError('Missing C3/ISS slit or its immediate inlet galleries')
    manifest = dict(diagnostic_only=True, calibrated=False, gaps=sorted(gaps),
                    galleries=sorted(galleries), gas_factor=gas_factor,
                    heat_factor=heat_factor, channels=[], thermal_contacts=[])
    for edge in network['channels']:
        # Include the immediate supply to the gallery, not just the final slit
        # inlet. Ruptures/internal pores and every other pad stay untouched.
        if edge['kind'] != 'passage' or not ({edge['from'], edge['to']} & region):
            continue
        before = copy.deepcopy(edge)
        edge['area_m2'] *= gas_factor
        edge['mixing_length_m'] /= gas_factor
        edge['discharge_coefficient'] = 1.
        manifest['channels'].append(dict(id=edge['id'], before=before, after=copy.deepcopy(edge)))
    for contact in contacts:
        before = copy.deepcopy(contact)
        for key in ('background_conductance_w_per_k', 'active_conductance_w_per_k',
                    'destroyed_background_conductance_w_per_k', 'destroyed_active_conductance_w_per_k'):
            contact['connectivity'][key] *= heat_factor
        manifest['thermal_contacts'].append(dict(port_id=contact['port_id'],
            compartment=contact['compartment'], before=before, after=copy.deepcopy(contact)))
    # Reconstruct the permitted edit and prove nothing else was changed.
    expected = copy.deepcopy(original)
    changed_edges = {e['id']: e['after'] for e in manifest['channels']}
    expected['gas_network']['channels'] = [changed_edges.get(e['id'], e)
        for e in expected['gas_network']['channels']]
    changed_contacts = {(e['compartment'], e['port_id']): e['after'] for e in manifest['thermal_contacts']}
    expected['gas_network']['thermal_contacts'] = [changed_contacts.get((c['compartment'], c['port_id']), c)
        for c in expected['gas_network']['thermal_contacts']]
    if scene != expected:
        raise ValueError('Unexpected nonlocal edit')
    return scene, manifest


def summary(scene, run):
    report = inspect(scene, run)
    gaps = {n for n in run['gas'] if n.startswith('gap-ISS1-')}
    exits = [e for e in scene['gas_network']['channels'] if e['id'].startswith('escape-ISS1-')]
    inlet = {e['to'] for e in exits}
    samples = []
    for at in (860., 885., 900., 930., 950., run['time_s'][-1]):
        def temperature(names):
            mass = {n: float(np.interp(at, run['time_s'], run['gas'][n]['mass_kg'])) for n in names}
            return sum(mass[n]*float(np.interp(at, run['time_s'], run['gas'][n]['temperature_k']))
                       for n in names)/sum(mass.values())-273.15
        samples.append(dict(time_s=at, surface_c=float(np.interp(at, run['time_s'],
            run['temperature_k']['108']))-273.15, gap_gas_c=temperature(gaps), inlet_gas_c=temperature(inlet),
            mixed_perimeter_mass_kg=sum(float(np.interp(at, run['time_s'],
                run['channels'][e['id']]['mixed_mass_kg'])) for e in exits)))
    report['C3_ISS_samples'] = samples
    report['residuals'] = {k: run[k] for k in ('max_thermal_residual_j', 'max_chemical_residual_j',
        'max_mass_residual_kg', 'max_oxygen_residual_kg') if k in run}
    if report['minimum_regular_step_s'] < .005-1e-12:
        raise ValueError('Regular step below 0.005 s')
    return report


def local_plot(out, labels=None, title=None):
    """Small memory footprint: retain curves, not three large JSON runs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    state = json.loads((out/'status.json').read_text())
    labels = labels or dict(baseline='Исходный', gas_heat_max='Газ + тепло: максимум', gas_max='Только газ: максимум')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    curve_summary = {}
    for case in state['cases']:
        folder = Path(state['source']) if case == 'baseline' else out/case
        scene = json.loads((folder/'scenario.json').read_text())
        with gzip.open(folder/'replay/run.json.gz', 'rt') as stream:
            run = json.load(stream)
        times = np.asarray(run['time_s'])
        window = (times >= 840.) & (times <= 1000.)
        curve_summary[case] = {}
        gaps = {n for n in run['gas'] if n.startswith('gap-ISS1-')}
        inlet = {e['to'] for e in scene['gas_network']['channels'] if e['id'].startswith('escape-ISS1-')}
        for key, names, ax in (('inlet_gas', inlet, axes[0, 0]), ('gap_gas', gaps, axes[0, 1])):
            masses = np.asarray([run['gas'][n]['mass_kg'] for n in sorted(names)])
            temperatures = np.asarray([run['gas'][n]['temperature_k'] for n in sorted(names)])
            temperature = np.sum(masses*temperatures, axis=0)/np.sum(masses, axis=0)-273.15
            ax.plot(times, temperature, label=labels[case])
            peak = int(np.argmax(temperature[window]))
            curve_summary[case][key] = dict(temperature_930_c=float(np.interp(930., times, temperature)),
                temperature_950_c=float(np.interp(950., times, temperature)),
                sampled_window=[840., 1000.], sampled_mean_peak_c=float(temperature[window][peak]),
                sampled_mean_peak_time_s=float(times[window][peak]),
                sampled_hottest_node_c=float(np.max(temperatures[:, window]))-273.15)
        axes[0, 2].plot(times, np.asarray(run['temperature_k']['108'])-273.15, label=labels[case])
        for solid, ax in zip(('C3:case:-x', 'ISS1', 'steel-L'), axes[1]):
            heat = sum(np.asarray(c['heat_to_solid_j']) for c in run['gas_film_trace'].values() if c['solid'] == solid)
            ax.plot(times, heat/1000., label=labels[case])
        del run
    titles = ('Газ на подводе к щели', 'Газ внутри щели', 'Расчётная поверхность C3/ISS',
              'Нетто газ → оболочка C3/ISS', 'Нетто газ → ISS1', 'Нетто газ → стальная плита')
    for i, (ax, panel_title) in enumerate(zip(axes.flat, titles)):
        ax.set(title=panel_title, xlabel='Время, с', ylabel='°C' if i < 3 else 'кДж', xlim=(840., 1000.))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(title or 'Локализация C3/ISS: подвод → щель → поверхности; не калибровка')
    fig.savefig(out/'local-path.png', dpi=140); plt.close(fig)
    cal.save(out/'local-path-summary.json', curve_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--heat-factor', type=float, default=1000.)
    parser.add_argument('--timeout', type=float, default=1800.)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if args.plot_only:
        local_plot(args.output.resolve())
        return 0
    source = dataio.ROOT/'data/calibration/upv-separated-chemistry-v6/thermal-gas-mixing-20261020-r1/temperature'
    original = json.loads((source/'scenario.json').read_text())
    out = args.output.resolve(); out.mkdir(exist_ok=False)
    core = hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    state = dict(status='running', pid=os.getpid(), source=str(source), native_sha256=core,
        calibrated=False, validation_run=False, diagnostic_only=True, cases={})
    cal.save(out/'status.json', state)
    runs = {'baseline': json.load(gzip.open(source/'replay/run.json.gz', 'rt'))}
    data, _ = dataio.load('no-cooling')
    labels = {'baseline': 'Последний исходный прогон', 'gas_heat_max': 'Локальный максимум газа и тепла',
              'gas_max': 'Локальный максимум только газа'}
    def stop(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        state['cases']['baseline'] = summary(original, runs['baseline'])
        cal.save(out/'status.json', state)
        # Combined test requested by the user first; the mass-only contrast then
        # distinguishes limited admission from rapid cooling to local solids.
        for label, factor in (('gas_heat_max', args.heat_factor), ('gas_max', 1.)):
            scene, manifest = local_maximum(original, heat_factor=factor)
            folder = out/label; folder.mkdir()
            cal.save(folder/'scenario.json', scene); cal.save(folder/'manifest.json', manifest)
            if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest() != core:
                raise ValueError('Native binary changed during diagnostic')
            state['active_case'] = label; cal.save(out/'status.json', state)
            print('START', label, 'channels', len(manifest['channels']), 'contacts', len(manifest['thermal_contacts']), flush=True)
            run = native(folder/'scenario.json', folder/'replay', dict(stop_after_first_rupture=False), args.timeout)
            report = summary(scene, run); cal.save(folder/'report.json', report)
            state['cases'][label] = report; runs[label] = run; cal.save(out/'status.json', state)
            plot(out, runs, data, labels=labels, title='Диагностика C3/ISS: экстремальный локальный перенос, без калибровки')
            print('CASE', label, 'speed', report['speed'], 'samples', json.dumps(report['C3_ISS_samples']), flush=True)
        state['status'] = 'complete'
    except KeyboardInterrupt:
        state['status'] = 'stopped'
    except Exception as exc:
        state.update(status='failed', error=str(exc))
    finally:
        state.pop('active_case', None); cal.save(out/'status.json', state)
        print('LOCAL_MAX_STATUS', state['status'], state.get('error', ''), flush=True)
    return 0 if state['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
