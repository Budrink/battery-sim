#!/usr/bin/env python3
"""Separate pressure retention, aperture speed, gas arrival and TC averaging.

Fresh ambient starts; exact explicit source scene. Separate rupture strength,
opening rate and an optional hot internal-connectivity response, uniformly for
equal materials. No fitted event trigger/time shift, chemical energy addition,
measured k/Cp, boundary-contact or geometry refit.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import calibrate_upv_r123 as cal
import upv_r123_data as dataio
from check_upv_restart import native
from upv_destruction_objective import reject_reason, score
from upv_cooling_review import phase_report, plot_cooling


CASES = dict(retain4=(4., 1.), fast4=(4., 100.), fast16=(16., 100.), bulk4=(4., 1.))
CASES['bulk4-r3'] = (4., 1.)


def steep_r3(original, activation_j_mol=80000.):
    """Isolate R3 temperature selectivity, retaining its reference rate/energy.

    No fitted ignition temperature or event time. Mobile fuel/O2, R1/R2 and
    every thermal/mass connection are unchanged. All identical gas mixtures
    use the same Arrhenius law, including gas that can re-enter an open cell.
    """
    if not np.isfinite(activation_j_mol) or not 40000. <= activation_j_mol <= 160000.:
        raise ValueError('R3 activation probe outside bounds')
    scene = copy.deepcopy(original);changed = 0
    for gas in scene['gas_network']['compartments']:
        if gas.get('reaction_rate_per_s',0.) <= 0.:continue
        gas['reaction_activation_j_per_mol'] = activation_j_mol
        changed += 1
    if not changed:raise ValueError('No reacting gas compartments')
    scene.setdefault('metadata',{})['r3_activation_probe'] = dict(
        activation_j_per_mol=activation_j_mol, reference_rate_unchanged=True,
        fuel_and_oxidizer_inventories_unchanged=True, diagnostic_only=True)
    return scene


def hot_internal_recovery(original, start_k=425., recovered_k=500.):
    """Constitutive probe of high-T internal mixing, NOT a new heat source.

    Retain the exact fitted linear response up to start_k. Recover towards
    nominal internal connectivity as a continuous function of local state,
    not a clock/event trigger. All equal cell materials get the same law.
    Measured k, Cp, shell/solid boundary and chemical kinetics stay unchanged.
    This extrapolation is unknown experimentally, never an accepted material.
    """
    if not np.isfinite(start_k+recovered_k) or not 390. <= start_k < recovered_k <= 900.:
        raise ValueError('Invalid internal recovery interval')
    scene = copy.deepcopy(original); changed = 0
    for cell in scene['cells']:
        if cell.get('role','battery') != 'battery': continue
        profile = cell['thermal']['bulk_response']
        linear = profile['linear']
        if len(set(linear)) != 1 or any(profile.get('quadratic',[0.,0.,0.])) or 'temperature_log_response' in profile:
            raise ValueError('This probe requires the explicit isotropic linear reference response')
        tref = profile['reference_temperature_k'];scale = profile['temperature_scale_k']
        knots = sorted({273.15,tref,330.,390.,start_k})
        values = [[t,linear[0]*(t-tref)/scale] for t in knots]
        values += [[recovered_k,0.],[max(900.,recovered_k+1.),0.]]
        profile['linear'] = [0.,0.,0.]
        profile['temperature_log_response'] = values
        changed += 1
    if not changed: raise ValueError('No cell interiors')
    scene.setdefault('metadata',{})['hot_internal_recovery_probe'] = dict(
        unchanged_through_k=start_k, recovered_k=recovered_k,
        diagnostic_only=True, temperature_is_local_state_not_event_trigger=True)
    return scene


def shell_trial(original, strength, rate, duration):
    if not np.isfinite(strength) or not .25 <= strength <= 64.:
        raise ValueError('Shell strength factor outside diagnostic bounds')
    if not np.isfinite(rate) or not .1 <= rate <= 100.:
        raise ValueError('Aperture speed factor outside diagnostic bounds')
    if not np.isfinite(duration) or duration <= 0.:
        raise ValueError('Duration must be finite and positive')
    scene = copy.deepcopy(original)
    changed = 0
    for c in scene['gas_network']['channels']:
        if c.get('kind') != 'rupture':
            continue
        law = c['damage_opening']
        if law['mode'] != 'deterministic_aperture':
            raise ValueError('Impulse diagnosis requires deterministic apertures')
        c['opening_pressure_pa'] *= strength
        law['area_rate_per_s'] *= rate
        changed += 1
    if not changed:
        raise ValueError('No rupture channels')
    scene['simulation']['duration_s'] = duration
    scene['surface_probe_trace'] = True
    scene.setdefault('metadata', {})['rupture_impulse_probe'] = dict(
        strength_factor=strength, area_rate_factor=rate,
        diagnostic_only=True, no_fitted_event_time=True,
        all_shells_share_parameters=True, fresh_from_ambient=True)
    return scene


def crossing(t, values, threshold):
    indices = np.flatnonzero(np.asarray(values) >= threshold)
    return float(t[indices[0]]) if len(indices) else None


def impulse_report(scene, run, data):
    t = np.asarray(run['time_s'], dtype=float)
    ids = [c['id'] for c in scene['gas_network']['channels']
           if c.get('kind') == 'rupture' and c['from'].startswith('inside-C2-')]
    def total(field):
        return sum((np.asarray(run['channels'][c][field], dtype=float) for c in ids), np.zeros(len(t)))
    mass = total('transferred_mass_kg')
    energy = total('delivered_gas_energy_j')
    area = total('effective_area_m2')
    source = run.get('chemistry', {}).get('C2')
    produced = (np.asarray(source['r2_generated_gas_kg']) + np.asarray(source['evaporated_kg']) -
                np.asarray(source['condensed_kg'])) if source else None
    flow = np.diff(mass)/np.diff(t)  # Accepted interval flux, not raw orifice proposal.
    pressure = np.asarray(dataio.chamber_pressure_pa(run), dtype=float)/1e5
    event = dataio.event(data)['time_s']
    mask = (t >= event-60.) & (t <= event+80.)
    peak_index = np.flatnonzero(mask)[np.argmax(pressure[mask])]
    late = mass[-1]
    edges = {str(f): crossing(t, mass, f*late) for f in (.01, .1, .5, .9)} if late > 0 else {}
    region = [g for name,g in run['gas'].items() if name.startswith('gap-ISS1-')]
    hot = np.max(np.asarray([g['temperature_k'] for g in region], dtype=float), axis=0)
    envelope = run.get('surface_probe_trace', {}).get('108', {})
    def first_hot(values):
        return crossing(t, np.asarray(values, dtype=float), 373.15)
    samples = []
    for target in (750., 850., 875., event, 890., 900., 910., 930., 950., t[-1]):
        if not t[0] <= target <= t[-1]: continue
        i = int(np.argmin(abs(t-target)))
        item = dict(time_s=float(t[i]), pressure_bar=float(pressure[i]),
            cumulative_mass_g=float(mass[i]*1000.), open_area_mm2=float(area[i]*1e6),
            delivered_energy_kj=float(energy[i]/1000.), gap_ISS1_max_c=float(hot[i]-273.15),
            C3_ISS_mean_c=float(run['temperature_k']['108'][i]-273.15))
        if envelope:
            item['C3_ISS_max_c'] = float(envelope['maximum_temperature_k'][i]-273.15)
        if source:
            item['net_gas_and_vapor_produced_g'] = float(produced[i]*1000.)
            item['r2_power_kw'] = float(source['r2_power_w'][i]/1000.)
        samples.append(item)
    result = dict(complete=not run.get('partial', False), known_until_s=float(t[-1]),
        first_open_sample_s=crossing(t, area, 1e-12), net_ejected_mass_g=float(mass[-1]*1000.),
        net_ejected_mass_crossings_s=edges,
        peak_accepted_interval_flow_g_s=float(np.max(flow)*1000.),
        peak_accepted_interval_flow_interval_s=t[np.argmax(flow):np.argmax(flow)+2].tolist(),
        chamber_peak_bar=float(pressure[peak_index]), chamber_peak_s=float(t[peak_index]),
        chamber_peak_delay_s=float(t[peak_index]-event),
        first_gap_ISS1_100c_s=first_hot(hot),
        first_C3_ISS_mean_100c_s=first_hot(run['temperature_k']['108']), samples=samples,
        observation_note='Face maximum is a spatial envelope, not a fitted thermocouple or proof of jet direction.')
    if envelope:
        result['first_C3_ISS_max_100c_s'] = first_hot(envelope['maximum_temperature_k'])
    return result


def plot_impulse(out, runs, data, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    event = dataio.event(data)['time_s']
    raw_t = np.asarray(data['time_s'])
    targets = dataio.observed(data, raw_t)
    axes[0,0].plot(raw_t, targets['pressure_bar'], 'k', label='Эксперимент')
    axes[2,0].plot(raw_t, targets['108']-273.15, 'k', label='Эксперимент')
    for name, r in runs.items():
        t = np.asarray(r['time_s']); label = labels.get(name, name)
        axes[0,0].plot(t, np.asarray(dataio.chamber_pressure_pa(r))/1e5, label=label)
        for ax, field, scale in ((axes[0,1], 'area', 1e6), (axes[1,0], 'mass', 1e3), (axes[1,1], 'energy', .001)):
            ax.plot(t, np.asarray(r[field])*scale, label=label)
        axes[2,0].plot(t, np.asarray(r['temperature_k']['108'])-273.15, label=label)
        line, = axes[2,1].plot(t, np.asarray(r['gap_max'])-273.15, label=label+' — газ')
        envelope = r.get('surface_probe_trace', {}).get('108')
        if envelope:
            axes[2,1].plot(t, np.asarray(envelope['maximum_temperature_k'])-273.15,
                '--', color=line.get_color(), label=label+' — максимум поверхности')
    titles = ('Давление камеры, бар', 'Открытая площадь C2, мм²', 'Нетто-выброс C2, г',
              'Энтальпия из C2 без химического запаса, кДж', 'C3/ISS: среднее поверхности, °C',
              'C3/ISS: локальный газ и поверхность, °C')
    for ax,title in zip(axes.flat,titles):
        ax.set(title=title, xlim=(event-40., event+80.), xlabel='Время, с')
        ax.axvline(event, color='grey', linestyle=':', linewidth=1)
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(out/'rupture-impulse.png', dpi=140); plt.close(fig)


def compact(scene, run):
    ids = [c['id'] for c in scene['gas_network']['channels']
           if c.get('kind') == 'rupture' and c['from'].startswith('inside-C2-')]
    result = {k:run[k] for k in ('time_s','temperature_k','gas_observers')}
    result['surface_probe_trace'] = run.get('surface_probe_trace', {})
    result['gap_max'] = np.max([g['temperature_k'] for name,g in run['gas'].items()
                              if name.startswith('gap-ISS1-')],axis=0).tolist()
    for short,key in (('area','effective_area_m2'),('mass','transferred_mass_kg'),('energy','delivered_gas_energy_j')):
        result[short] = np.sum([run['channels'][c][key] for c in ids],axis=0).tolist()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cases', nargs='+', choices=CASES, default=['retain4','fast4'])
    ap.add_argument('--duration', type=float, default=1186.639)
    args = ap.parse_args(); out = args.output.resolve(); out.mkdir(exist_ok=False)
    source = args.reference.resolve(); original = json.loads((source/'scenario.json').read_text())
    data,_ = dataio.load('no-cooling'); event = dataio.event(data)
    core = hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    state = dict(status='running', source=str(source), native_sha256=core, cases={},
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 calibrated=False, validation_run=False, diagnostic_only=True)
    cal.save(out/'status.json', state)
    runs = {}; labels = dict(retain4='Прочность ×4; прежняя скорость', fast4='Прочность ×4; скорость ×100',
                            fast16='Прочность ×16; скорость ×100',
                            bulk4='Прочность ×4; восстановление внутренней S при 425–500 K',
                            **{'bulk4-r3':'Горячая внутренняя S; R3 Ea=80 кДж/моль'})
    try:
        for name in args.cases:
            if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest() != core:
                raise RuntimeError('Native binary changed during diagnosis')
            scene = shell_trial(original, *CASES[name], args.duration)
            if name in ('bulk4','bulk4-r3'):
                scene = hot_internal_recovery(scene)
                scene['metadata']['zone_diagnostics'] = True
            if name == 'bulk4-r3': scene = steep_r3(scene)
            scene['failure_sample_times_s'] = sorted(set(dataio.observation_times(data,event,args.duration)) | {0.,args.duration})
            folder = out/name; folder.mkdir(); cal.save(folder/'scenario.json',scene)
            state['active_case'] = name; cal.save(out/'status.json',state)
            print('START',name,flush=True)
            try:
                run = native(folder/'scenario.json',folder/'replay',dict(stop_after_first_rupture=False),
                             max(1600.,args.duration*2.),
                             prune_fn=lambda r: reject_reason(r,dict(data=data,event=event)))
            except (RuntimeError, TimeoutError) as exc:
                state['cases'][name] = dict(status='rejected',reason=str(exc));cal.save(out/'status.json',state)
                print('REJECTED',name,str(exc),flush=True);continue
            report = impulse_report(scene,run,data)
            report['temperature_phases'] = phase_report(run,data,args.duration)
            report['objective'] = score(run,dict(data=data,event=event,fit_end_s=args.duration))
            report['wall_s'] = run['wall_s'];report['realtime_factor'] = args.duration/run['wall_s']
            report['minimum_regular_step_s'] = run['solver_profile']['minimum_regular_step_s']
            report['balance'] = {k:run[k] for k in ('max_thermal_residual_j','max_chemical_residual_j','max_mass_residual_kg')}
            cal.save(folder/'report.json',report);state['cases'][name] = report
            runs[name] = compact(scene,run);del run
            cal.save(folder/'curves.json',runs[name]);cal.save(out/'status.json',state)
            plot_impulse(out,runs,data,labels);plot_cooling(out,runs,data,labels)
            print('RESULT',name,json.dumps(report),flush=True)
        state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='stopped' if isinstance(exc,KeyboardInterrupt) else 'failed', error=str(exc));raise
    finally:
        state.pop('active_case',None);cal.save(out/'status.json',state)


if __name__ == '__main__': main()
