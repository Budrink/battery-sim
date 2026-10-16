#!/usr/bin/env python3
"""Analytic warmup energy balance for the no-cooling UPV experiment.

Internal conduction and the unknown contact conductances cancel in a balance
taken over the whole stack, so the required heat source follows from measured
temperatures, the fixed measured capacities of the scene and the measured
current alone. Unobserved layers are reported as an explicit band, not as a
single interpolated number.

The 0-D kinetic fit uses the same rate laws as native/include/battery_native
(conversion.hpp, joule.hpp) driven by the *measured* C2 temperature. It is a
starting point for the CUDA search, not a replacement for it: it collapses the
3x3x3 mesh and every gradient into one lumped temperature.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import upv_r123_data as dataio
import upv_r123_model as model

RGAS = 8.31446261815324
# Battery faces carrying thermocouples, as mapped in upv_r123_model.build.
SENSOR_FACE = {('C1', 'ISS'): '101', ('C1', 'TIM'): '102',
               ('C3', 'TIM'): '107', ('C3', 'ISS'): '108',
               ('C2', 'TIM'): 'C2_TIM', ('C2', 'ISS'): 'C2_ISS'}


def stack(scene):
    """Ordered (id, capacity_j_per_k, centre_x) for every solid in the stack."""
    rows = []
    shells = {s['contents_id']: s for s in scene['shells']}
    for cell in scene['cells']:
        if cell['id'] == 'chamber-wall':
            continue
        capacity = cell['thermal']['heat_capacity_j_per_k']
        shell = shells.get(cell['id'])
        if shell:
            dims = cell['geometry']['dimensions_m']
            outer = [v + 2*shell['thickness_m'] for v in dims]
            volume = math.prod(outer) - math.prod(dims)
            capacity += volume*shell['density_kg_m3']*shell['specific_heat_j_kg_k']
        rows.append((cell['id'], capacity, cell['pose']['position_m'][0], cell['material_id']))
    return sorted(rows, key=lambda r: r[2])


def sensor_positions(scene):
    """Thermocouple x-coordinates: the battery face each probe is attached to."""
    cells = {c['id']: c for c in scene['cells']}
    out = {}
    for (ident, material), channel in SENSOR_FACE.items():
        cell = cells[ident]
        half = cell['geometry']['dimensions_m'][0]/2
        neighbours = [r for r in stack(scene) if r[3] == material]
        left = min(neighbours, key=lambda r: abs(r[2] - (cell['pose']['position_m'][0] - half)))
        right = min(neighbours, key=lambda r: abs(r[2] - (cell['pose']['position_m'][0] + half)))
        side = -1 if abs(left[2] - cell['pose']['position_m'][0]) < abs(right[2] - cell['pose']['position_m'][0]) else 1
        out[channel] = cell['pose']['position_m'][0] + side*half
    return out


def layer_temperatures(scene, observed, mode):
    """Temperature history per solid layer under one unobserved-layer rule."""
    positions = sensor_positions(scene)
    order = sorted(positions, key=lambda c: positions[c])
    x = np.array([positions[c] for c in order])
    sensed = np.array([observed[c] for c in order])            # (sensors, times)
    ambient = observed['ambient']
    result = {}
    for ident, _capacity, centre, _material in stack(scene):
        if ident in ('C1', 'C2', 'C3'):
            faces = [c for (cell, _m), c in SENSOR_FACE.items() if cell == ident]
            result[ident] = np.mean([observed[c] for c in faces], axis=0)
            continue
        if mode == 'observed-only':
            result[ident] = ambient
        elif mode == 'neighbour-max':
            j = int(np.argmin(np.abs(x - centre)))
            near = [j] + ([j-1] if j and x[j] > centre else []) + ([j+1] if j+1 < len(x) and x[j] < centre else [])
            result[ident] = np.max(sensed[near], axis=0)
        else:
            inside = (centre >= x[0]) & (centre <= x[-1])
            if inside:
                j = int(np.searchsorted(x, centre)) - 1
                f = (centre - x[j])/(x[j+1] - x[j])
                result[ident] = sensed[j] + f*(sensed[j+1] - sensed[j])
            else:
                # Outside the instrumented span only the room is known.
                j = 0 if centre < x[0] else len(x)-1
                result[ident] = .5*(sensed[j] + ambient)
    return result


def environment_loss(scene, temperatures, observed):
    """Lateral gas/room losses with the scene's own contact areas."""
    ports = {p['id']: p for p in scene['ports']}
    loss = np.zeros_like(observed['ambient'])
    for contact in scene['gas_network']['thermal_contacts']:
        port = ports[contact['port_id']]
        ident = port['cell_id'].split(':')[0]
        if ident not in temperatures:
            continue
        loss = loss + contact['h_w_per_m2_k']*port['area_m2']*(temperatures[ident] - observed['ambient'])
    return loss


def smooth(y, window):
    if window <= 1:
        return y
    kernel = np.ones(window)/window
    padded = np.r_[np.full(window, y[0]), y, np.full(window, y[-1])]
    return np.convolve(padded, kernel, mode='same')[window:-window]


def chain_power(time, temperature, current, p, net_heat_thermal):
    """Exact two-stage chain at frozen T per step, as in conversion.hpp."""
    reactant, activated = p['mass_kg'], 0.
    power = np.zeros_like(time)
    for i in range(len(time)-1):
        dt = time[i+1] - time[i]
        t = temperature[i]
        ka = p['r1_rate']*math.exp(min(80., p['r1_activation']/RGAS*(1/400. - 1/t))) + \
            p['overcharge_gain']*max(0., current[i])/p['nominal_charge_c'] * \
            math.exp(min(80., p['r1_activation']/RGAS*(1/model.REFERENCE - 1/t)))
        kb = p['r2_rate']*math.exp(min(80., max(-80., p['r2_activation']/RGAS*(1/450. - 1/t))))
        da = reactant*(-math.expm1(-ka*dt))
        difference = abs(kb - ka)
        feed = reactant*ka*math.exp(-min(ka, kb)*dt) * \
            (-math.expm1(-difference*dt)/difference if difference > 0 else dt)
        db = activated*(-math.expm1(-kb*dt)) + max(0., da - min(da, feed))
        reactant -= da
        activated += da - db
        power[i] = net_heat_thermal*db/dt
    power[-1] = power[-2] if len(power) > 1 else 0.
    return power, 1 - reactant/p['mass_kg']


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('--output', type=Path, default=dataio.ROOT/'data/calibration/upv-r123-overcharge-v2/heat-balance.json')
    cli.add_argument('--smooth-s', type=float, default=30.)
    cli.add_argument('--energy-j', type=float, help='Override the fixed R2 reservoir of the heat stage')
    cli.add_argument('--gas-fraction', type=float, help='Override the fuel share locked into gas')
    cli.add_argument('--plot', type=Path)
    # The warmup chain is precursor-limited, so every kb >> ka reproduces the
    # same heat. Capping kb selects the numerically cheapest member of that
    # family instead of an arbitrarily stiff one.
    cli.add_argument('--max-r2-rate', type=float)
    args = cli.parse_args()

    data, _record = dataio.load('no-cooling')
    mark = dataio.event(data)
    scene, _meta = model.build('no-cooling', warmup=True)
    end = scene['simulation']['duration_s']
    time = data['time_s'][data['time_s'] <= end]
    observed = {c: np.mean([data['T'+i+'_k'] for i in ids], axis=0)[:len(time)]
                for c, ids in dataio.CHANNELS.items()}
    observed['ambient'] = data['T109_k'][:len(time)]
    observed['pressure_bar'] = 2.5*data['U114_v'][:len(time)]

    electrical, electrical_meta = dataio.electrical_record()
    offset = electrical_meta_offset = mark['time_s'] - electrical[-1, 0]
    current = np.interp(time, electrical[:, 0] + offset, electrical[:, 2], left=0., right=0.)
    supplied = np.r_[0., np.cumsum(.5*(np.maximum(0., electrical[:, 1]*electrical[:, 2])[:-1] +
                                      np.maximum(0., electrical[:, 1]*electrical[:, 2])[1:]) *
                                  np.diff(electrical[:, 0]))]
    work = float(np.interp(end, electrical[:, 0] + offset, supplied, left=0., right=supplied[-1]))

    window = max(1, int(args.smooth_s/np.median(np.diff(time))))
    capacities = {r[0]: r[1] for r in stack(scene)}
    report = {'duration_s': float(end), 'event_s': mark['time_s'], 'supplied_ui_work_j': work,
              'stack_capacity_j_per_k': float(sum(capacities.values())),
              'electrical_offset_s': float(electrical_meta_offset), 'bands': {}}

    bands = {}
    for mode in ('observed-only', 'interpolated', 'neighbour-max'):
        temperatures = layer_temperatures(scene, observed, mode)
        stored = sum(capacities[k]*(v - v[0]) for k, v in temperatures.items())
        loss = environment_loss(scene, temperatures, observed)
        cumulative_loss = np.r_[0., np.cumsum(.5*(loss[:-1] + loss[1:])*np.diff(time))]
        bands[mode] = dict(stored=stored, loss=loss, cumulative_loss=cumulative_loss,
                           temperatures=temperatures)
        report['bands'][mode] = dict(
            stored_j=float(stored[-1]), lost_j=float(cumulative_loss[-1]),
            required_source_j=float(stored[-1] + cumulative_loss[-1]),
            per_cell_j={k: float(capacities[k]*(v[-1] - v[0])) for k, v in temperatures.items()})

    r_max = model.BOUNDS['resistance_ohm'][1]
    squared = np.r_[0., np.cumsum(.5*(current[:-1]**2 + current[1:]**2)*np.diff(time))]
    report['joule'] = dict(
        squared_current_integral_a2s=float(squared[-1]),
        max_resistance_ohm=r_max, max_joule_j=float(squared[-1]*r_max),
        voltage_limited_resistance_ohm=float((electrical[0, 1] + .05 - 3.65)/electrical[0, 2]))

    required = bands['interpolated']['stored'] + bands['interpolated']['cumulative_loss']
    smoothed = smooth(required, window)
    power = np.gradient(smoothed, time)

    names = ('resistance_ohm', 'overcharge_gain', 'r2_rate', 'r2_activation')
    fixed = dict(model.DEFAULT)
    if args.energy_j:
        fixed['energy_j'] = args.energy_j
    if args.gas_fraction is not None:
        fixed['gas_fraction'] = args.gas_fraction
    net_heat_thermal = fixed['energy_j']*(1 - fixed['gas_fraction'])/.2
    chain = dict(mass_kg=.2, r1_rate=fixed['r1_rate'], r1_activation=fixed['r1_activation'],
                 nominal_charge_c=35.*3600.)

    def residual(x):
        p = dict(chain, **{k: float(np.exp(v)) if k != 'r2_activation' else float(v)
                           for k, v in zip(names, x)})
        chemical, _converted = chain_power(time, observed['C2_TIM'], current, p, net_heat_thermal)
        joule = p['resistance_ohm']*current**2
        return (smooth(chemical, window) + joule - power)/10.

    bounds = dict(model.BOUNDS)
    if args.max_r2_rate:
        bounds['r2_rate'] = (bounds['r2_rate'][0], args.max_r2_rate)
        fixed['r2_rate'] = min(fixed['r2_rate'], args.max_r2_rate)
    lo = [math.log(bounds[k][0]) if k != 'r2_activation' else bounds[k][0] for k in names]
    hi = [math.log(bounds[k][1]) if k != 'r2_activation' else bounds[k][1] for k in names]
    start = [math.log(fixed[k]) if k != 'r2_activation' else fixed[k] for k in names]
    solution = least_squares(residual, start, bounds=(lo, hi), x_scale='jac', diff_step=.05)
    fitted = dict(chain, **{k: float(np.exp(v)) if k != 'r2_activation' else float(v)
                            for k, v in zip(names, solution.x)})
    chemical, converted = chain_power(time, observed['C2_TIM'], current, fitted, net_heat_thermal)
    joule = fitted['resistance_ohm']*current**2
    cumulative = np.r_[0., np.cumsum(.5*((chemical+joule)[:-1] + (chemical+joule)[1:])*np.diff(time))]
    report['fit'] = dict(
        parameters={k: fitted[k] for k in names},
        energy_j=fixed['energy_j'], gas_fraction=fixed['gas_fraction'],
        reactant_converted_fraction=float(converted),
        chemical_j=float(np.trapz(chemical, time)), joule_j=float(np.trapz(joule, time)),
        delivered_j=float(cumulative[-1]), required_j=float(required[-1]),
        power_rmse_w=float(np.sqrt(np.mean((smooth(chemical, window) + joule - power)**2))),
        thermal_budget_j=float(net_heat_thermal*.2),
        initial_ir_v=float(3.65 + electrical[0, 2]*fitted['resistance_ohm']),
        measured_initial_voltage_v=float(electrical[0, 1]))
    # Feasibility of the chemical part against the sealed pouch. R2 heat and R2
    # gas are the same event in this model, so the required heat implies a
    # minimum gas mass and a minimum internal pressure before any venting.
    inside = sum(n['volume_m3'] for n in scene['gas_network']['compartments'] if n['id'].startswith('inside-C2'))
    gas_model = scene['gas_network']
    chemical_required = required[-1] - squared[-1]*r_max
    best_budget = model.BOUNDS['energy_j'][1]*(1 - model.BOUNDS['gas_fraction'][0])/.2
    cases = {'heat-stage defaults': (net_heat_thermal, fixed['gas_yield']),
             'most favourable bounds': (best_budget, model.BOUNDS['gas_yield'][0])}
    temperature = float(observed['C2_TIM'][-1])
    feasibility = {}
    for label, (heat_per_kg, yield_) in cases.items():
        converted = chemical_required/heat_per_kg
        gas = yield_*converted
        feasibility[label] = dict(
            thermal_j_per_kg=heat_per_kg, gas_yield=yield_,
            converted_kg=converted, converted_fraction=converted/.2, gas_kg=gas,
            internal_pressure_pa=gas*gas_model['specific_r_j_per_kg_k']*temperature/inside,
            max_shell_strength_pa=model.BOUNDS['shell_strength_pa'][1])
    chamber = next(n for n in gas_model['compartments'] if n['id'] == 'chamber')
    observed_rise_pa = float((np.nanmax(observed['pressure_bar']) - observed['pressure_bar'][0])*1e5)
    # Part of the measured chamber rise is simply the warmed chamber gas.
    thermal_share = float(observed['pressure_bar'][0]*1e5 *
        (observed['ambient'][-1] - observed['ambient'][0])/observed['ambient'][0])
    # A seam/valve that weeps before the matrix removes the pressure objection,
    # so the binding constraint becomes the gas inventory the chamber actually
    # saw. Retained mass is negligible: the pouch void cannot bank the gas.
    shell = next(s for s in scene['shells'] if s['contents_id'] == 'C2')
    strength = fixed['shell_strength_pa']*math.exp(-max(0., temperature - model.REFERENCE)/300.)
    crack = .2*strength
    vent_window = end - float(time[np.argmax(observed['C2_TIM'] > 330.)])
    gamma = 1 + gas_model['specific_r_j_per_kg_k']/gas_model['cv_j_per_kg_k']
    choked = math.sqrt(gamma)*(2/(gamma + 1))**((gamma + 1)/(2*(gamma - 1)))
    unit_flow = .7*crack/math.sqrt(gas_model['specific_r_j_per_kg_k']*temperature)*choked
    seam = dict(
        crack_pressure_pa=crack, shell_strength_at_temperature_pa=strength,
        retained_mass_kg=crack*inside/(gas_model['specific_r_j_per_kg_k']*temperature),
        vent_window_s=vent_window, damage_onset_k=shell['damage_onset_k'])
    report['gas_feasibility'] = dict(
        seam=seam,
        chemical_required_j=float(chemical_required), inside_c2_volume_m3=inside,
        cases=feasibility, chamber_volume_m3=chamber['volume_m3'],
        observed_chamber_rise_pa=observed_rise_pa, thermal_expansion_share_pa=thermal_share,
        vented_mass_upper_bound_kg=float(max(0., observed_rise_pa - thermal_share)*chamber['volume_m3'] /
                                        (gas_model['specific_r_j_per_kg_k']*float(observed['ambient'][-1]))))
    vented = report['gas_feasibility']['vented_mass_upper_bound_kg']
    report['gas_feasibility'].update(
        required_heat_per_gas_j_per_kg=float(chemical_required/vented),
        available_heat_per_gas_j_per_kg=float(best_budget/model.BOUNDS['gas_yield'][0]),
        seam_area_for_vented_mass_m2=float(vented/vent_window/unit_flow))
    report['series'] = dict(
        time_s=time.tolist(), required_j=required.tolist(), required_power_w=power.tolist(),
        fitted_power_w=(chemical + joule).tolist(), joule_power_w=joule.tolist())

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].fill_between(time, bands['observed-only']['stored'] + bands['observed-only']['cumulative_loss'],
                             bands['neighbour-max']['stored'] + bands['neighbour-max']['cumulative_loss'],
                             alpha=.25, label='unobserved-layer band')
        axes[0].plot(time, required, label='required source, interpolated')
        axes[0].plot(time, squared*r_max, '--', label='Joule ceiling at R=24.5 mΩ')
        axes[0].plot(time, cumulative, label='analytic fit, chemistry + I²R')
        axes[0].set_xlabel('Physical time, s');axes[0].set_ylabel('Energy, J');axes[0].legend();axes[0].grid(alpha=.3)
        axes[1].plot(time, power, label='required power')
        axes[1].plot(time, smooth(chemical, window) + joule, label='fitted power')
        axes[1].plot(time, joule, '--', label='I²R only')
        axes[1].set_xlabel('Physical time, s');axes[1].set_ylabel('Power, W');axes[1].legend();axes[1].grid(alpha=.3)
        figure.suptitle('no-cooling warmup: analytic stack balance')
        figure.tight_layout();figure.savefig(args.plot, dpi=110)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    printable = {k: v for k, v in report.items() if k != 'series'}
    print(json.dumps(printable, indent=2))


if __name__ == '__main__':
    main()
