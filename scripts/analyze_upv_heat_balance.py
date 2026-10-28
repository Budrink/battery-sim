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


def precursor_branch(p, temperature, voltage):
    """Same frozen-on-substep R1 byproduct fraction as conversion.hpp."""
    phi0 = p.get('precursor_yield', 1.)
    bt = p.get('precursor_temperature', 0.)
    bv = p.get('precursor_voltage', 0.)
    if phi0 >= 1. and bt == 0. and bv == 0.:
        return 1.
    exponent = bt*(temperature - model.REFERENCE) + bv*(voltage - 3.65)
    return min(1., max(0., phi0*math.exp(min(80., max(-80., exponent)))))


def chain_power(time, temperature, current, p, net_heat_thermal, voltage=None):
    """Uncapped initial-guess chain, NOT an electrical-budget validation.

    Same independent formation rates and shared B energy as conversion.hpp;
    CUDA still enforces local terminal-work limits and spatial temperatures.
    extra is stored electrical energy expressed in equivalent B kilograms.
    """
    reactant, activated = p['mass_kg'], 0.
    extra=0.
    power = np.zeros_like(time)
    completed = np.zeros_like(time)
    for i in range(len(time)-1):
        dt = time[i+1] - time[i]
        t = temperature[i]
        v = 3.65 if voltage is None else float(voltage[i])
        kd=p['r1_rate']*math.exp(min(80., p['r1_activation']/RGAS*(1/400. - 1/t)))
        kc=p['charge_formation_gain']*max(0., current[i])/p['nominal_charge_c']
        ka=kd+kc
        kb = p['r2_rate']*math.exp(min(80., max(-80., p['r2_activation']/RGAS*(1/450. - 1/t))))
        phi = precursor_branch(p, t, v)
        da = reactant*(-math.expm1(-ka*dt))
        difference = abs(kb - ka)
        feed = reactant*ka*math.exp(-min(ka, kb)*dt) * \
            (-math.expm1(-difference*dt)/difference if difference > 0 else dt)
        to_precursor = phi*da
        db = activated*(-math.expm1(-kb*dt)) + phi*max(0., da - min(da, feed))
        share=kc/ka if ka>0 else 0.
        extra_burned=extra*(-math.expm1(-kb*dt))+phi*max(0.,da-min(da,feed))*share
        extra=extra*math.exp(-kb*dt)+phi*min(da,feed)*share
        reactant -= da
        activated += to_precursor - db
        power[i] = net_heat_thermal*(db+extra_burned)/dt
        completed[i] = db
    power[-1] = power[-2] if len(power) > 1 else 0.
    return power, 1 - reactant/p['mass_kg'], completed


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

    cooling_data, _cooling_record = dataio.load('active-cooling')
    cooling_mark = dataio.event(cooling_data)
    _drive, _volts, cooling_electrical = dataio.current_drive('active-cooling', cooling_data, cooling_mark)
    cooling_onset = cooling_electrical['onset_s']
    cooling_u = float(np.interp(cooling_onset, cooling_data['time_s'], cooling_data['U115_v']))
    cooling_r_max = (cooling_u - 3.65)/cooling_electrical['current_a']
    r_max = min(model.BOUNDS['resistance_ohm'][1],
                float((electrical[0, 1] + .05 - 3.65)/electrical[0, 2]),
                cooling_r_max)
    squared = np.r_[0., np.cumsum(.5*(current[:-1]**2 + current[1:]**2)*np.diff(time))]
    report['joule'] = dict(
        squared_current_integral_a2s=float(squared[-1]),
        max_resistance_ohm=r_max, max_joule_j=float(squared[-1]*r_max),
        voltage_limited_resistance_ohm=float((electrical[0, 1] + .05 - 3.65)/electrical[0, 2]),
        cooling_onset_resistance_ohm=float(cooling_r_max),
        cooling_onset_voltage_v=cooling_u,
        envelope_is_electrical_work_not_thermal_selection=True)

    required = bands['interpolated']['stored'] + bands['interpolated']['cumulative_loss']
    smoothed = smooth(required, window)
    power = np.gradient(smoothed, time)

    names = ('resistance_ohm', 'charge_formation_gain', 'precursor_yield', 'r2_rate', 'r2_activation')
    linear = {'r2_activation', 'precursor_yield'}
    fixed = dict(model.DEFAULT)
    if args.energy_j:
        fixed['energy_j'] = args.energy_j
    if args.gas_fraction is not None:
        fixed['gas_fraction'] = args.gas_fraction
    net_heat_thermal = fixed['energy_j']*(1 - fixed['gas_fraction'])/.2
    chain = dict(mass_kg=.2, r1_rate=fixed['r1_rate'], r1_activation=fixed['r1_activation'],
                 nominal_charge_c=35.*3600., precursor_temperature=fixed['precursor_temperature'],
                 precursor_voltage=fixed['precursor_voltage'])
    voltage = np.interp(time, electrical[:, 0] + offset, electrical[:, 1],
                        left=electrical[0, 1], right=electrical[-1, 1])

    def encode(key, value):
        return value if key in linear else math.log(value)

    def decode(key, value):
        return float(value if key in linear else np.exp(value))

    def residual(x):
        p = dict(chain, **{k: decode(k, v) for k, v in zip(names, x)})
        chemical, _converted, _db = chain_power(time, observed['C2_TIM'], current, p, net_heat_thermal, voltage)
        joule = p['resistance_ohm']*current**2
        return (smooth(chemical, window) + joule - power)/10.

    bounds = dict(model.BOUNDS)
    bounds['resistance_ohm'] = (bounds['resistance_ohm'][0], r_max)
    if args.max_r2_rate:
        bounds['r2_rate'] = (bounds['r2_rate'][0], args.max_r2_rate)
        fixed['r2_rate'] = min(fixed['r2_rate'], args.max_r2_rate)
    lo = [encode(k, bounds[k][0]) for k in names]
    hi = [encode(k, bounds[k][1]) for k in names]
    start = [encode(k, min(max(fixed[k], bounds[k][0]), bounds[k][1])) for k in names]
    solution = least_squares(residual, start, bounds=(lo, hi), x_scale='jac', diff_step=.05)
    fitted = dict(chain, **{k: decode(k, v) for k, v in zip(names, solution.x)})
    chemical, converted, completed = chain_power(time, observed['C2_TIM'], current, fitted, net_heat_thermal, voltage)
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
    # The model already suppresses the yield below the reaction reference with
    # gas_yield_temperature_per_k, but upv_r123_model fixes it at 5e-4/K and the
    # stages never fit it. Solve for the slope that keeps the pre-vent gas under
    # the measured ceiling, weighting each gram by when it is actually produced.
    weights = completed/completed.sum()
    driver = np.asarray(observed['C2_TIM']) - 450.
    for label, case in feasibility.items():
        def produced(slope):
            yield_ = np.minimum(.8, case['gas_yield']*np.exp(slope*driver))
            return case['converted_kg']*float(np.sum(weights*yield_))
        lo_slope, hi_slope = 0., .2
        for _ in range(64):
            mid = .5*(lo_slope + hi_slope)
            if produced(mid) > vented:
                lo_slope = mid
            else:
                hi_slope = mid
        case['required_gas_yield_slope_per_k'] = .5*(lo_slope + hi_slope)
        case['equivalent_activation_j_per_mol'] = float(case['required_gas_yield_slope_per_k']*RGAS*450.**2)
        case['runaway_yield_at_600k'] = float(min(.8, case['gas_yield']*math.exp(case['required_gas_yield_slope_per_k']*150.)))
        case['pre_vent_yield'] = float(np.sum(weights*np.minimum(.8, case['gas_yield'] *
            np.exp(case['required_gas_yield_slope_per_k']*driver))))
    # Seed the heat search with the slope that closes the chamber-gas ceiling
    # at the heat-stage defaults. energy_j / gas_yield themselves stay fixed.
    report['fit']['parameters']['gas_yield_temperature'] = feasibility['heat-stage defaults']['required_gas_yield_slope_per_k']
    report['gas_feasibility'].update(
        required_heat_per_gas_j_per_kg=float(chemical_required/vented),
        available_heat_per_gas_j_per_kg=float(best_budget/model.BOUNDS['gas_yield'][0]),
        seam_area_for_vented_mass_m2=float(vented/vent_window/unit_flow))
    report['series'] = dict(
        time_s=time.tolist(), required_j=required.tolist(), required_power_w=power.tolist(),
        fitted_power_w=(chemical + joule).tolist(), joule_power_w=joule.tolist())

    # Held-out cooling is a check on the heat path, never a fit target.
    cooling_scene, _cooling_meta = model.build('active-cooling', warmup=True)
    cooling_end = cooling_scene['simulation']['duration_s']
    cooling_time = cooling_data['time_s'][cooling_data['time_s'] <= cooling_end]
    cooling_obs = {c: np.mean([cooling_data['T'+i+'_k'] for i in ids], axis=0)[:len(cooling_time)]
                   for c, ids in dataio.CHANNELS.items()}
    cooling_obs['ambient'] = cooling_data['T109_k'][:len(cooling_time)]
    cooling_layers = layer_temperatures(cooling_scene, cooling_obs, 'interpolated')
    cooling_cap = {r[0]: r[1] for r in stack(cooling_scene)}
    cooling_stored = sum(cooling_cap[k]*(v - v[0]) for k, v in cooling_layers.items() if k in cooling_cap)
    charge = (cooling_time >= cooling_onset) & (cooling_time <= cooling_end)
    c2_cool = .5*(cooling_obs['C2_TIM'] + cooling_obs['C2_ISS'])
    c2_train = .5*(observed['C2_TIM'] + observed['C2_ISS'])
    train_charge = current > 1.
    dTdt_train = float(np.polyfit(time[train_charge], c2_train[train_charge], 1)[0]) if train_charge.any() else 0.
    dTdt_cool = float(np.polyfit(cooling_time[charge], c2_cool[charge], 1)[0]) if charge.any() else 0.
    stack_c = float(sum(capacities.values()))
    implied_g = stack_c*(dTdt_train - dTdt_cool)/max(1e-6, float(np.mean(c2_cool[charge]) - 298.15))
    report['held_out_cooling'] = dict(
        used_for_selection=False,
        onset_s=float(cooling_onset), duration_s=float(cooling_end),
        c2_start_k=float(c2_cool[charge][0]) if charge.any() else None,
        c2_end_k=float(c2_cool[charge][-1]) if charge.any() else None,
        stored_j=float(cooling_stored[-1]),
        training_charge_dTdt_k_per_s=dTdt_train,
        cooling_charge_dTdt_k_per_s=dTdt_cool,
        implied_extraction_w=float(stack_c*(dTdt_train - dTdt_cool)),
        implied_sink_g_w_per_k=float(implied_g),
        prior_sink_g_w_per_k=model.DEFAULT['sink_g'],
        note='Order-of-magnitude plate path only; chemistry also slows when C2 is held down')
    seed = dict(model.DEFAULT, **report['fit']['parameters'])
    report['fit']['parameters'] = seed
    report['fit']['seed_note'] = 'DEFAULT plus 0-D no-cooling balance; cooling not used for selection'

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        args.plot.parent.mkdir(parents=True, exist_ok=True)
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

    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    printable = {k: v for k, v in report.items() if k != 'series'}
    print(json.dumps(printable, indent=2))


if __name__ == '__main__':
    main()
