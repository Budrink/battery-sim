"""Identify ONLY the cooled experiment's boundary from channels 110--112.

The cheap plate surrogate initializes h/G; it is not a replacement solver.
Each selected boundary must be checked by a fresh native replay. No battery
thermocouple, pressure, chemistry, material property or old contact is fitted.
"""
import copy
import math
import numpy as np
from scipy.optimize import least_squares

from upv_frozen_validation import digest

COOLING_CONTACTS = tuple(f'Al{i}--cooling' for i in (1, 2, 3))
COEFFICIENTS = ('background_conductance_w_per_k', 'active_conductance_w_per_k',
                'destroyed_background_conductance_w_per_k', 'destroyed_active_conductance_w_per_k')
FLOW_HEAT_CAPACITY_W_K = 2.e-3/60.*998.*4180.
# Effective engineering bounds, not measured material-property intervals.
BOUNDS = np.array([[100., .05], [2.e5, 100.]])


def measurements(data, times):
    t = np.asarray(data['time_s'])
    baseline = (t >= 200.) & (t <= 450.)
    if not np.any(baseline):
        raise ValueError('Measured resting water baseline is required')
    delta = data['T112_k']-data['T111_k']
    offset = float(np.median(delta[baseline]))
    if min(times) < t[0] or max(times) > t[-1]:
        raise ValueError('Cooling observation extrapolation is forbidden')
    return dict(contact_k=np.interp(times, t, data['T110_k']),
                inlet_k=np.interp(times, t, data['T111_k']),
                incremental_water_power_w=FLOW_HEAT_CAPACITY_W_K*(np.interp(times, t, delta)-offset),
                resting_outlet_minus_inlet_k=offset)


def _reference(law):
    return law['background_conductance_w_per_k'] + law['active_conductance_w_per_k']/(1.+math.exp(-law.get('geometry_reference_logit', 0.)))


def _scale(law, factor):
    for key in COEFFICIENTS:
        if key in law:
            law[key] *= factor


def candidate(source, data, parameters):
    """Modify just new Al/plate contacts, the water boundary and observations."""
    h, g = parameters['contact_h_w_per_m2_k'], parameters['water_g_w_per_k']
    if not (BOUNDS[0, 0] <= h <= BOUNDS[1, 0] and BOUNDS[0, 1] <= g <= BOUNDS[1, 1]):
        raise ValueError('Cooling parameters outside explicit engineering bounds')
    scene = copy.deepcopy(source)
    ports = {p['id']: p for p in scene['ports']}
    found = set()
    for edge in scene['interfaces']:
        if edge['id'] in COOLING_CONTACTS:
            area = ports[edge['port_a']]['area_m2']
            _scale(edge['law'], h*area/_reference(edge['law']))
            found.add(edge['id'])
    if found != set(COOLING_CONTACTS):
        raise ValueError('Exactly three physical Al/plate cooling contacts required')
    water = [b for b in scene['boundaries'] if b['id'] == 'water']
    if len(water) != 1 or 'connectivity' not in water[0]:
        raise ValueError('One directed water boundary required')
    water = water[0]
    _scale(water['connectivity'], g/_reference(water['connectivity']))
    water['conductance_w_per_k'] = g
    end = scene['simulation']['duration_s']
    times = np.unique(np.r_[0., np.arange(10., end, 10.), end])
    inlet = measurements(data, times)['inlet_k']
    water['temperature_k'] = float(inlet[0])
    water['temperature_profile'] = [[float(t), float(k)] for t, k in zip(times, inlet)]
    if any(p['id'] == '110' for p in scene['measurement_probes']):
        raise ValueError('Cooling observer 110 already exists')
    scene['measurement_probes'].append(dict(id='110', cell_id='Al2', interface_id='Al2--cooling'))
    scene['thermal_path_trace'] = scene['surface_probe_trace'] = True
    scene['metadata']['cooling_boundary_identification'] = dict(
        parameters=copy.deepcopy(parameters), source_cooled_scene_sha256=digest(source),
        fit_channels=['110', '111', '112'], battery_temperatures_used_for_selection=False,
        pressure_used_for_selection=False, chemistry_refitted=False,
        thermal_materials_or_old_contacts_refitted=False,
        inlet_driver='Measured channel 111, linearly resampled every 10 s; no extrapolation',
        contact_observer='110: Al2 hot contact FACE, native half-cell resistance correction; not plate bulk',
        limitations=['Contact sensor side/location is an explicit observation assumption, not fitted',
                     'Small water differential is baseline-corrected; absolute type-K accuracy does not ensure differential accuracy',
                     'Not fully blind validation: only the cooling boundary is identified from the cooled test'])
    scene['metadata']['frozen_validation'].update(parameters_refitted=True,
        validation_used_for_selection=True, cooling_boundary_only_refitted=True,
        chemistry_refitted=False, battery_materials_or_old_contacts_refitted=False)
    return scene


def half_resistances(scene):
    cells = {c['id']: c for c in scene['cells']}
    al, plate = cells['Al2'], cells['cooling-plate']
    area = math.prod(al['geometry']['dimensions_m'][:2])
    ra = al['geometry']['dimensions_m'][2]/al['thermal']['divisions'][2]/2./(235.*area)
    rb = plate['geometry']['dimensions_m'][2]/plate['thermal']['divisions'][2]/2./(235.*area)
    rw = plate['geometry']['dimensions_m'][2]/plate['thermal']['divisions'][2]/2./(235.*math.prod(plate['geometry']['dimensions_m'][:2]))
    # This read-only reconstruction is valid only for the documented intact Al.
    for name in ('Al1', 'Al2', 'Al3', 'cooling-plate'):
        c = cells[name]
        if c['thermal']['conductivity_w_per_m_k'] != [235.]*3 or c.get('conversion'):
            raise ValueError('Cooling audit requires documented passive Al conductivity')
        response = c['thermal'].get('bulk_response', {})
        if any(any(response.get(key, [0.]*3)) for key in ('linear', 'quadratic', 'damage', 'temperature_contrast', 'damage_contrast')) or response.get('temperature_log_response'):
            raise ValueError('Nonconstant Al bulk law needs an explicit audit extension')
    return ra, rb, rw


def response(surface, source, receiver):
    table = surface.get('temperature_log_response')
    if table:
        a = np.array(table)
        z = np.interp(source, a[:, 0], a[:, 1])
    else:
        theta = (source-surface.get('reference_temperature_k', 298.15))/surface.get('temperature_scale_k', 100.)
        z = surface.get('linear', [0.]*3)[2]*theta + surface.get('quadratic', [0.]*3)[2]*theta**2
    return z + surface.get('temperature_contrast', [0.]*3)[2]*(source-receiver)/surface.get('temperature_scale_k', 100.)


def directed_g(law, a_surface, b_surface, ta, tb, series_r):
    za, zb = response(a_surface, ta, tb), response(b_surface, tb, ta)
    z = law.get('geometry_reference_logit', 0.) + np.where(np.asarray(ta) >= tb, za, zb)
    raw = law['background_conductance_w_per_k'] + law['active_conductance_w_per_k']/(1.+np.exp(-np.clip(z, -40., 40.)))
    return 1./(1./raw+series_r)


def native_observations(scene, run, data):
    times = np.array(run['time_s'])
    obs = measurements(data, times)
    ra, _, rw = half_resistances(scene)
    interface = run['thermal_paths']['interfaces']['Al2--cooling']
    reconstructed = np.array(interface['a_adjacent_temperature_k'])-ra*np.array(interface['a_to_b_power_w'])
    contact = np.array(run['temperature_k'].get('110', reconstructed))
    plate = np.array(run['thermal_paths']['objects']['cooling-plate']['reference_capacity_weighted_temperature_k'])
    water = next(b for b in scene['boundaries'] if b['id'] == 'water')
    cells = {c['id']: c for c in scene['cells']}
    gw = directed_g(water['connectivity'], cells['cooling-plate']['thermal']['surface_connectivity'],
                    water.get('remote_surface_connectivity', {}), plate, obs['inlet_k'], rw)
    power = gw*(plate-obs['inlet_k'])
    baseline = (times >= 200.) & (times <= 450.)
    if not np.any(baseline):
        raise ValueError('Native baseline is missing')
    return dict(time_s=times, contact_k=contact, water_power_w=power,
        incremental_water_power_w=power-float(np.median(power[baseline])),
        hot_face_reconstruction_max_difference_k=float(np.max(np.abs(contact-reconstructed))),
        water_power_audit='Directed S-law at capacity-weighted plate T; spatial approximation, not exact boundary ledger',
        inlet_k=obs['inlet_k'], resting_outlet_minus_inlet_k=obs['resting_outlet_minus_inlet_k'])


def objective(predicted_contact, predicted_power, observed, times, window=(650., 1250.)):
    mask = (times >= window[0]) & (times <= window[1])
    if np.sum(mask) < 10:
        raise ValueError('Insufficient independent cooling observations')
    # Engineering tolerances, NOT claimed instrument uncertainties.
    return np.r_[(predicted_contact[mask]-observed['contact_k'][mask])/2.,
                 (predicted_power[mask]-observed['incremental_water_power_w'][mask])/
                 np.maximum(4., .2*np.abs(observed['incremental_water_power_w'][mask]))]


def boundary_score(scene, run, data):
    values = native_observations(scene, run, data)
    times = np.arange(650., min(1250., run['time_s'][-1])+1e-6, 5.)
    contact, power = (np.interp(times, values['time_s'], values[k]) for k in ('contact_k', 'incremental_water_power_w'))
    obs = measurements(data, times)
    residual = objective(contact, power, obs, times)
    return dict(score=float(np.mean(residual**2)), fit_window_s=[float(times[0]), float(times[-1])],
        contact_rmse_k=float(np.sqrt(np.mean((contact-obs['contact_k'])**2))),
        incremental_water_power_rmse_w=float(np.sqrt(np.mean((power-obs['incremental_water_power_w'])**2))),
        contact_observer_reconstruction_error_k=values['hot_face_reconstruction_max_difference_k'],
        fitted_channels=['110', '111', '112'], battery_or_pressure_target_used=False)


def surrogate_seed(scene, run, data):
    """Use recorded native Al temperatures as forcing, not experimental cell TC."""
    times = np.arange(0., min(1250., run['time_s'][-1])+1e-7, 2.)
    observed = measurements(data, times)
    paths = run['thermal_paths']['interfaces']
    al = np.array([np.interp(times, run['time_s'], paths[k]['a_adjacent_temperature_k']) for k in COOLING_CONTACTS])
    cells = {c['id']: c for c in scene['cells']}
    thermal = cells['cooling-plate']['thermal']
    ra, rb, rw = half_resistances(scene)
    base_law = next(e['law'] for e in scene['interfaces'] if e['id'] == COOLING_CONTACTS[0])
    base_water = next(b for b in scene['boundaries'] if b['id'] == 'water')
    area = math.prod(cells['Al2']['geometry']['dimensions_m'][:2])
    al_s = cells['Al2']['thermal']['surface_connectivity']
    plate_s = thermal['surface_connectivity']
    def predict(x):
        h, g = np.exp(x)
        law, water = copy.deepcopy(base_law), copy.deepcopy(base_water['connectivity'])
        _scale(law, h*area/_reference(law)); _scale(water, g/_reference(water))
        plate = np.empty(len(times)); plate[0] = thermal['initial_temperature_k']
        for j in range(1, len(times)):
            gi = directed_g(law, al_s, plate_s, al[:, j-1], plate[j-1], ra+rb)
            gw = float(directed_g(water, plate_s, base_water.get('remote_surface_connectivity', {}), plate[j-1], observed['inlet_k'][j-1], rw))
            total = float(np.sum(gi)+gw)
            equilibrium = (float(np.dot(gi, al[:, j-1]))+gw*observed['inlet_k'][j-1])/total
            plate[j] = equilibrium+(plate[j-1]-equilibrium)*math.exp(-total*2./thermal['heat_capacity_j_per_k'])
        flux = directed_g(law, al_s, plate_s, al[1], plate, ra+rb)*(al[1]-plate)
        q = directed_g(water, plate_s, base_water.get('remote_surface_connectivity', {}), plate, observed['inlet_k'], rw)*(plate-observed['inlet_k'])
        baseline = (times >= 200.) & (times <= 450.)
        return al[1]-ra*flux, q-float(np.median(q[baseline]))
    def residual(x):
        contact, power = predict(x)
        return objective(contact, power, observed, times)
    fit = least_squares(residual, np.log([20000., 6.]), bounds=np.log(BOUNDS), max_nfev=45)
    h, g = np.exp(fit.x)
    return dict(contact_h_w_per_m2_k=float(np.clip(h, *BOUNDS[:, 0])),
                water_g_w_per_k=float(np.clip(g, *BOUNDS[:, 1]))), dict(
        method='Frozen native Al-state driven single-plate initializer; fresh CUDA verification mandatory',
        surrogate_score=float(np.mean(fit.fun**2)),
        contact_at_upper_bound=bool(h > .99*BOUNDS[1, 0]),
        inference='Upper-bound h indicates only a strong-contact lower bound, not an identified exact conductivity' if h > .99*BOUNDS[1, 0] else 'Finite interior effective h')


def plot_boundary(folder, scene, run, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    v = native_observations(scene, run, data); t = v['time_s']; obs = measurements(data, t)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for ax, prediction, target, ylabel in (
        (axes[0], v['contact_k']-273.15, obs['contact_k']-273.15, '110: поверхность радиатора, °C'),
        (axes[1], v['incremental_water_power_w'], obs['incremental_water_power_w'], 'Дополнительный теплоотвод, Вт')):
        ax.plot(t, target, color='#222222', label='Эксперимент 110–112')
        ax.plot(t, prediction, label='Расчёт')
        ax.axvspan(650., 1250., color='#79a9c4', alpha=.12, label='Область определения границы')
        ax.set(xlabel='Время, с', ylabel=ylabel); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Определение охлаждения; температуры батарей и давление не подбирались')
    fig.savefig(folder/'cooling-boundary.png', dpi=140); plt.close(fig)
