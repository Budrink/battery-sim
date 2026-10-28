"""Mapping-free EXPLORATORY loss, cascade diagnostics, numerical comparisons.

TC positions and calorimeter boundary are not identified. No per-sensor
likelihood, measured component-failure order or independent-validation claim
can be manufactured from this loss. Nine TC envelopes vs thirty component
means are a declared search heuristic, NOT interchangeable measurements.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter


QUANTILES = np.array([0., .5, .9, 1.])
PHASE_WEIGHTS = dict(warmup=1., propagation=1., cooling=1.5)


def huber(x):
    a = np.abs(x)
    return np.where(a <= 2., a*a, 4.*a-4.)


def model_fields(run):
    t = np.asarray(run['time_s'], dtype=float)
    ids = sorted(run['cell_mean_temperature_k'])
    values = np.array([run['cell_mean_temperature_k'][key] for key in ids], dtype=float).T
    if len(t) < 2 or values.shape != (len(t), 30) or not np.all(np.isfinite(values)) or np.any(np.diff(t) <= 0):
        raise ValueError('Invalid/incomplete FSRI model observations')
    return t, ids, values


def targets(data):
    t = np.asarray(data['time_s'])
    if np.max(np.abs(np.diff(t)-1.)) > 1e-8:
        raise ValueError('Pulse detector requires published 1 Hz data')
    observed = np.array([data[f'T{i}_k'] for i in range(1, 10)]).T
    smooth = median_filter(observed, size=(11, 1), mode='nearest')
    derivative = np.gradient(smooth, t, axis=0)
    # Uncertain jet pulses are softened +/- 15s, not ignored. No fitting shift.
    burst = maximum_filter1d(np.any(np.abs(derivative) > 5., axis=1).astype(int), size=31) > 0
    event = float(t[np.flatnonzero(data['runaway'])[0]])
    return dict(time=t, q=np.quantile(smooth, QUANTILES, axis=1).T, burst=burst,
        phases=dict(warmup=t < event, propagation=(t >= event)&(t < 3600.), cooling=t >= 3600.), event=event)


def cascade(scene, run):
    """Count condensed R2 conversions, not native voxel/shell event messages.

    Ten percent of INITIAL reactive mass converted is a reporting threshold,
    never an ignition rule. Gas + residue close the R2 mass conversion budget.
    A hot thermocouple is deliberately not called a measured failed battery.
    """
    t, ids, values = model_fields(run)
    inventory = {c['id']: c['conversion']['mass_kg'] for c in scene['cells'] if c['role'] == 'battery'}
    events = []
    for key in ids:
        chemistry = run['chemistry'][key]
        converted = np.array(chemistry['residue_kg'])+np.array(chemistry['r2_generated_gas_kg'])
        hits = np.flatnonzero(converted >= .1*inventory[key])
        if len(hits):
            j = int(hits[0])
            when = float(t[j]) if j == 0 else float(np.interp(.1*inventory[key], converted[j-1:j+1], t[j-1:j+1]))
            events.append(dict(component=key, time_s=when))
    events.sort(key=lambda e: (e['time_s'], e['component']))
    return dict(reacted_components=len(events), events=events,
        inter_event_delays_s=[b['time_s']-a['time_s'] for a, b in zip(events, events[1:])],
        autonomous_cascade=len(events) >= 2,
        observed_failure_count=None, observed_component_order=None,
        note='Model R2 conversion order only; nine TC channels do NOT identify thirty failure events')


def loss(scene, run, data, *, include_features=True):
    t, ids, values = model_fields(run)
    target = targets(data)
    grid = target['time']
    known = grid <= t[-1]+1e-8
    q = np.quantile(values, QUANTILES, axis=1).T
    predicted = np.stack([np.interp(grid[known], t, q[:, j]) for j in range(len(QUANTILES))], axis=1)
    errors = predicted-target['q'][known]
    # Relative errors use ABSOLUTE KELVIN. Difference K == difference Celsius.
    scale = 15.+.06*target['q'][known]
    point_weight = np.where(target['burst'][known], .2, 1.)[:, None]
    losses = huber(errors/scale)*point_weight
    parts, rmse = {}, {}
    for name, mask in target['phases'].items():
        local = mask[known]
        # Fixed full-window denominator makes prefix loss a genuine lower bound.
        value = float(losses[local].sum()/max(1, mask.sum()*len(QUANTILES)))
        parts[name] = value*PHASE_WEIGHTS[name]
        rmse[name] = float(np.sqrt(np.mean(errors[local]**2))) if np.any(local) else None
    result = dict(objective=float(sum(parts.values())), phase_loss=parts, envelope_rmse_k=rmse,
        known_until_s=float(t[-1]), complete=t[-1] >= grid[-1]-1e-8 and not run.get('partial', False),
        peak_weight=.2, calibrated=False, validation_passed=False,
        scope='Exploratory TC-envelope versus component-mean heuristic; exact sensor placement unresolved')
    if include_features and result['complete']:
        process = cascade(scene, run)
        # Source says propagation occurs, NOT how many of thirty components fail.
        # Require only the minimal mechanistic meaning of cascade: >= two R2s.
        propagation = 8.*max(0, 2-process['reacted_components'])**2
        timing = huber((process['events'][0]['time_s']-target['event'])/90.).item() if process['events'] else 16.
        result['features'] = dict(minimum_autonomous_propagation=float(propagation), first_conversion_timing=float(timing))
        result['objective'] += propagation+timing
        result['cascade'] = process
    return result


def reject_reason(scene, run, data, best=None, *, incumbent_relative=False):
    """Never extrapolate a checkpoint to the uncomputed future/cooling tail."""
    if len(run.get('time_s', [])) < 2:
        return None
    t, ids, values = model_fields(run)
    if np.max(values) > 3500.:
        return 'critical component mean temperature above 3500 K'
    prefix = loss(scene, run, data, include_features=False)
    if best is not None and prefix['objective'] > max(best*1.25, best+5.):
        return f"full-window prefix lower bound {prefix['objective']:.4g} already exceeds incumbent {best:.4g}"
    if incumbent_relative:
        # Sensor mapping is unresolved and the initial baseline itself badly
        # underheats. Do not censor an improving full candidate for sharing
        # that baseline's cold prefix; use its proven lower loss bound instead.
        return None
    target = targets(data)
    mask = (target['time'] >= t[-1]-60.)&(target['time'] <= t[-1])&~target['burst']
    if np.sum(mask) < 50:
        return None
    q = np.quantile(values, QUANTILES, axis=1).T
    predicted = np.stack([np.interp(target['time'][mask], t, q[:, j]) for j in range(4)], axis=1)
    error = predicted-target['q'][mask]
    margin = np.maximum(300., .6*target['q'][mask])
    if np.any(np.mean(error > margin, axis=0) > .9) or np.any(np.mean(error < -margin, axis=0) > .9):
        return 'persistent gross envelope divergence for 60 s outside uncertain peaks'
    return None


def numerical_comparison(a, b):
    ta, _, va = model_fields(a)
    tb, _, vb = model_fields(b)
    if a.get('partial') or b.get('partial') or abs(ta[-1]-tb[-1]) > 1e-7:
        return dict(accepted=False, reason='Incomplete or different physical horizons')
    grid = np.arange(0., min(ta[-1], tb[-1])+1e-8, 5.)
    # Compare smoothed distributions, not unstable point maxima at a sharp front.
    qa, qb = (np.quantile(v, QUANTILES, axis=1).T for v in (va, vb))
    pa = np.stack([np.interp(grid, ta, qa[:, j]) for j in range(4)], axis=1)
    pb = np.stack([np.interp(grid, tb, qb[:, j]) for j in range(4)], axis=1)
    relative = float(np.sqrt(np.mean(((pa-pb)/np.maximum(pa, pb))**2)))
    def energies(run):
        return np.array([float(np.sum(np.array(run['released_energy_j_history']).reshape(len(run['time_s']), 30)[-1])),
            sum(float(g['released_energy_j'][-1]) for g in run['gas'].values())])
    ea, eb = energies(a), energies(b)
    energy_relative = (np.abs(ea-eb)/np.maximum(100., np.maximum(ea, eb))).tolist()
    return dict(accepted=relative <= .05 and max(energy_relative) <= .05,
        envelope_rms_relative_kelvin=relative, r2_sensible_relative_difference=energy_relative[0],
        recorded_external_r3_relative_difference=energy_relative[1], tolerance=.05,
        note='Necessary numerical check, NOT experimental validation or a proof of all local errors')


def data_audit(data, manifest):
    return dict(dataset_id=manifest['dataset_id'], csv_sha256=manifest['csv_sha256'],
        duration_s=float(data['time_s'][-1]), untimed_rows_excluded=manifest['untimed_rows_excluded'],
        sensor_mapping_confirmed=False, calorimeter_boundary_confirmed=False,
        observed_hrr_integral_j=float(np.trapz(data['hrr_kw']*1000., data['time_s'])),
        excluded_targets=['per-component measured failure order/count', 'individual TC RMSE', 'calorimeter HRR amplitude/integral'],
        known_omissions=['ABS combustion/pyrolysis', 'thermal radiation'],
        independent_validation_available=False, development_search_allowed=True,
        quantitative_validation_claim_allowed=False,
        note='Raw sources confirm thirty components and propagation, but do not resolve these identifiability limitations')
