"""Necessary ALL-wave checks from individual temperature histories.

No median, fitted time shift, invented sensor positions or failure count.
Heating pulses and first hot arrivals are distinct observations; neither is
automatically a measured battery explosion. Passing this gate is necessary,
not sufficient for a physical calibration or matched-thermocouple validation.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


@dataclass(frozen=True)
class WavePolicy:
    phase_end_s: float = 3600.
    smoothing_s: int = 5
    pulse_half_window_s: int = 10
    pulse_minimum_rise_k: float = 150.
    pulse_minimum_slope_k_s: float = 5.
    pulse_separation_s: int = 25
    group_span_s: float = 35.
    hot_threshold_k: float = 773.15
    timing_limit_s: float = 100.
    timing_target_s: float = 10.
    confirmation_s: float = 40.


POLICY = WavePolicy()


def group_events(events, policy=POLICY):
    """Bounded-span groups, not single-link chaining of a long cascade."""
    groups = []
    for event in sorted(events, key=lambda x: (x['time_s'], x['channel'])):
        if not groups or event['time_s']-groups[-1]['time_s'] > policy.group_span_s:
            groups.append(dict(time_s=event['time_s'], end_s=event['time_s'], events=[]))
        groups[-1]['events'].append(event)
        groups[-1]['end_s'] = event['time_s']
    for group in groups:
        group['channels'] = sorted({e['channel'] for e in group['events']})
    return groups


def signatures(t, values, ids, policy=POLICY):
    t, values = np.asarray(t, float), np.asarray(values, float)
    if (len(t) < 2 or values.shape != (len(t), len(ids)) or
            np.any(np.diff(t) <= 0) or not np.isfinite(values).all()):
        raise ValueError('Invalid wave observations')
    # A fixed physical 1 s grid avoids sample-count-dependent filtering.
    grid = np.arange(np.ceil(t[0]), np.floor(min(t[-1], policy.phase_end_s))+1.)
    if len(grid) < 2:
        return dict(pulses=[], arrivals=[], pulse_groups=[], arrival_groups=[])
    y = np.stack([np.interp(grid, t, values[:, i]) for i in range(len(ids))], axis=1)
    y = median_filter(y, size=(policy.smoothing_s, 1), mode='nearest')
    slope = np.gradient(y, grid, axis=0)
    half = policy.pulse_half_window_s
    margin = half+policy.smoothing_s//2
    pulses, arrivals = [], []
    for i, ident in enumerate(ids):
        peaks, _ = find_peaks(slope[:, i], height=policy.pulse_minimum_slope_k_s,
                             prominence=3., distance=policy.pulse_separation_s)
        for j in peaks:
            if j < margin or j+margin >= len(grid):
                continue  # Never extrapolate a peak into the uncomputed future.
            rise = float(y[j+half, i]-y[j-half, i])
            if rise >= policy.pulse_minimum_rise_k:
                pulses.append(dict(channel=str(ident), time_s=float(grid[j]), rise_k=rise,
                                   slope_k_s=float(slope[j, i])))
        hits = np.flatnonzero(y[:, i] >= policy.hot_threshold_k)
        if len(hits) and hits[0]+margin < len(grid):
            j = int(hits[0])
            arrivals.append(dict(channel=str(ident), time_s=float(grid[j])))
    return dict(pulses=pulses, arrivals=arrivals,
                pulse_groups=group_events(pulses, policy),
                arrival_groups=group_events(arrivals, policy))


def reference(data, policy=POLICY):
    ids = [f'T{i}' for i in range(1, 10)]
    result = signatures(data['time_s'], np.stack([data[f'{i}_k'] for i in ids], axis=1), ids, policy)
    if not result['pulse_groups'] or not result['arrival_groups']:
        raise ValueError('No resolved reference waves; revise observation policy explicitly')
    return dict(result, policy=asdict(policy),
                scope='All individual TC pulses and first hot-arrival groups; sensor mapping unknown',
                csv_runaway_flag_s=float(data['time_s'][np.flatnonzero(data['runaway'])[0]]))


def match_groups(truth, predicted, end, policy=POLICY):
    """Injective correspondence: one model wave cannot fill two data waves.

    Only expired reference windows can reject a prefix. Extra model groups
    are listed, not called extra measured explosions (30 cells / 9 TC).
    """
    due = [g for g in truth if end >= g['end_s']+policy.timing_limit_s+policy.confirmation_s]
    if not due:
        return dict(due=0, matched=0, missing=[], pairs=[], worst_offset_s=0., target_met=False)
    cost = np.full((len(due), len(predicted)+len(due)), 1e6)
    for i, a in enumerate(due):
        for j, b in enumerate(predicted):
            error = abs(b['time_s']-a['time_s'])
            if error <= policy.timing_limit_s:
                cost[i, j] = error
    rows, columns = linear_sum_assignment(cost)
    missing, pairs = [], []
    for i, j in zip(rows, columns):
        if j >= len(predicted) or cost[i, j] >= 1e6:
            nearest = min((abs(g['time_s']-due[i]['time_s']) for g in predicted), default=None)
            missing.append(dict(reference_time_s=due[i]['time_s'], channels=due[i]['channels'],
                                nearest_any_model_group_s=nearest))
        else:
            pairs.append(dict(reference_time_s=due[i]['time_s'], model_time_s=predicted[j]['time_s'],
                              offset_s=predicted[j]['time_s']-due[i]['time_s'],
                              reference_channels=due[i]['channels'], model_channels=predicted[j]['channels']))
    worst = max((abs(p['offset_s']) for p in pairs), default=0.)
    return dict(due=len(due), matched=len(pairs), missing=missing, pairs=pairs,
                worst_offset_s=worst, target_met=not missing and worst <= policy.timing_target_s)


def assess_fields(t, values, ids, ref, policy=POLICY):
    end = float(t[-1])
    model = signatures(t, values, ids, policy)
    pulses = match_groups(ref['pulse_groups'], model['pulse_groups'], end, policy)
    arrivals = match_groups(ref['arrival_groups'], model['arrival_groups'], end, policy)
    first_ref = ref['pulse_groups'][0]['time_s']
    first_model = model['pulse_groups'][0]['time_s'] if model['pulse_groups'] else None
    early = first_model is not None and first_model < first_ref-policy.timing_limit_s
    reasons = []
    if early:
        reasons.append(f'Primary thermal-front timing error: model {first_model:g}s versus data {first_ref:g}s')
    for label, matched in [('thermal pulse', pulses), ('first hot arrival', arrivals)]:
        if matched['missing']:
            times = ', '.join(f"{g['reference_time_s']:g}" for g in matched['missing'])
            reasons.append(f'Missing {label} wave(s) at {times}s within {policy.timing_limit_s:g}s timing error limit')
    complete = end >= policy.phase_end_s-1e-7
    evaluated = bool(reasons) or complete
    primary_offset = first_model-first_ref if first_model is not None else None
    worst = max(pulses['worst_offset_s'], arrivals['worst_offset_s'], abs(primary_offset or 0.))
    missing = len(pulses['missing'])+len(arrivals['missing'])
    violation = max(float(missing), max(0., -primary_offset/policy.timing_limit_s-1.) if early else 0.)
    return dict(evaluated=evaluated, accepted=(not reasons) if evaluated else None,
                phase_window_complete=complete, last_observed_s=end,
                reason='; '.join(reasons) if reasons else None,
                constraint=max(.001, violation) if reasons else 0.,
                primary_offset_s=primary_offset, timing_error_s=worst,
                timing_target_met=bool(complete and not reasons and pulses['target_met'] and arrivals['target_met']),
                pulse_match=pulses, arrival_match=arrivals, model=model,
                scope=ref['scope'], necessary_not_sufficient=True,
                objective=(sum(abs(p['offset_s'])/10. for m in (pulses, arrivals) for p in m['pairs'])
                           if complete and not reasons else None))


def assess(scene, run, data, policy=POLICY):
    from fsri30_objective import model_fields
    t, ids, values = model_fields(run)
    return assess_fields(t, values, ids, reference(data, policy), policy)
