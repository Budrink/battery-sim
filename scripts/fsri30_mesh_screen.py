"""Mesh-aware SEARCH windows, never corrected temperatures or validation scores.

The only measured discrepancy is the primary event on one matched 80/20 mm
parameter pair. Other events and 40 mm receive bounded engineering allowances,
not a fabricated bias or a claimed statistical confidence interval.
"""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from fsri30_wave_fit import POLICY, assess_fields


def primary_bias_from_pair(coarse, fine):
    result={}
    for key in ('pulse_groups','arrival_groups'):
        a=coarse['model'][key];b=fine['model'][key]
        if not a or not b or a[0]['channels']!=b[0]['channels']:
            raise ValueError('Paired mesh primary events must identify the same model component')
        result[key]=float(a[0]['time_s']-b[0]['time_s'])
    return dict(primary_bias_s=result,observed_parameter_pairs=1,
                measured_level_mm=80,target_level_mm=20,
                primary_uncertainty_s=100.,
                unmeasured_allowance_s=max(100.,50.*math.ceil(max(abs(v) for v in result.values())/50.)),
                scope='Primary events only; local numerical evidence, not a global time correction',
                allowance_is_engineering_prior=True)


def window(level,kind,index,evidence):
    if level==20:return 0.,POLICY.timing_limit_s
    if level not in (80,40):raise ValueError('Unsupported mesh level')
    if level==80 and index==0:
        return evidence['primary_bias_s'][kind],POLICY.timing_limit_s+evidence['primary_uncertainty_s']
    # Do not assume linear mesh convergence, halve the bias at 40 mm, or
    # copy a primary-event shift to secondary waves.
    return 0.,POLICY.timing_limit_s+evidence['unmeasured_allowance_s']


def match(truth,predicted,end,level,kind,evidence):
    due=[]
    for i,a in enumerate(truth):
        bias,limit=window(level,kind,i,evidence)
        if end>=a['end_s']+bias+limit+POLICY.confirmation_s:
            due.append((i,a,bias,limit))
    cost=np.full((len(due),len(predicted)+len(due)),1e6)
    for k,(i,a,bias,limit) in enumerate(due):
        for j,b in enumerate(predicted):
            if (i==0)!=(j==0):continue  # Never substitute the primary for a cascade.
            error=abs(b['time_s']-a['time_s']-bias)
            if error<=limit:cost[k,j]=error/limit
    rows,cols=linear_sum_assignment(cost)
    pairs=[];missing=[]
    for k,j in zip(rows,cols):
        i,a,bias,limit=due[k]
        if j>=len(predicted) or cost[k,j]>=1e6:
            missing.append(dict(reference_time_s=a['time_s'],channels=a['channels'],
                                search_center_offset_s=bias,search_half_width_s=limit))
        else:
            b=predicted[j];offset=b['time_s']-a['time_s']
            pairs.append(dict(reference_time_s=a['time_s'],model_time_s=b['time_s'],
                              offset_s=offset,search_offset_s=offset-bias,
                              search_center_offset_s=bias,search_half_width_s=limit,
                              reference_channels=a['channels'],model_channels=b['channels']))
    return dict(due=len(due),matched=len(pairs),pairs=pairs,missing=missing,
                worst_offset_s=max((abs(p['offset_s']) for p in pairs),default=0.),target_met=False)


def assess(t,values,ids,ref,level,evidence):
    raw=assess_fields(t,values,ids,ref)
    if level==20:return raw  # EXACT original fine-grid policy, no bias or relaxed guard.
    model=raw['model'];end=raw['last_observed_s']
    pulse=match(ref['pulse_groups'],model['pulse_groups'],end,level,'pulse_groups',evidence)
    arrival=match(ref['arrival_groups'],model['arrival_groups'],end,level,'arrival_groups',evidence)
    bias,limit=window(level,'pulse_groups',0,evidence)
    offset=raw['primary_offset_s'];search_offset=None if offset is None else offset-bias
    reasons=[]
    if search_offset is not None and abs(search_offset)>limit:
        reasons.append(f'Primary outside coarse search window: raw offset {offset:g}s; center {bias:g}s, half-width {limit:g}s')
    for name,matched in [('thermal pulse',pulse),('first hot arrival',arrival)]:
        if matched['missing']:
            times=', '.join(f"{m['reference_time_s']:g}" for m in matched['missing'])
            reasons.append(f'Missing {name} wave(s) at {times}s even with mesh uncertainty')
    complete=raw['phase_window_complete'];evaluated=complete or bool(reasons)
    missing=sum(len(m['missing']) for m in (pulse,arrival))
    violation=max(float(missing),abs(search_offset or 0.)/limit-1.,0.)
    return dict(raw,evaluated=evaluated,accepted=not reasons if evaluated else None,
                reason='; '.join(reasons) or None,constraint=max(.001,violation) if reasons else 0.,
                pulse_match=pulse,arrival_match=arrival,
                raw_assessment={k:v for k,v in raw.items() if k!='model'},
                screening_only=True,fine_grid_verification_required=True,
                search_primary_offset_s=search_offset,primary_target_offset_s=bias,
                primary_search_tolerance_s=50.,
                primary_deadline_s=ref['pulse_groups'][0]['end_s']+bias+limit+POLICY.confirmation_s,
                timing_target_met=False,
                objective=(sum(abs(p['search_offset_s'])/p['search_half_width_s']
                               for m in (pulse,arrival) for p in m['pairs'])
                           if complete and not reasons else None),
                scope='Coarse mesh refinement eligibility ONLY, not experimental acceptance')


def uncertain_temperature_window(end,ref,level,evidence):
    if level==20:return False
    for i,g in enumerate(ref['pulse_groups']):
        bias,limit=window(level,'pulse_groups',i,evidence)
        if end>=g['time_s']+bias-limit and end-60.<=g['end_s']+bias+limit+POLICY.confirmation_s:
            return True
    return False
