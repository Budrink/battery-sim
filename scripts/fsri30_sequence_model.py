"""Shared, bounded all-wave search around the retained FSRI scene.

No forced failures, time shifts, energy inventory changes or per-cell fits.
The transferred R2 rate is an unverified cross-chemistry PRIOR, not measured
18650 data. Geometry coefficients remain fixture-specific and separately named.
"""
from copy import deepcopy
import math

import numpy as np

from bounded_bayes import propose
from fsri30_search_model import Parameter


SPACE=(
    Parameter('r2_rate_scale',.001,2.,1.,True,'Shared multiplier of inherited R2 rate; Ea unchanged'),
    Parameter('gas_can_contact_scale',.25,8.,1.,True,'Shared directed gas/can conductances, both directions kept'),
    Parameter('cross_column_open_fraction',.1,1.,1.,True,'Accessible fraction of existing x-directed free passages'),
    Parameter('axial_open_fraction',.1,1.,1.,True,'Accessible fraction of existing cap passages'),
    Parameter('thermal_stirring_coefficient',.01,.5,.06722339346183823,True,'Existing free-gas conservative mixing'),
    Parameter('effective_emissivity',.02,.95,.7664775109756737,False,'Existing common exterior boundary prior'),
)


def encode(parameters):
    return np.asarray([p.encode(parameters[p.name])for p in SPACE])


def decode(x):
    return {p.name:p.decode(float(v))for p,v in zip(SPACE,x)}


def initial(base):
    p={s.name:s.initial for s in SPACE}
    p['thermal_stirring_coefficient']=base['gas_network']['thermal_stirring_coefficient']
    p['effective_emissivity']=base['metadata']['exterior_radiation']['effective_emissivity']
    encode(p)
    return p


def candidate(base, parameters, duration=3600.):
    encode(parameters)
    if duration not in (3600.,5945.):raise ValueError('Explicit sequence or full-tail horizon required')
    # Rebuild only the existing radiation coefficients; geometry/physical state stays.
    scene=deepcopy(base)
    # add_exterior_radiation requires a non-radiating input. Update existing
    # radiating panels directly so inherited seams and gas routes cannot reset.
    for cell in scene['cells']:
        if cell.get('role')=='battery':
            cell['conversion']['reaction_rate_per_s']*=parameters['r2_rate_scale']
    for boundary in scene['boundaries']:
        if 'radiation' in boundary:
            boundary['radiation']['effective_emissivity']=parameters['effective_emissivity']
    gas=scene['gas_network']
    for film in gas['thermal_contacts']:
        if film['port_id'].startswith('B') and ':case:' in film['port_id']:
            for key in film['connectivity']:
                if key.endswith('conductance_w_per_k'):
                    film['connectivity'][key]*=parameters['gas_can_contact_scale']
    for edge in gas['channels']:
        if edge['id'].startswith('free-') and edge['mixing_axis']==0:
            edge['area_m2']*=parameters['cross_column_open_fraction']
        elif edge['id'].startswith('cap-'):
            edge['area_m2']*=parameters['axial_open_fraction']
    gas['thermal_stirring_coefficient']=parameters['thermal_stirring_coefficient']
    scene['simulation']['duration_s']=duration
    scene['failure_sample_times_s']=[float(t)for t in base['failure_sample_times_s']if t<=duration+1e-8]
    if abs(scene['failure_sample_times_s'][-1]-duration)>1e-8:
        raise ValueError('Horizon must be an existing observation, not a new integration landing point')
    scene['metadata']['sequence_search_parameters']=dict(parameters)
    scene['metadata']['sequence_search_scope']='All-wave calibration candidate; no sensor-position identification'
    scene['metadata']['exterior_radiation']['effective_emissivity']=parameters['effective_emissivity']
    scene['metadata']['calibration_status']='UNVALIDATED all-wave development candidate'
    scene['metadata']['r2_transfer']['used_as_unmodified_law']=parameters['r2_rate_scale']==1.
    return scene


def primary_error(assessment):
    return assessment.get('search_primary_offset_s',assessment.get('primary_offset_s'))


def primary_aligned(assessment):
    error=primary_error(assessment)
    return error is not None and abs(error)<=assessment.get('primary_search_tolerance_s',20.)


def censored_primary_late(row):
    """An expired first observed wave is evidence of lateness, not a time fit."""
    if row.get('status') == 'numerical-failed' or row.get('physical_error'):
        return False
    a=row.get('assessment',{})
    if primary_error(a) is not None:
        return False
    for key in ('pulse_match','arrival_match'):
        match=a.get(key,{})
        # Only the first due group can establish absence of the first wave.
        missing=match.get('missing',[])
        if missing and match.get('matched',0)==0:
            return True
    return a.get('last_observed_s',0.)>a.get('primary_deadline_s',1910.)


def sequence_rank(row):
    a=row.get('assessment',{})
    matches=[a.get(k,{})for k in ('pulse_match','arrival_match')]
    primary=primary_error(a)
    primary_good=primary_aligned(a)
    passed=sum(m.get('matched',0)for m in matches)
    delivery=row.get('delivery') or {}
    heating=delivery.get('top4_neighbor_mean_k',-1e6) if not (
        row.get('physical_error') or row.get('temperature_rejection')) else -1e6
    return (bool(a.get('accepted')),primary_good,passed,
            heating,
            -(abs(primary)if primary is not None else 1e6),
            -a.get('timing_error_s',1e6))


def suggest(rows, base, anchor=None):
    # Finer meshes inherit PARAMETERS, never scores or native restart states.
    start=dict(initial(base) if anchor is None else anchor)
    used=np.asarray([encode(r['parameters'])for r in rows])
    def distinct(p):return not len(used)or np.min(np.linalg.norm(used-encode(p),axis=1))>1e-4
    if not rows:
        return start,dict(method='fresh-mesh-anchor',scores_transferred=False)
    # A changed gas geometry/contact can shift the primary. Recenter THAT
    # family before moving on; an aligned old geometry is not its calibration.
    last=rows[-1];last_p=last['parameters'];offset=primary_error(last.get('assessment',{}))
    changed=any(abs(last_p[k]-v)>1e-10 for k,v in start.items() if k!='r2_rate_scale')
    if changed and offset is not None and abs(offset)>20. and last.get('status')!='numerical-failed':
        family=[r for r in rows if all(abs(r['parameters'][k]-v)<1e-10
                for k,v in last_p.items() if k!='r2_rate_scale')]
        bracket=[(a,b) for a in family for b in family
                 if (primary_error(a['assessment']) or 0)<0.<(primary_error(b['assessment']) or 0)
                 and a['parameters']['r2_rate_scale']>b['parameters']['r2_rate_scale']]
        if bracket:
            a,b=min(bracket,key=lambda pair:math.log(pair[0]['parameters']['r2_rate_scale']/pair[1]['parameters']['r2_rate_scale']))
            da,db=primary_error(a['assessment']),primary_error(b['assessment'])
            f=float(np.clip(-da/(db-da),.15,.85))
            rate=math.exp((1-f)*math.log(a['parameters']['r2_rate_scale'])+f*math.log(b['parameters']['r2_rate_scale']))
        else:rate=last_p['r2_rate_scale']*float(np.exp(np.clip(.006*offset,-1.,1.)))
        trial=dict(last_p,r2_rate_scale=float(np.clip(rate,.001,2.)))
        if distinct(trial):return trial,dict(method='recenter-primary-after-gas-change',source_trial=last['trial'])
    # First isolate the premature source with the retained geometry. This is
    # a labelled kinetic sensitivity bracket, never an ignition-time control.
    scalar=[r for r in rows if all(abs(r['parameters'][k]-v)<1e-10
                                  for k,v in start.items()if k!='r2_rate_scale')]
    aligned=[r for r in rows if primary_aligned(r.get('assessment',{}))]
    if not aligned:
        early=[r for r in scalar if primary_error(r['assessment'])is not None and
               primary_error(r['assessment'])<0.]
        late=[r for r in scalar if (primary_error(r['assessment'])or 0)>0. or censored_primary_late(r)]
        if early and late:
            pairs=[(a,b)for a in early for b in late if a['parameters']['r2_rate_scale']>b['parameters']['r2_rate_scale']]
            if pairs:
                a,b=min(pairs,key=lambda ab:math.log(ab[0]['parameters']['r2_rate_scale']/ab[1]['parameters']['r2_rate_scale']))
                x,y=math.log(a['parameters']['r2_rate_scale']),math.log(b['parameters']['r2_rate_scale'])
                da,db=primary_error(a['assessment']),primary_error(b['assessment'])
                f=.5 if db is None else float(np.clip(-da/(db-da),.15,.85))
                trial=dict(start,r2_rate_scale=math.exp(x+f*(y-x)))
                if distinct(trial):return trial,dict(method='measured-primary-bracket',
                    bracketing_trials=[a['trial'],b['trial']],late_censored=db is None)
        measured=[r for r in scalar if primary_error(r['assessment'])is not None]
        if measured:
            best=min(measured,key=lambda r:abs(primary_error(r['assessment'])))
            factor=float(np.exp(np.clip(.006*primary_error(best['assessment']),-3.9,1.6)))
            trial=dict(start,r2_rate_scale=float(np.clip(best['parameters']['r2_rate_scale']*factor,.001,2.)))
            if distinct(trial):return trial,dict(method='expand-primary-rate-bracket')
        elif late:
            best=max(late,key=lambda r:r['parameters']['r2_rate_scale'])
            trial=dict(start,r2_rate_scale=min(2.,best['parameters']['r2_rate_scale']*5.))
            if distinct(trial):return trial,dict(method='censored-primary-rate-bracket')
    center=max(rows,key=sequence_rank)if rows else None
    if aligned:
        center=max(aligned,key=sequence_rank)
        for contact in (2.,4.,8.,.5):
            trial=dict(center['parameters'],gas_can_contact_scale=contact)
            if distinct(trial):return trial,dict(method='shared-gas-can-delivery-bracket',source_trial=center['trial'])
        for key,value in [('cross_column_open_fraction',.4),('axial_open_fraction',.4),
                          ('thermal_stirring_coefficient',.15),('thermal_stirring_coefficient',.03)]:
            trial=dict(center['parameters']);trial[key]=value
            if distinct(trial):return trial,dict(method='bounded-shared-route-sensitivity',source_trial=center['trial'])
    observations=[]
    for r in rows:
        a=r.get('assessment',{})
        observations.append(dict(x=encode(r['parameters']).tolist(),
            complete=bool(a.get('phase_window_complete')and a.get('accepted')),
            objective=a.get('objective'),constraint=a.get('constraint',4.),wall_s=r.get('wall_s',0.)))
    x,how=propose(observations,len(SPACE),encode(center['parameters']if center else start),
                  seed=20261026,initial_design=4,admissible=lambda x:distinct(decode(x)))
    how['objective_scope']='All-wave gate; rejected prefixes provide constraints, not invented final scores'
    return decode(x),how
