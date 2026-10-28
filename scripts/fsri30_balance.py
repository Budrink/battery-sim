"""Measured hot/cold bracketing with a provisional isochronous burst response.

Partial tails cannot supply a full-tail sign. Predicted timing is only a
proposal: every candidate still passes the unchanged actual-CUDA burst gate.
"""
import math
import numpy as np
from fsri30_cooling_fit import SPACE,encode,decode,burst_rank

KEYS=('peak_time_s','rise50_time_s','rise90_time_s')


def admissible(row):
    b=row.get('burst',{})
    return bool(b.get('accepted')and b.get('phase_window_complete')and b.get('timing_target_met'))


def timing_compensation(rows,epsilon,center):
    x0=encode(center['parameters']);b=center['burst']
    y0=np.array([b['timing_offsets_s'][k]for k in KEYS])
    samples={tuple(encode(r['parameters'])):r for r in rows if r.get('burst',{}).get('accepted')and
        r['burst'].get('phase_window_complete')and all(r['burst'].get('timing_offsets_s',{}).get(k)is not None for k in KEYS)}
    # Prefer an ACTUAL same-boundary timing bracket over a broad regression.
    # A global fit previously proposed less mixing than the measured early
    # endpoint after its neighbour was late: it discarded the strongest evidence.
    same=[r for r in samples.values()if abs(r['parameters']['effective_emissivity']-epsilon)<1e-8]
    pairs=[]
    for a in same:
        ya=np.array([a['burst']['timing_offsets_s'][k]for k in KEYS])
        for c in same:
            yc=np.array([c['burst']['timing_offsets_s'][k]for k in KEYS])
            if np.mean(ya)>=0. or np.mean(yc)<=0.:continue
            ma=a['parameters']['thermal_stirring_coefficient'];mc=c['parameters']['thermal_stirring_coefficient']
            if abs(math.log(mc/ma))<1e-8:continue
            pairs.append((abs(math.log(mc/ma)),a,c,ya,yc))
    if pairs:
        _,a,c,ya,yc=min(pairs,key=lambda p:p[0]);delta=yc-ya
        fraction=float(np.clip(-float(delta@ya)/float(delta@delta),.15,.85))
        ma=a['parameters']['thermal_stirring_coefficient'];mc=c['parameters']['thermal_stirring_coefficient']
        mixing=math.exp(math.log(ma)+fraction*math.log(mc/ma))
        forecast=ya+fraction*delta
        return dict(effective_emissivity=epsilon,thermal_stirring_coefficient=mixing),dict(
            timing_prediction_available=True,method='measured-same-boundary-phase-secant',
            phase_bracket_trials=[a['trial'],c['trial']],phase_bracket_mixing=[ma,mc],fraction=fraction,
            predicted_timing_offsets_s=dict(zip(KEYS,forecast.tolist())),
            proposal_only=True,actual_burst_gate_required=True)
    x=x0.copy();x[0]=SPACE[0].encode(epsilon)
    if len(samples)<3:
        return decode(x),dict(timing_prediction_available=False,reason='Insufficient independent phase measurements')
    X=np.array([encode(r['parameters'])-x0 for r in samples.values()])
    Y=np.array([[r['burst']['timing_offsets_s'][k]for k in KEYS]for r in samples.values()])-y0
    weights=np.exp(-np.sum((X/.3)**2,axis=1))
    if np.linalg.matrix_rank(X)<2:
        return decode(x),dict(timing_prediction_available=False,reason='Phase response has no two-dimensional support')
    J=np.linalg.solve(X.T@(weights[:,None]*X)+np.eye(2)*1e-5,X.T@(weights[:,None]*Y))
    before=y0+(x[0]-x0[0])*J[0];denom=float(J[1]@J[1])
    if denom<=1e-10:return decode(x),dict(timing_prediction_available=False,reason='Unresolved mixing sensitivity')
    delta=-float(J[1]@before)/denom
    # Bounded proposal, NOT a temperature/physics clamp. No more than a factor
    # of two in stirring from the measured admissible center in one proposal.
    radius=math.log(2.)/math.log(SPACE[1].upper/SPACE[1].lower)
    x[1]=float(np.clip(x0[1]+delta,max(0.,x0[1]-radius),min(1.,x0[1]+radius)))
    forecast=before+(x[1]-x0[1])*J[1]
    return decode(x),dict(timing_prediction_available=True,predicted_timing_offsets_s=dict(zip(KEYS,forecast.tolist())),
        derivative_samples=len(samples),proposal_only=True,actual_burst_gate_required=True)


def suggest_balance(rows):
    full=[r for r in rows if admissible(r)and r.get('full_trajectory')and 'full_median_bias_k'in r]
    if not full:return None
    # A repeated exact-core replay must not reweight the inferred response.
    full=list({tuple(encode(r['parameters'])):r for r in full}.values())
    center=min(full,key=lambda r:abs(r['full_median_bias_k']))
    if abs(center['full_median_bias_k'])<5.:return None
    hot=[r for r in full if r['full_median_bias_k']>0.]
    cold=[r for r in full if r['full_median_bias_k']<0.]
    pairs=[(a,b)for a in hot for b in cold if a['parameters']['effective_emissivity']<b['parameters']['effective_emissivity']-.01]
    how=dict(objective_scope='Measured full-tail hot/cold balance ONLY inside the target burst band',
        center_trial=center['trial'],center_full_median_bias_k=center['full_median_bias_k'])
    if pairs:
        a,b=min(pairs,key=lambda ab:ab[1]['parameters']['effective_emissivity']-ab[0]['parameters']['effective_emissivity'])
        ea,eb=(r['parameters']['effective_emissivity']for r in (a,b))
        weight=a['full_median_bias_k']/(a['full_median_bias_k']-b['full_median_bias_k'])
        epsilon=ea+float(np.clip(weight,.2,.8))*(eb-ea)
        how.update(method='measured-full-tail-safeguarded-secant',hot_trial=a['trial'],cold_trial=b['trial'],
            hot_bias_k=a['full_median_bias_k'],cold_bias_k=b['full_median_bias_k'])
    else:
        edge=SPACE[0].lower if center['full_median_bias_k']<0. else SPACE[0].upper
        epsilon=.5*(center['parameters']['effective_emissivity']+edge)
        how.update(method='expand-full-tail-sign-bracket',
            note='Opposite full-tail sign has NOT yet been measured with the target burst; test an intermediate boundary')
    if abs(epsilon-center['parameters']['effective_emissivity'])<.008:return None
    params,prediction=timing_compensation(rows,epsilon,center);how.update(prediction)
    same=[r for r in rows if abs(r['parameters']['effective_emissivity']-epsilon)<1e-8]
    if same:
        # If the proposed boundary has a measured, settled burst, correct its
        # timing locally rather than repeating the identical full experiment.
        settled=[r for r in same if r.get('burst',{}).get('phase_window_complete')and
            r['burst'].get('converted_components',0)>=2 and
            all(r['burst'].get('timing_offsets_s',{}).get(k)is not None for k in KEYS)]
        if settled:
            latest=settled[-1];params,prediction=timing_compensation(rows,epsilon,latest)
            how.update(prediction,method='correct-mixing-at-measured-intermediate-radiation',timing_source=latest['trial'])
    used=np.array([encode(r['parameters'])for r in rows]);distance=float(np.min(np.linalg.norm(used-encode(params),axis=1)))
    if distance<.002:return None
    how['nearest_previous_distance']=distance
    return params,how
