"""Bounded fixture-level cooling parameters, not chemistry or material fits.

Fixed-window residuals guide proposals. A pruned prefix is never labelled a
complete experiment, and only a full trajectory can satisfy convergence.
"""
import numpy as np
from bounded_bayes import propose
from fsri30_objective import model_fields, targets
from fsri30_radiation import add_exterior_radiation
from fsri30_search_model import Parameter

SPACE=(
    Parameter('effective_emissivity',.02,.95,.1,False,
        'Unmeasured exterior grey-body exchange, shared by all ABS faces'),
    Parameter('thermal_stirring_coefficient',.01,.5,.05,True,
        'Existing conservative free-gas mixing closure; fixed 20mm physical mixing length'),
)
ANCHOR_END=3720.


def candidate(base,parameters):
    for p in SPACE:p.encode(parameters[p.name])
    scene=add_exterior_radiation(base,parameters['effective_emissivity'])
    scene['gas_network']['thermal_stirring_coefficient']=parameters['thermal_stirring_coefficient']
    scene['metadata']['cooling_search_parameters']=dict(parameters)
    scene['metadata']['cooling_search_scope']='Fixture boundary and conservative gas mixing only; chemistry/residue/materials/numerics frozen'
    return scene


def encode(parameters):return np.array([p.encode(parameters[p.name])for p in SPACE])
def decode(x):return {p.name:p.decode(float(v))for p,v in zip(SPACE,x)}


def observation(run,data):
    t,_,v=model_fields(run);target=targets(data)
    result=dict(last_observed_s=float(t[-1]),full_trajectory=bool(t[-1]>=5945.-1e-7 and not run.get('partial',False)))
    if t[-1]<ANCHOR_END-1e-7:return result
    mask=(target['time']>=3600.)&(target['time']<=ANCHOR_END)
    grid=target['time'][mask];q=np.quantile(v,[0.,.5,.9,1.],axis=1).T
    pred=np.stack([np.interp(grid,t,q[:,i])for i in range(4)],axis=1)
    error=pred-target['q'][mask]
    # Exactly the SAME observed anchor in every trial. Not a guessed full score.
    result.update(anchor_end_s=ANCHOR_END,anchor_rmse_k=float(np.sqrt(np.mean(error**2))),
        anchor_residual_vector_k=error.ravel().tolist())
    return result


def converged(run,data,*,converted_components):
    obs=observation(run,data)
    if not obs['full_trajectory']:return dict(accepted=False,reason='Full fresh trajectory required')
    if converted_components<2:return dict(accepted=False,reason='No autonomous cascade')
    t,_,v=model_fields(run);target=targets(data)
    mask=target['time']>=3600.;truth=target['q'][mask]
    q=np.quantile(v,[0.,.5,.9,1.],axis=1).T
    pred=np.stack([np.interp(target['time'][mask],t,q[:,i])for i in range(4)],axis=1)
    error=pred-truth;rmse=float(np.sqrt(np.mean(error**2)))
    limit=max(80.,.1*float(np.sqrt(np.mean(truth**2))))
    median_relative=float(np.sqrt(np.mean((error[:,1]/truth[:,1])**2)))
    accepted=bool(rmse<=limit and median_relative<=.05)
    return dict(accepted=accepted,cooling_rmse_k=rmse,cooling_limit_k=limit,
        median_rms_relative_kelvin=median_relative,median_limit=.05,
        scope='Engineering cooling-shape fit only, not matched-sensor or independent validation')


def rank(row):
    # Longer non-rejected evidence outranks an artificially cheap short prefix.
    metric=row.get('full_cooling_rmse_k') if row.get('full_trajectory') else row.get('anchor_rmse_k')
    return (bool(row.get('full_trajectory')),row.get('last_observed_s',0.),
        -(metric if metric is not None else 1e9))


def suggest(rows,initial_epsilon):
    used=np.array([encode(r['parameters'])for r in rows]) if rows else np.empty((0,2))
    def distinct(p):return not len(used) or np.min(np.linalg.norm(used-encode(p),axis=1))>.01
    first=dict(effective_emissivity=initial_epsilon,thermal_stirring_coefficient=.05)
    if distinct(first):return first,dict(method='measured-common-window-radiation-secant')
    anchored=[r for r in rows if 'anchor_residual_vector_k'in r]
    scalar=[r for r in anchored if abs(r['parameters']['thermal_stirring_coefficient']-.05)<1e-12]
    # Correct the direction from ACTUAL observed temperature changes, rather
    # than extending an arbitrary grid after finding an improving direction.
    if len(scalar)>=2 and len(scalar)<4:
        a,b=sorted(scalar,key=lambda r:r['parameters']['effective_emissivity'])[-2:]
        da=b['parameters']['effective_emissivity']-a['parameters']['effective_emissivity']
        d=(np.array(b['anchor_residual_vector_k'])-a['anchor_residual_vector_k'])/da
        denom=float(d@d)
        if denom>1e-12:
            value=b['parameters']['effective_emissivity']-float(d@np.array(b['anchor_residual_vector_k']))/denom
            p=dict(first,effective_emissivity=float(np.clip(value,SPACE[0].lower,SPACE[0].upper)))
            if distinct(p):return p,dict(method='updated-measured-radiation-secant',unbounded_estimate=value)
    best=min(anchored,key=lambda r:r['anchor_rmse_k'])if anchored else rows[0]
    e=best['parameters']['effective_emissivity']
    # Existing gas mixing is the second direction if a single sink coefficient
    # cannot fit hot and cold branches simultaneously. No new mechanism.
    for mix in (.15,.3,.02):
        p=dict(effective_emissivity=e,thermal_stirring_coefficient=mix)
        if distinct(p):return p,dict(method='shared-mixing-bracket-at-best-observed-radiation')
    observations=[]
    for r in rows:
        observations.append(dict(x=encode(r['parameters']).tolist(),complete=r.get('full_trajectory',False),
            objective=r.get('full_cooling_rmse_k'),constraint=r.get('constraint',1.),wall_s=r.get('wall_s',0.)))
    x,how=propose(observations,2,encode(first),initial_design=3,seed=20261025,
        admissible=lambda x:distinct(decode(x)))
    return decode(x),how
