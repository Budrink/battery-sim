"""Read-only conservative restriction of fine gas fields to physical boxes.

Temperature is volume-averaged for this diagnostic, NOT presented as energy.
Fuel energy and oxygen mass are extensive: their restriction preserves totals.
Neither grid nor observation times are shifted to obtain agreement.
"""
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


def read_fields(path):
    path=Path(path)
    with path.open('rb') as stream:
        if stream.read(8)!=b'BATTGAS1':raise ValueError('Unknown gas field format')
        count=int.from_bytes(stream.read(8),'little')
    if count<=0:raise ValueError('Empty gas field')
    dtype=np.dtype([('time','<f8'),('fields','<f4',(count,3))])
    if (path.stat().st_size-16)%dtype.itemsize:raise ValueError('Truncated gas field frame')
    return np.memmap(path,mode='r',offset=16,dtype=dtype)


def overlap_restriction(coarse_boxes,fine_boxes):
    """Return temperature and extensive sparse restrictions plus box volumes.

    Reject holes, overlapping boxes and different domains rather than silently
    renormalizing away missing volume/mass. Axis-aligned physical intersections
    also work when the two rectilinear partitions are not exactly nested.
    """
    def bounds(boxes):
        low=np.asarray([b['low_m'] for b in boxes],dtype=float)
        high=np.asarray([b['high_m'] for b in boxes],dtype=float)
        if low.ndim!=2 or low.shape[1]!=3 or high.shape!=low.shape or not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError('Invalid three-dimensional gas boxes')
        volume=np.prod(high-low,axis=1)
        if np.any(high<=low):raise ValueError('Nonpositive gas box volume')
        return low,high,volume
    cl,ch,cv=bounds(coarse_boxes);fl,fh,fv=bounds(fine_boxes)
    rows=[];columns=[];values=[]
    for i,(low,high) in enumerate(zip(cl,ch)):
        overlap=np.prod(np.maximum(0.,np.minimum(fh,high)-np.maximum(fl,low)),axis=1)
        indices=np.flatnonzero(overlap>0.)
        rows.extend([i]*len(indices));columns.extend(indices.tolist());values.extend(overlap[indices].tolist())
    intersection=csr_matrix((values,(rows,columns)),shape=(len(cv),len(fv)))
    if not np.allclose(np.asarray(intersection.sum(axis=1)).ravel(),cv,rtol=1e-9,atol=1e-18):
        raise ValueError('Fine domain does not cover coarse boxes exactly')
    if not np.allclose(np.asarray(intersection.sum(axis=0)).ravel(),fv,rtol=1e-9,atol=1e-18):
        raise ValueError('Coarse domain does not cover fine boxes exactly')
    temperature=intersection.multiply((1./cv)[:,None]).tocsr()
    extensive=intersection.multiply((1./fv)[None,:]).tocsr()
    return temperature,extensive,cv,fv


def compare_spatial_fields(coarse_scene,fine_scene,coarse_path,fine_path,until_s):
    a,b=read_fields(coarse_path),read_fields(fine_path)
    ia=np.flatnonzero(a['time']<=until_s+1e-7);ib=np.flatnonzero(b['time']<=until_s+1e-7)
    if len(ia)<2 or len(ia)!=len(ib) or not np.allclose(a['time'][ia],b['time'][ib],atol=1e-7,rtol=0.):
        raise ValueError('Gas field observations do not align')
    ids={};indices={};geometry={}
    for name,scene,frames in (('coarse',coarse_scene,a),('fine',fine_scene,b)):
        ids[name]=[c['id'] for c in scene['gas_network']['compartments']]
        if len(ids[name])!=len(set(ids[name])) or len(ids[name])!=frames['fields'].shape[1]:
            raise ValueError('Gas field ordering cannot be matched to the scenario')
        indices[name]={ident:i for i,ident in enumerate(ids[name])}
        geometry[name]=scene['metadata']['gas_volume_geometry']
    ca=[ident for ident in ids['coarse'] if ident in geometry['coarse']]
    fb=[ident for ident in ids['fine'] if ident in geometry['fine']]
    other_a=set(ids['coarse'])-set(ca);other_b=set(ids['fine'])-set(fb)
    if other_a!=other_b:raise ValueError('Non-air physical gas compartments changed')
    tr,er,cv,fv=overlap_restriction([geometry['coarse'][x] for x in ca],[geometry['fine'][x] for x in fb])
    aix=np.array([indices['coarse'][x] for x in ca]);bix=np.array([indices['fine'][x] for x in fb])
    other=sorted(other_a)
    oi=np.array([indices['coarse'][x] for x in other],dtype=int)
    oj=np.array([indices['fine'][x] for x in other],dtype=int)
    squared_volume=0.;worst=(-1.,None);maximum_fuel=0.;maximum_oxygen=0.;maximum_projection_residual=0.
    internal_squared=0.;internal_count=0;internal_worst=(-1.,None)
    rows=[];worst_fuel=(-1.,None);worst_normalized_fuel=(-1.,None)
    for start in range(0,len(ia),16):
        # Bound memory on full-horizon reference files; mmap remains read-only.
        ax=np.asarray(a['fields'][ia[start:start+16]],dtype=float)
        bx=np.asarray(b['fields'][ib[start:start+16]],dtype=float)
        ac=ax[:,aix,:];bc=bx[:,bix,:]
        mapped=np.stack([(tr@bc[:,:,0].T).T,(er@bc[:,:,1].T).T,(er@bc[:,:,2].T).T],axis=2)
        relative=np.abs(ac[:,:,0]-mapped[:,:,0])/np.maximum(ac[:,:,0],mapped[:,:,0])
        squared_volume+=float(np.sum(relative**2*cv[None,:]))
        fi,ni=np.unravel_index(relative.argmax(),relative.shape)
        if relative[fi,ni]>worst[0]:
            worst=(float(relative[fi,ni]),dict(time_s=float(a['time'][ia[start+fi]]),coarse_node=ca[ni],
                coarse_k=float(ac[fi,ni,0]),restricted_fine_k=float(mapped[fi,ni,0])))
        chemical=np.sum(np.abs(ac[:,:,1]-mapped[:,:,1]),axis=1)/np.maximum(100.,np.maximum(ac[:,:,1].sum(axis=1),mapped[:,:,1].sum(axis=1)))
        oxygen=np.sum(np.abs(ac[:,:,2]-mapped[:,:,2]),axis=1)/np.maximum(1e-12,np.maximum(ac[:,:,2].sum(axis=1),mapped[:,:,2].sum(axis=1)))
        maximum_fuel=max(maximum_fuel,float(chemical.max()));maximum_oxygen=max(maximum_oxygen,float(oxygen.max()))
        for field in (1,2):
            reference=bc[:,:,field].sum(axis=1)
            residual=np.abs(mapped[:,:,field].sum(axis=1)-reference)/np.maximum(1e-12,np.abs(reference))
            maximum_projection_residual=max(maximum_projection_residual,float(residual.max()))
        if len(other):
            ir=np.abs(ax[:,oi,0]-bx[:,oj,0])/np.maximum(ax[:,oi,0],bx[:,oj,0])
            internal_squared+=float(np.sum(ir**2));internal_count+=ir.size
            fi2,ni2=np.unravel_index(ir.argmax(),ir.shape)
            if ir[fi2,ni2]>internal_worst[0]:
                internal_worst=(float(ir[fi2,ni2]),dict(time_s=float(a['time'][ia[start+fi2]]),node=other[ni2],
                    coarse_k=float(ax[fi2,oi[ni2],0]),fine_k=float(bx[fi2,oj[ni2],0])))
        for k in range(len(ac)):
            total_a=float(ac[k,:,1].sum());total_b=float(mapped[k,:,1].sum())
            normalized=None
            if min(total_a,total_b)>=100.:
                normalized=float(.5*np.sum(np.abs(ac[k,:,1]/total_a-mapped[k,:,1]/total_b)))
                if normalized>worst_normalized_fuel[0]:
                    worst_normalized_fuel=(normalized,dict(time_s=float(a['time'][ia[start+k]]),
                        coarse_fuel_j=total_a,fine_fuel_j=total_b))
            if chemical[k]>worst_fuel[0]:
                worst_fuel=(float(chemical[k]),dict(time_s=float(a['time'][ia[start+k]]),
                    coarse_fuel_j=total_a,fine_fuel_j=total_b,
                    absolute_fuel_difference_l1_j=float(np.sum(np.abs(ac[k,:,1]-mapped[k,:,1])))))
            rows.append(dict(time_s=float(a['time'][ia[start+k]]),
                air_volume_rms_relative_kelvin=float(np.sqrt(np.sum(relative[k]**2*cv)/cv.sum())),
                air_maximum_local_relative_kelvin=float(relative[k].max()),
                air_fuel_l1_relative=float(chemical[k]),air_oxygen_l1_relative=float(oxygen[k]),
                coarse_air_fuel_j=total_a,fine_air_fuel_j=total_b,
                normalized_fuel_distribution_total_variation=normalized))
    return dict(status='spatial-patterns-compared',until_s=float(a['time'][ia[-1]]),
        snapshots=len(ia),coarse_air_nodes=len(ca),fine_air_nodes=len(fb),same_physical_non_air_nodes=len(other),
        air_volume_rms_relative_kelvin=float(np.sqrt(squared_volume/(cv.sum()*len(ia)))),
        maximum_air_local_relative_kelvin=worst[0],worst_air=worst[1],
        maximum_air_fuel_l1_relative=maximum_fuel,maximum_air_oxygen_l1_relative=maximum_oxygen,
        worst_air_fuel=worst_fuel[1],maximum_normalized_fuel_distribution_total_variation=worst_normalized_fuel[0] if worst_normalized_fuel[1] else None,
        worst_normalized_fuel_distribution=worst_normalized_fuel[1],minimum_fuel_per_mesh_for_normalized_pattern_j=100.,
        maximum_extensive_projection_relative_residual=maximum_projection_residual,
        same_node_rms_relative_kelvin=float(np.sqrt(internal_squared/internal_count)) if internal_count else None,
        same_node_maximum_local_relative_kelvin=internal_worst[0] if internal_count else None,
        worst_same_node=internal_worst[1],history=rows,
        temperature_weighting='Physical overlap volume; NOT sensible energy or a mass-weighted temperature',
        extensive_weighting='Physical overlap/fine volume: preserves fuel joules and oxygen kg',
        scope='Observed field redistribution diagnostic, no fabricated heat/mass fluxes or fitted time shifts')


def event_patterns(runs,events,source='B10',tolerance=.05):
    """Decompose raw arrival errors and event-aligned response shapes.

    Event alignment is explicitly diagnostic and cannot change the unshifted
    spatial acceptance tests. A held-out time predictor is NOT a new solver.
    """
    by={mm:{e['component']:e['time_s'] for e in events[mm]} for mm in (20,10)}
    shared=sorted(by[20].keys()&by[10].keys(),key=lambda k:(by[20][k],k))
    if source not in shared:raise ValueError('Source conversion event missing')
    times={mm:np.asarray(runs[mm]['time_s'],dtype=float) for mm in (20,10)}
    rows=[];tau=np.arange(-60.,120.+1e-9,5.)
    for ident in shared:
        row=dict(component=ident,coarse_event_s=by[20][ident],fine_event_s=by[10][ident],
            raw_delay_s=by[10][ident]-by[20][ident])
        if all(by[mm][ident]+tau[0]>=times[mm][0] and by[mm][ident]+tau[-1]<=times[mm][-1] for mm in (20,10)):
            curves={mm:np.interp(by[mm][ident]+tau,times[mm],runs[mm]['cell_mean_temperature_k'][ident]) for mm in (20,10)}
            x,y=curves[20],curves[10]
            row.update(event_aligned_rms_relative_kelvin=float(np.sqrt(np.mean(((x-y)/np.maximum(x,y))**2))),
                event_aligned_maximum_relative_kelvin=float(np.max(np.abs(x-y)/np.maximum(x,y))),
                coarse_peak_k=float(x.max()),fine_peak_k=float(y.max()),
                event_aligned_shape_passes=bool(np.sqrt(np.mean(((x-y)/np.maximum(x,y))**2))<=tolerance))
        rows.append(row)
    train=[r for r in rows if r['component']!=source]
    predictor=dict(status='not-enough-events',promoted=False)
    if len(train)>=6:
        # First chronological half only. Predict the remaining cells without
        # fitting their arrival times. Never use this to time-shift acceptance.
        split=len(train)//2;fit_rows=train[:split];test_rows=train[split:]
        origin={mm:by[mm][source] for mm in (20,10)}
        x=np.array([r['coarse_event_s']-origin[20] for r in fit_rows]);y=np.array([r['fine_event_s']-origin[10] for r in fit_rows])
        denominator=float(x@x)
        if denominator>0.:
            scale=float((x@y)/denominator)
            held=[]
            for r in test_rows:
                guess=origin[10]+scale*(r['coarse_event_s']-origin[20])
                held.append(dict(component=r['component'],predicted_fine_event_s=guess,
                    observed_fine_event_s=r['fine_event_s'],residual_s=guess-r['fine_event_s']))
            predictor=dict(status='diagnostic-fit',scale=scale,training_components=[r['component'] for r in fit_rows],
                held_out=held,held_out_rmse_s=float(np.sqrt(np.mean([r['residual_s']**2 for r in held]))),
                held_out_raw_delay_rmse_s=float(np.sqrt(np.mean([r['raw_delay_s']**2 for r in test_rows]))),
                promoted=False,scope='Arrival-law hypothesis only, NOT physical flux correction or a solver validation')
    return dict(events=rows,shared_events=len(rows),missing_on_fine=sorted(by[20].keys()-by[10].keys()),
        missing_on_coarse=sorted(by[10].keys()-by[20].keys()),
        shape_window_relative_to_own_event_s=[float(tau[0]),float(tau[-1])],
        shape_metric_is_diagnostic_only=True,time_predictor=predictor,patterns_promoted=False,
        limitation='Fields alone do not uniquely identify flux/source terms. Any learned closure needs conservative replay and held-out layout checks.')
