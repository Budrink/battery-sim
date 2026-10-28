#!/usr/bin/env python3
"""A-priori representability of fine fields by coarse means + three slopes.

Coefficients here are ORACLE fits to the known fine state at each snapshot,
NOT predictions from a coarse solver. This cannot qualify a dynamic closure.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_fsri30_bayes import atomic_save
from fsri30_data import sha
from fsri30_spatial_patterns import overlap_restriction,read_fields


def linear_reconstruction(values,coordinates,weights):
    """Weighted zero-mean linear modes and a mean-preserving positivity limit.

    The limit applies at sampled fine-cell centers only. A future face-flux
    reconstruction still needs its own positivity/stability checks.
    """
    values=np.asarray(values,dtype=float);coordinates=np.asarray(coordinates,dtype=float)
    weights=np.asarray(weights,dtype=float)
    if values.ndim!=2 or coordinates.shape!=(values.shape[1],3) or weights.shape!=(values.shape[1],) or np.any(weights<=0):
        raise ValueError('Invalid weighted subcell geometry')
    w=weights/weights.sum();z=coordinates-(w[:,None]*coordinates).sum(axis=0)
    projector=z@np.linalg.pinv(z.T@(z*w[:,None]),rcond=1e-12)@(z.T*w[None,:])
    mean=values@w;deviation=values-mean[:,None]
    fitted=deviation@projector.T
    # Remove last-bit mean leakage; never alter the physical coarse inventory.
    fitted-=(fitted@w)[:,None]
    minimum=fitted.min(axis=1);theta=np.ones(len(values));negative=mean+minimum<0.
    theta[negative]=np.maximum(0.,mean[negative])/(-minimum[negative])
    reconstructed=mean[:,None]+fitted
    limited=mean[:,None]+fitted*theta[:,None]
    return reconstructed,limited,deviation


def analyze(coarse,fine):
    a=json.loads((coarse/'scenario.json').read_text());b=json.loads((fine/'scenario.json').read_text())
    ga,gb=(s['metadata']['gas_volume_geometry'] for s in (a,b))
    ca=[(i,c['id']) for i,c in enumerate(a['gas_network']['compartments']) if c['id'] in ga]
    fb=[(i,c['id']) for i,c in enumerate(b['gas_network']['compartments']) if c['id'] in gb]
    tr,er,cv,fv=overlap_restriction([ga[k] for _,k in ca],[gb[k] for _,k in fb])
    frames=read_fields(fine/'gas-fields.bin');fi=np.array([i for i,_ in fb])
    centers=np.array([(np.array(gb[k]['low_m'])+np.array(gb[k]['high_m']))/2 for _,k in fb])
    result={}
    for phase,start,end in (('first_burst',1105.,1230.),('held_out_cascade',2200.,2525.)):
        index=np.flatnonzero((frames['time']>=start-1e-7)&(frames['time']<=end+1e-7))
        if len(index)<2 or frames['time'][index[-1]]<end-1e-7:raise ValueError('Missing requested observed phase')
        fields=np.asarray(frames['fields'][index][:,fi,:],dtype=float)
        fields[:,:,1:]/=fv[None,:,None]
        stats={label:dict(variance=0.,residual=0.,limited_residual=0.,negative_samples=0,observed_samples=0) for label in ('temperature_k','fuel_density_j_m3','oxygen_density_kg_m3')}
        for parent,(_,name) in enumerate(ca):
            row=er.getrow(parent);js=row.indices;weights=row.data*fv[js]
            lo=np.array(ga[name]['low_m']);hi=np.array(ga[name]['high_m'])
            coordinates=(centers[js]-(lo+hi)/2)/(hi-lo)
            for field,label in enumerate(stats):
                value=fields[:,js,field];recon,limited,deviation=linear_reconstruction(value,coordinates,weights)
                item=stats[label];item['variance']+=float(np.sum(deviation**2*weights[None,:]))
                item['residual']+=float(np.sum((value-recon)**2*weights[None,:]))
                item['limited_residual']+=float(np.sum((value-limited)**2*weights[None,:]))
                threshold=1e-12*max(1e-12,float(value.max()))
                item['negative_samples']+=int(np.sum(recon < -threshold));item['observed_samples']+=recon.size
        result[phase]={label:dict(subcell_variance_captured=1.-s['residual']/s['variance'],
            subcell_variance_captured_after_sample_positivity_limit=1.-s['limited_residual']/s['variance'],
            negative_reconstruction_fraction=s['negative_samples']/s['observed_samples']) for label,s in stats.items()}
    return dict(status='oracle-linear-subcell-representability',all_coarse_air_volumes=len(ca),fine_air_volumes=len(fb),results=result,
        coarse_coefficients_predicted=False,physical_model_changed=False,patterns_promoted=False,
        representation='Mean plus three zero-mean spatial slopes per scalar; irregular and nested boxes included',
        limitation='Oracle coefficients from fine snapshots, not a dynamic coarse closure. Temperature is volume-weighted, not sensible energy. Sample-center positivity is not face-flux positivity.',
        source_sha256=sha(Path(__file__)),fine_fields_sha256=sha(fine/'gas-fields.bin'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coarse',required=True,type=Path);p.add_argument('--fine',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Refusing to overwrite a prior diagnostic')
    result=analyze(a.coarse,a.fine);atomic_save(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
