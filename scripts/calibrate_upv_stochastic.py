#!/usr/bin/env python3
"""Training-only, common-seed barrier search. CUDA evolves all states.

Physical thermal inputs and chemistry stay frozen. Only shell discretization
and its explicitly conservative initial-field remap differ from the old trial.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

import calibrate_upv_permeability as protocol
import calibrate_upv_mass_paths as fit
import calibrate_upv_shell_strength as strength
import upv_mass_paths as geometry
from calibrate_upv import save

OUT = geometry.ROOT/'data/calibration/upv-stochastic-v23'
GRID = (6,48,96)
SEEDS = (17,29,43)
AUDIT_SEEDS = (101,211,307)
BASE = dict(pressure_pa=1e6, rate_per_s=20., pressure_exponent=1.)
CASES = [
    ('baseline', BASE),
    ('weak-fast', dict(pressure_pa=2e5,rate_per_s=2000.,pressure_exponent=1.)),
    ('strong-fast', dict(pressure_pa=5e6,rate_per_s=2000.,pressure_exponent=1.)),
    ('strong-slow', dict(pressure_pa=5e6,rate_per_s=.02,pressure_exponent=1.)),
    ('weak-slow', dict(pressure_pa=2e5,rate_per_s=.02,pressure_exponent=1.)),
    ('quadratic', dict(pressure_pa=1e6,rate_per_s=20.,pressure_exponent=2.)),
]
BOUNDS = dict(pressure_pa=(2e5,5e6),rate_per_s=(.02,2000.),pressure_exponent=(1.,2.))


def physical_signature(scene):
    s=copy.deepcopy(scene)
    for shell in s['shells']:
        shell.pop('surface_divisions',None)
        shell.pop('initial_temperature_fields_k',None)
    return geometry.thermal_signature(s)


def remap_uniform_field(values,old,new):
    """Volume-overlap average of a cell-constant field, x-fastest storage.

    Supports both refinement and coarsening without inventing heat. This
    averages enthalpy only for the frozen, spatially constant casing cp.
    """
    if any(not isinstance(n,(int,np.integer)) or n<1 for n in (*old,*new)):
        raise ValueError('Grid counts must be positive integers')
    field=np.asarray(values,dtype=float).reshape(tuple(reversed(old)))
    for axis,(before,after) in enumerate(zip(old,new)):
        left=np.maximum(np.arange(after)[:,None]/after,np.arange(before)[None,:]/before)
        right=np.minimum((np.arange(after)[:,None]+1)/after,(np.arange(before)[None,:]+1)/before)
        weights=np.maximum(0.,right-left)*after
        moved=np.moveaxis(field,2-axis,0)
        field=np.moveaxis(np.tensordot(weights,moved,axes=(1,0)),0,2-axis)
    return field.ravel().tolist()


def refine_shells(scene,grid):
    """Remap casing fields; grid=None matches each contents grid by face."""
    s=copy.deepcopy(scene);cells={c['id']:c for c in s['cells']};remap=[]
    for sh in s['shells']:
        old_grid=sh.get('surface_divisions',cells[sh['contents_id']]['thermal']['divisions'])
        target=list(grid) if grid is not None else list(cells[sh['contents_id']]['thermal']['divisions'])
        sh['surface_divisions']=target
        for face,values in sh['initial_temperature_fields_k'].items():
            axis='xyz'.index(face[1]);old=list(old_grid);new=list(target);old[axis]=new[axis]=1
            a=np.asarray(values);b=np.asarray(remap_uniform_field(values,old,new))
            delta=float(np.mean(b)-np.mean(a))
            if abs(delta)>1e-10:
                raise ValueError('Shell remap changed mean enthalpy for constant shell cp')
            sh['initial_temperature_fields_k'][face]=b.ravel().tolist()
            remap.append(dict(panel=sh['contents_id']+':'+face,old=old,new=new,mean_temperature_change_k=delta))
    if physical_signature(s)!=physical_signature(scene):
        raise ValueError('Physical thermal inputs changed during refinement')
    s['metadata']['shell_remap']=dict(method='Uniform-volume overlap average, conservative for fixed casing cp',panels=remap)
    return s


def build(scene,p,seed,duration=30.,grid=GRID):
    if set(p)!=set(BOUNDS):
        raise ValueError('Only stochastic barrier parameters may change')
    for k,(lo,hi) in BOUNDS.items():
        if not np.isfinite(p[k]) or not lo<=p[k]<=hi:
            raise ValueError('Invalid parameter '+k)
    if not isinstance(seed,int) or seed<0:
        raise ValueError('Invalid seed')
    s=refine_shells(geometry.attach(scene,geometry.initial()),grid)
    for ch in s['gas_network']['channels']:
        if ch['kind']!='rupture':continue
        ch['opening_pressure_pa']=p['pressure_pa']
        old=ch['damage_opening']
        ch['damage_opening']={k:old[k] for k in ('seam_width_m','seam_strength_factor','fracture_energy_j_per_m2')}
        ch['damage_opening'].update(mode='stochastic_pixels',seed=seed,pixel_rate_per_s=p['rate_per_s'],pixel_pressure_exponent=p['pressure_exponent'])
    times=[t for t in s['failure_sample_times_s'] if t<=duration]
    times += [float(t) for t in np.arange(0,min(5.,duration),.02)]+[duration]
    s['failure_sample_times_s']=sorted(set(times));s['simulation']['duration_s']=duration
    s['metadata']['stochastic_trial']=dict(parameters=p,seed=seed,grid=list(grid),thermal_physics_sha256=physical_signature(scene))
    return s


def replay(runner,scene,prior,p,seed,duration=30.,refinement=1.,grid=GRID):
    candidate=build(scene,p,seed,duration,grid)
    candidate['simulation']['stability_factor']*=refinement
    run=runner.run(candidate,refinement)
    if run.get('max_oxidizer_residual_kg',0)>1e-10:raise ValueError('Oxidizer balance failed')
    return dict(parameters=p,seed=seed,grid=list(grid),**protocol.summarize(run,candidate,prior),result=run,
                meta=prior['meta'],observed_temperature_k=prior['observed_temperature_k'],observed_pressure_bar=prior['observed_pressure_bar']),candidate


def ensemble(reports):
    scores=[r['score']['objective'] for r in reports]
    return dict(mean_objective=float(np.mean(scores)),std_objective=float(np.std(scores)),
        mean_pressure_rmse_bar=float(np.mean([r['metrics']['pressure_0_30_rmse_bar'] for r in reports])),
        peaks_bar=[r['features']['sampled_chamber_peak_bar'] for r in reports],
        source_peaks_bar=[r['features']['sampled_max_source_pressure_bar'] for r in reports],
        seeds=[r['seed'] for r in reports])


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode',choices=['smoke','scan','audit','validate'])
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()+Path(protocol.__file__).read_bytes()+fit.fingerprint(fit.PROBE).encode()).hexdigest()
    manifest=dict(fingerprint=fingerprint,cases=CASES,seeds=SEEDS,audit_seeds=AUDIT_SEEDS,grid=GRID,bounds=BOUNDS,
        objective='Mean per-realization pressure score at original measurement times, equal normalized 0..3s and 3..30s windows; temperatures reported, never refitted',
        frozen='Thermal properties/contacts, casing thickness, initial mean enthalpy, chemistry, gas geometry, Cd, fracture work, seam ratio',
        changed='Stochastic barrier law; independent fine shell grid with conservative initial-field remap',
        validation_used_for_selection=False,calibration_accepted=False)
    m=args.output/'manifest.json'
    if m.exists() and geometry.digest(geometry.read(m))!=geometry.digest(manifest):
        raise ValueError('Changed model/protocol: use a fresh output folder')
    save(m,manifest);runner=fit.native.Runner(fit.PROBE,args.output)
    def get(stem,scene,prior,p,seed,**kw):
        path=args.output/(stem+'.json')
        if path.exists():return geometry.read(path)
        print('starting',stem,p,'seed',seed,kw,flush=True)
        report,candidate=replay(runner,scene,prior,p,seed,**kw)
        save(path,report);save(args.output/(stem+'-scenario.json'),candidate)
        print('completed',stem,report['score'],report['features'],'steps',report['result']['steps'],'wall_s',report['result']['wall_s'],flush=True)
        return report
    try:
        scene,prior=geometry.inputs('no-cooling')
        if args.mode=='smoke':
            get('baseline-seed17',scene,prior,BASE,SEEDS[0]);return
        if args.mode=='scan':
            history=[]
            for name,p in CASES:
                reports=[get(f'{name}-seed{seed}',scene,prior,p,seed) for seed in SEEDS]
                history.append(dict(id=name,parameters=p,**ensemble(reports)))
                save(args.output/'history.json',history);print('ensemble',history[-1],flush=True)
            best=min(history,key=lambda r:r['mean_objective'])
            save(args.output/'selection.json',dict(best,fingerprint=fingerprint,validation_used_for_selection=False,
                relative_improvement=1-best['mean_objective']/history[0]['mean_objective'],calibration_accepted=False));return
        selected=geometry.read(args.output/'selection.json')
        if selected['fingerprint']!=fingerprint:raise ValueError('Stale selection')
        name,p=selected['id'],selected['parameters']
        if args.mode=='audit':
            rows=[]
            for seed in SEEDS:
                a=get(f'{name}-seed{seed}',scene,prior,p,seed)
                b=get(f'{name}-seed{seed}-fine',scene,prior,p,seed,refinement=.5)
                delta=protocol.refinement_metrics(a,b)
                rows.append(dict(seed=seed,**delta,passed=protocol.refinement_passes(delta)))
                save(args.output/'numerics.json',dict(rows=rows,passed=all(r['passed'] for r in rows),calibration_accepted=False))
                print('numerics',rows[-1],flush=True)
            independent=[get(f'audit-seed{seed}',scene,prior,p,seed) for seed in AUDIT_SEEDS]
            save(args.output/'independent-seeds.json',ensemble(independent));return
        # A failed numerical audit may still be examined diagnostically, but
        # cannot be promoted by a held-out replay.
        numerics=geometry.read(args.output/'numerics.json')
        scene,prior=geometry.inputs('active-cooling')
        reports=[get(f'validation-seed{seed}',scene,prior,p,seed) for seed in AUDIT_SEEDS]
        save(args.output/'validation-summary.json',dict(ensemble(reports),parameters=p,numerics_passed=numerics['passed'],
             calibration_accepted=False,scope='Frozen 0..30s diagnostic on previously inspected second experiment; no selection or refit'))
    finally:runner.close()


if __name__=='__main__':main()
