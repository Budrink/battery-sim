#!/usr/bin/env python3
"""Sequential, explicitly conditional casing heat-transfer / conversion trials.

Only no-cooling is optimized. Validation is run after freezing each profile.
All integration is CUDA. Whole-cell measured capacity is partitioned, not fitted.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares
from calibrate_upv import Worker, save, metrics as heat_metrics, residuals as heat_errors
import calibrate_upv_failure as old
import calibrate_upv_failure_transition as late
import calibrate_upv_links as links
from generate_upv import load_case

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'data/calibration/upv-links-contrast-3x3x3'
OUT = ROOT / 'data/calibration/upv-physical-casing-v5'
CASE = dict(material_id='unidentified-pouch-laminate-PROXY', thickness_m=.00015,
            density_kg_m3=1600., specific_heat_j_kg_k=1250.,
            conductivity_w_per_m_k=[.3,50.,50.], conductivity_frame='normal_tangential',
            seam_contact_h=500., damage_onset_k=425., damage_rate_per_s=0.,
            damage_scale_k=25., destroyed_conductance_factor=.02)
HEAT_NAMES = links.NAMES + (links.INTERNAL_CONTRAST, 'contents_case_h')
HEAT_LO = np.array([-6,-6,-6,math.log(.02),math.log(.02),-12,math.log(20.)])
HEAT_HI = np.array([6,6,6,math.log(99.99),math.log(99.99),12,math.log(2000.)])
FAIL_NAMES = ('energy_j','rate_per_s','destroyed_iss_h','destroyed_tim_h','residue_through','residue_inplane')
FAIL_LO = np.log([1000,.001,.01,.01,.005,.005])
FAIL_HI = np.log([1000000,5,10000,10000,2,2])
PROTOCOL = dict(revision='physical-casing-conditional-v5', fit='no-cooling', validation='active-cooling',
    casing=CASE, casing_status='Engineering proxy, NOT the identified UPV pouch formulation or thickness',
    fixed='Whole battery C=412.87 J/K; reference k; all passive materials; exterior dimensions and layer positions; sensor material-side binding; measured forcing',
    capacity_partition='Ccontents = Cmeasured - rho_case*cp_case*(Vouter-Vinner); no capacity optimization',
    conductivity_note='Reported whole-cell k is retained as the contents reference prior; this partition is not separately measured',
    numbering='Retained historical unconfirmed swapped-witnesses hypothesis; no topology/sensor fitting',
    heat_score='Previous four-witness objective at 2-second times, t>=60 s, same forcing-defined pre-event end',
    failure_score='Same four-witness 300..1800 s late objective as v4; source/early transients diagnostic only',
    gas_scope='Source gas generated and transported into the actual 120 L chamber volume. Single effective ideal gas; no flame chemistry. Measured ch109 remains conditional solid ambient forcing; modeled chamber is NOT fed back as a second ambient heat source.',
    pressure_scope='Ch114 is chamber pressure, NOT cell pressure. Pressure is diagnostic, not fitted; internal volume, gas constants/yield, opening area/strength are unmeasured priors.',
    validation_note='Held out from this optimization, previously inspected during development; not blind validation')


def fingerprint(probe):
    paths = sorted((ROOT/'native').rglob('*.hpp')) + sorted((ROOT/'native').rglob('*.cpp')) + sorted((ROOT/'native').rglob('*.cu')) + sorted((ROOT/'native').rglob('*.cuh'))
    paths += [Path(__file__), Path(old.__file__), Path(links.__file__), Path(late.__file__), probe]
    h=hashlib.sha256()
    for p in paths:h.update(str(p).encode());h.update(p.read_bytes())
    return h.hexdigest()


def read_scene(name):
    return json.loads((PRIOR/(name+'-fitted-scenario.json')).read_text())


def heat_bake(scene, p):
    return links.bake(scene, {k:p[k] for k in HEAT_NAMES if k!='contents_case_h'})


def casing(scene, p, warm=None, thickness=None):
    """Keep outer geometry and total capacity; redirect every exterior battery port."""
    s=copy.deepcopy(scene);s['shells']=[];objects={c['id']:c for c in s['cells']}
    inventory={};panel_dims={}
    for c in s['cells']:
        if c['role']!='battery':continue
        spec=dict(CASE,contents_id=c['id'],contents_contact_h=p['contents_case_h'])
        if thickness is not None:spec['thickness_m']=thickness
        t=spec['thickness_m'];outer=c['geometry']['dimensions_m'];inner=[d-2*t for d in outer]
        vol=math.prod(outer)-math.prod(inner);cap=vol*spec['density_kg_m3']*spec['specific_heat_j_kg_k']
        inventory[c['id']]=dict(total_capacity_j_k=c['thermal']['heat_capacity_j_per_k'],case_capacity_j_k=cap,
            case_mass_kg=vol*spec['density_kg_m3'],outer_dimensions_m=outer)
        c['geometry']['dimensions_m']=inner;c['thermal']['heat_capacity_j_per_k']-=cap
        if c['thermal']['heat_capacity_j_per_k']<=0:raise ValueError('Casing exceeds measured capacity')
        spec['surface_connectivity']=copy.deepcopy(c['thermal']['surface_connectivity'])
        if warm is not None:
            fields={face:warm[c['id']+':case:'+face] for face in ('-x','+x','-y','+y','-z','+z') if c['id']+':case:'+face in warm}
            if fields:spec['initial_temperature_fields_k']=fields
        for axis in range(3):
            dims=[t if k==axis else inner[k]+(2*t if k>axis else 0) for k in range(3)]
            for sign in ('-','+'):panel_dims[c['id']+':case:'+sign+'xyz'[axis]]=dims
        s['shells'].append(spec)
    area_scale={}
    for port in s['ports']:
        if port['cell_id'] not in inventory:continue
        axis='xyz'.index(port['surface'][1]);port['cell_id']+=':case:'+port['surface'];dims=panel_dims[port['cell_id']]
        new_area=dims[(axis+1)%3]*dims[(axis+2)%3];area_scale[port['id']]=new_area/port['area_m2']
        port['area_m2']=new_area;port['rectangle_m']=[dims[(axis+1)%3],dims[(axis+2)%3]]
    ports={x['id']:x for x in s['ports']}
    for b in s['boundaries']:
        if b['port_id'] in area_scale:b['conductance_w_per_k']*=area_scale[b['port_id']]
    for obs in s['measurement_probes']:
        if obs['cell_id'] not in inventory:continue  # e.g. Al2/cooling-plate contact
        edge=next(e for e in s['interfaces'] if e['id']==obs['interface_id'])
        candidates=[ports[edge[k]] for k in ('port_a','port_b') if ports[edge[k]]['cell_id'].startswith(obs['cell_id']+':case:')]
        if len(candidates)!=1:raise ValueError('Sensor does not bind to exactly one casing surface')
        obs['cell_id']=candidates[0]['cell_id'];obs['port_id']=candidates[0]['id']
    s['metadata'].update(casing_protocol=PROTOCOL,casing_inventory=inventory,calibration_status='Conditional physical-casing proxy trial; not identified packaging')
    return s


def audit(run, scene):
    actual={c['id']:c for c in run['applied_objects']}
    for c in scene['cells']:
        a=actual[c['id']]
        if a['capacity_j_k']!=c['thermal']['heat_capacity_j_per_k'] or a['conductivity_w_m_k']!=c['thermal']['conductivity_w_per_m_k'] or a['dimensions_m']!=c['geometry']['dimensions_m']:
            raise ValueError('Frozen material/geometry changed in native replay')
    for identifier,v in scene['metadata']['casing_inventory'].items():
        panel_cap=sum(c['capacity_j_k'] for c in actual.values() if c['shell_contents_id']==identifier)
        if abs(actual[identifier]['capacity_j_k']+panel_cap-v['total_capacity_j_k'])>1e-9:raise ValueError('Measured capacity double counted')
    if run['max_thermal_residual_j']>1e-4 or run['max_chemical_residual_j']>1e-4 or run['max_mass_residual_kg']>1e-10:raise ValueError('Conservation failed')
    for y in run['temperature_k'].values():
        if not np.isfinite(y).all() or min(y)<=0:raise ValueError('Invalid sensor temperature')


class Runner:
    def __init__(self, probe, directory):
        self.path=directory/'candidate.json';self.worker=Worker(probe,self.path)
    def run(self,scene,refinement=1.):
        save(self.path,scene);r=self.worker.run(dict(scenario_path=str(self.path),refinement=refinement));audit(r,scene);return r
    def close(self):self.worker.close()


def pack_heat(p):
    return np.array([math.log(p[k]) if k in links.CONTACTS or k=='contents_case_h' else p[k] for k in HEAT_NAMES])


def unpack_heat(x):
    return {k:float(math.exp(v) if k in links.CONTACTS or k=='contents_case_h' else v) for k,v in zip(HEAT_NAMES,x)}


def optimize(runner, build, error, start, lo, hi, unpack, out, budget):
    cache={};best=float('inf');count=0;clock=time.monotonic()
    def residual(x):
        nonlocal best,count
        key=tuple(x)
        if key in cache:return cache[key]
        p=unpack(x);scene=build(p);r=runner.run(scene);v=error(r);count+=1;rmse=float(np.sqrt(np.mean(v*v)))
        if rmse<best:
            best=rmse;save(out/'checkpoint.json',dict(parameters=p,rmse_k=rmse,evaluations=count));save(out/'best-run.json',r)
        print(f'candidate {count}: RMSE {rmse:.5f} K, best {best:.5f}, native {r["wall_s"]:.2f}s, elapsed {time.monotonic()-clock:.0f}s',flush=True)
        cache[key]=v;return v
    fit=least_squares(residual,start,bounds=(lo,hi),diff_step=.01,x_scale='jac',max_nfev=budget,ftol=2e-5,xtol=2e-5,gtol=2e-5)
    return json.loads((out/'checkpoint.json').read_text())['parameters'],dict(success=bool(fit.success),message=fit.message,nfev=fit.nfev,evaluations=count,
        jacobian_singular_values=np.linalg.svd(fit.jac,compute_uv=False).tolist())


def start_heat():
    return dict(json.loads((PRIOR/'profile.json').read_text())['parameters'],contents_case_h=500.)


def failure_prepare(name, thermal_p, runner):
    original=heat_bake(read_scene(name),thermal_p);data,record=old.load_record(name);event=old.event_marker(data)
    warm=copy.deepcopy(original);warm['simulation']['duration_s']=event['last_pre_event_s'];old.refresh_boundaries(warm,data,0,event['last_pre_event_s'])
    wr=runner.run(casing(warm,thermal_p))
    post=old.post_scene(original,wr,data,event,1800.)
    post['primary_failures']=[]
    source=next(c for c in post['cells'] if c['id']=='C2');source['runaway']=dict(energy_j=0,duration_s=1)
    source['thermal']['bulk_connectivity']['endpoint_damage_weight']=0.
    # This is an event-conditioned experiment, not an identified overcharge clock.
    post=casing(post,thermal_p,wr['final_fields_k'])
    for sh in post['shells']:
        if sh['contents_id']=='C2':
            sh['damage_rate_per_s']=.1
            sh['initial_temperature_fields_k']={}
            i=event['index']-1
            for axis in range(3):
                dims=[1 if k==axis else 3 for k in range(3)]
                for sign in ('-','+'):
                    values=[]
                    for z in range(dims[2]):
                        fz=(0. if sign=='-' else 1.) if axis==2 else (z+.5)/3
                        iss=data['T105_k'][i]*(1-fz)+data['T106_k'][i]*fz
                        tim=data['T103_k'][i]*(1-fz)+data['T104_k'][i]*fz
                        for y in range(dims[1]):
                            for x in range(dims[0]):
                                fx=(0. if sign=='-' else 1.) if axis==0 else (x+.5)/3
                                values.append(float(iss*(1-fx)+tim*fx))
                    sh['initial_temperature_fields_k'][sign+'xyz'[axis]]=values
    source=next(c for c in post['cells'] if c['id']=='C2')
    gas_t=float(np.mean(source['thermal']['initial_temperature_field_k']));tref=298.15
    source['conversion']=dict(mass_kg=.2,reactant_cp=1007.,activated_cp=1007.,residue_cp=800.,
        absorption_j_per_kg=30000.,net_heat_j_per_kg=750000.,l0_m=1e-6,lcrit_m=1.2e-6,energy_scale_j_per_kg=100000.,
        activation_rate_per_s=.1,reaction_rate_per_s=.1,reaction_reference_k=400.,reaction_activation_j_per_mol=60000.,
        gas_yield=.2,gas_chemical_fraction=.2,gas_compartment='inside-C2',residue_conductivity=[.02,.02,.02],
        max_fraction_per_step=.01,max_temperature_step_k=.5)
    # 120 L is specified by the source. Free cell volume/area/strength are NOT.
    ambient=float(data['T109_k'][event['index']-1]);p0=event['pressure_before_bar']*1e5;r=287.;vfree=5e-6
    post['gas_network']=dict(reference_temperature_k=tref,cv_j_per_kg_k=1000.,specific_r_j_per_kg_k=r,
        max_fraction_per_step=.01,max_temperature_step_k=.5,
        compartments=[dict(id='inside-C2',contents_id='C2',volume_m3=vfree,initial_mass_kg=p0*vfree/(r*gas_t),initial_temperature_k=gas_t),
                      dict(id='chamber',volume_m3=.120,initial_mass_kg=p0*.120/(r*ambient),initial_temperature_k=ambient)],
        channels=[dict(id='C2-pouch-rupture',kind='rupture',port_id='C2:case:+z:outer',**{'from':'inside-C2','to':'chamber'},
                       area_m2=1e-5,opening_pressure_pa=100000.,strength_reference_k=400.,strength_temperature_scale_k=100.)],thermal_contacts=[])
    post['metadata'].update(calibration_status='Conditional conversion/gas trial with unmeasured gas/rupture priors',failure_protocol=PROTOCOL)
    at,obs=old.observations(data,event['last_pre_event_s'],1800.)
    return post,obs,dict(event=event,source_csv_sha256=record['canonical_sha256'],warmup_energy_residual_j=wr['energy_residual_j'])


def fail_bake(scene,p):
    s=copy.deepcopy(scene);c=next(c for c in s['cells'] if c['id']=='C2');law=c['conversion']
    law['net_heat_j_per_kg']=p['energy_j']/law['mass_kg'];law['activation_rate_per_s']=law['reaction_rate_per_s']=p['rate_per_s']
    law['residue_conductivity']=[p['residue_through'],p['residue_inplane'],p['residue_inplane']]
    ports={v['id']:v for v in s['ports']}
    for e in s['interfaces']:
        if e['id'] in ('failure-C2-ISS','failure-C2-TIM'):
            key='destroyed_iss_h' if e['id'].endswith('ISS') else 'destroyed_tim_h';e['law']['destroyed_background_conductance_w_per_k']=p[key]*ports[e['port_a']]['area_m2']
    return s


def main():
    cli=argparse.ArgumentParser(__doc__);cli.add_argument('stage',choices=['heat','failure']);cli.add_argument('mode',choices=['baseline','fit','validate'])
    cli.add_argument('--probe',type=Path,default=Path('/tmp/battery-sim-cuda-v1/battery_sim_casing_probe'))
    cli.add_argument('--output',type=Path,default=OUT);cli.add_argument('--max-evaluations',type=int,default=20)
    cli.add_argument('--initial-profile',type=Path,help='Earlier training-only profile used only as optimizer initialization')
    args=cli.parse_args();out=args.output/args.stage;out.mkdir(parents=True,exist_ok=True)
    profile_path=out/'profile.json';name='active-cooling' if args.mode=='validate' else 'no-cooling'
    if args.mode=='fit' and profile_path.exists():raise ValueError('Use a fresh output directory; frozen profile exists')
    if args.mode=='validate':
        profile=json.loads(profile_path.read_text())
        if profile['fingerprint']!=fingerprint(args.probe):raise ValueError('Model changed after profile freeze')
    runner=Runner(args.probe,out)
    try:
        if args.stage=='heat':
            original=read_scene(name);data,record,_=load_case(old.DATA,name)
            build=lambda p:casing(heat_bake(original,p),p)
            error=lambda r:np.concatenate(list(heat_errors(r,data).values()))
            score=lambda r:heat_metrics(r,data)
            start=start_heat();unpack=unpack_heat;lo,hi=HEAT_LO,HEAT_HI;x0=pack_heat(start)
            meta=dict(source_csv_sha256=record['canonical_sha256'],prior_sha256=old.digest(PRIOR/(name+'-fitted-scenario.json')))
        else:
            heat_profile=json.loads((args.output/'heat/profile.json').read_text())
            if heat_profile['fingerprint']!=fingerprint(args.probe):raise ValueError('Changed model since heat fit')
            scene,observed,meta=failure_prepare(name,heat_profile['parameters'],runner)
            save(out/(name+'-unfitted-scenario.json'),scene)
            build=lambda p:fail_bake(scene,p)
            error=lambda r:np.concatenate(list(late.errors(r,observed,meta['event']).values()))
            score=lambda r:late.metrics(r,observed,meta['event'])
            start=dict(zip(FAIL_NAMES,[150000.,.1,5.,5.,.02,.02]));unpack=lambda x:dict(zip(FAIL_NAMES,map(float,np.exp(x))))
            lo,hi=FAIL_LO,FAIL_HI;x0=np.log(list(start.values()))
            meta['frozen_heat_profile_sha256']=old.digest(args.output/'heat/profile.json')
        if args.initial_profile:
            initial=json.loads(args.initial_profile.read_text())
            if initial.get('validation_used_for_selection') is not False:raise ValueError('Initialization must be training-only')
            start=initial['parameters']
            x0=pack_heat(start) if args.stage=='heat' else np.log([start[k] for k in FAIL_NAMES])
            meta['initial_profile_sha256']=old.digest(args.initial_profile)
        if args.mode=='validate':p=profile['parameters']
        elif args.mode=='baseline':p=start
        else:
            p,search=optimize(runner,build,error,x0,lo,hi,unpack,out,args.max_evaluations)
            profile=dict(parameters=p,search=search,protocol=PROTOCOL,meta=meta,fingerprint=fingerprint(args.probe),
                         status='conditional_proxy_fit_not_identified_materials',validation_used_for_selection=False)
            save(profile_path,profile)
        s=build(p);run=runner.run(s);report=dict(experiment=name,stage=args.stage,parameters=p,meta=meta,protocol=PROTOCOL,metrics=score(run),result=run)
        if args.mode!='baseline':
            fine=runner.run(s,.5);report['refined_metrics']=score(fine)
            report['max_refinement_difference_k']=max(float(np.max(np.abs(np.array(v)-np.array(fine['temperature_k'][k])))) for k,v in run['temperature_k'].items())
            save(out/(name+'-fitted-scenario.json'),s)
        save(out/(args.mode+'.json'),report)
        print(json.dumps({k:report[k] for k in ('experiment','stage','parameters','metrics')},indent=2),flush=True)
    finally:runner.close()


if __name__=='__main__':main()
