#!/usr/bin/env python3
"""Controlled hot-gas response contrasts from one exact pre-rupture state.

No per-cell thermal multipliers, new energy source or solid refit.
Explicit route probes alter existing passages only and are never calibration.
Cold gas references remain identical; hot effective convection is unresolved
by the dataset and must NOT be identified with molecular diffusivity alone.
"""
import argparse
import copy
import hashlib
import gzip
import json
import math
from pathlib import Path

import calibrate_upv_r123 as cal
import upv_r123_data as dataio
from check_upv_restart import native
from check_upv_local_max import summary,local_maximum
from check_upv_averaged_gas import averaged_scene
from check_upv_monotone_gas import plot
from upv_cooling_review import phase_report, plot_cooling
from upv_destruction_objective import reject_reason
from upv_vessel_heat_budget import wall_inventory


def reference(law):
    return law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1.+math.exp(-law.get('geometry_reference_logit',0.)))


def hot_law(g,ceiling,gain600,gain900,gain500=None):
    floor=.01;span=ceiling-floor;z=math.log(.99/(ceiling-1.))
    def metric(gain):
        f=(gain-floor)/span
        return math.log(f/(1.-f))-z
    law=dict(kind='directed_geometry',geometry_reference_logit=z,
        background_conductance_w_per_k=floor*g,active_conductance_w_per_k=span*g,
        endpoint_damage_weight=0.,destroyed_geometry_reference_logit=z,
        destroyed_background_conductance_w_per_k=floor*g,destroyed_active_conductance_w_per_k=span*g)
    knots=[[298.15,0.],[350.,0.],[450.,0.]]
    if gain500 is not None:knots.append([500.,metric(gain500)])
    knots.extend([[600.,metric(gain600)],[900.,metric(gain900)]])
    profile=dict(reference_temperature_k=298.15,temperature_scale_k=100.,temperature_log_response=knots)
    return law,profile


def warm_surface_law(g):
    law,profile=hot_law(g,2000.,1000.,1500.)
    floor=.01;span=2000.-floor;z=law['geometry_reference_logit']
    profile['temperature_log_response']=[]
    for t,gain in ((298.15,1.),(350.,1.),(400.,100.),(450.,600.),(600.,1000.),(900.,1500.)):
        f=(gain-floor)/span
        profile['temperature_log_response'].append([t,math.log(f/(1.-f))-z])
    return law,profile


def warm_mixing_law(strength=1.):
    if not math.isfinite(strength) or not .01<=strength<=1.:
        raise ValueError('Warm mixing strength outside diagnostic range')
    law,profile=hot_law(1.,100000.*strength,30000.*strength,90000.*strength)
    floor=.01;span=100000.*strength-floor;z=law['geometry_reference_logit']
    profile['temperature_log_response']=[]
    for t,gain in ((298.15,1.),(350.,1.),(400.,1000.),(450.,10000.),(600.,30000.),(900.,90000.)):
        if t>=400.:gain*=strength
        f=(gain-floor)/span
        profile['temperature_log_response'].append([t,math.log(f/(1.-f))-z])
    return law,profile


def ambient_mixing_law():
    """Convection decays towards ambient, not abruptly towards 350 K.

    This is an unresolved free-gas constitutive probe, not molecular D(T).
    The reference at 298.15 K, pore law and closed barriers remain unchanged.
    """
    law,profile=warm_mixing_law(.1)
    floor=law['background_conductance_w_per_k'];span=law['active_conductance_w_per_k']
    z=law['geometry_reference_logit'];profile['temperature_log_response']=[]
    for t,gain in ((298.15,1.),(320.,10.),(350.,100.),(400.,1000.),(450.,1000.),(600.,3000.),(900.,9000.)):
        f=(gain-floor)/span
        profile['temperature_log_response'].append([t,math.log(f/(1.-f))-z])
    return law,profile


def wall_mass_scene(original,factor):
    """Only effective vessel mass changes; never fit a material's specific cp.

    The vessel's shape/mass are not published. Its existing cuboid is a lumped
    thermal proxy, not a measured steel thickness. Fresh initial conditions are
    mandatory; a checkpoint with different capacity is not reused.
    """
    if not math.isfinite(factor) or not 1.<=factor<=8.:
        raise ValueError('Wall mass factor must lie between 1 and 8')
    scene=copy.deepcopy(original)
    walls=[c for c in scene['cells']if c['id']=='chamber-wall']
    if len(walls)!=1:raise ValueError('Expected one lumped chamber wall')
    thermal=walls[0]['thermal'];before=thermal['heat_capacity_j_per_k']
    thermal['heat_capacity_j_per_k']=before*factor
    scene.setdefault('metadata',{})['wall_mass_probe']=dict(diagnostic_only=True,
        effective_mass_ratio=factor,reference_capacity_j_k=before,
        trial_capacity_j_k=before*factor,specific_heat_unchanged=True,
        thermal_proxy_dimensions_are_not_measured_vessel_geometry=True)
    return scene


def candidate(original,mixing=False,surfaces=False,scale=1.,rapid=False,rapid_surface=True,warm_surface=False,reverse_surface=False,warm_mixing=False,ambient_mixing=False):
    scene=copy.deepcopy(original);net=scene['gas_network']
    if mixing:
        law,profile=hot_law(1.,100000.,60000.,90000.,30000.)if rapid else hot_law(1.,10000.*scale,300.*scale,3000.*scale)
        if warm_mixing:law,profile=warm_mixing_law(float(warm_mixing))
        if ambient_mixing:law,profile=ambient_mixing_law()
        net['mixing_connectivity']=law;net['surface_connectivity']=profile
    if surfaces:
        for c in net['thermal_contacts']:
            if c['port_id'].startswith('chamber-wall:'):continue
            old=c['connectivity']
            law,profile=hot_law(reference(old),2000.,600.,1500.,200.)if rapid and rapid_surface else hot_law(reference(old),1000.*scale,100.*scale,500.*scale)
            if warm_surface:law,profile=warm_surface_law(reference(old))
            law['endpoint_damage_weight']=old.get('endpoint_damage_weight',1.)
            c['connectivity']=law;c['gas_surface_connectivity']=profile
            if reverse_surface:c['solid_surface_connectivity']=warm_surface_law(reference(old))[1]
    # Both profile families are one for every gas/contact, never selected by C3
    # or thermocouple ID. All cold reference conductances are unchanged;
    # the optional warm probe deliberately starts its response above 350 K.
    return scene


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True,help='Exact last completed control folder, never guessed')
    ap.add_argument('--cases',nargs='+',default=['control','mixing','surfaces','both'])
    ap.add_argument('--duration',type=float,default=950.)
    ap.add_argument('--implicit-mixing',action='store_true')
    ap.add_argument('--free-gas-only',action='store_true')
    ap.add_argument('--joint-surfaces',action='store_true')
    ap.add_argument('--minimum-step',type=float,default=None,help='Explicit coarse floor in seconds, never below 0.005')
    ap.add_argument('--wall-mass-factor',type=float,default=None,
        help='Explicit vessel effective-mass probe (1..8); ALWAYS run from ambient, not checkpoint')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(exist_ok=False)
    if args.minimum_step is not None and not .005<=args.minimum_step<=.02:
        raise ValueError('Diagnostic coarse floor must be between 5 and 20 ms')
    seed=args.seed.resolve();original=json.loads((args.reference/'scenario.json').read_text())
    contract=json.loads((seed/'status.json').read_text())['native_sha256']
    core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    state=dict(status='running',source=str(seed),native_sha256=core,source_checkpoint_core=contract,
        reference=str(args.reference.resolve()),diagnostic_only=True,calibrated=False,validation_run=False,cases={})
    state['full_from_ambient']=args.wall_mass_factor is not None
    if args.wall_mass_factor is not None:state['wall_mass_factor']=args.wall_mass_factor
    cal.save(out/'status.json',state);runs={};data,_=dataio.load('no-cooling')
    specs=dict(control={},mixing=dict(mixing=True),surfaces=dict(surfaces=True),both=dict(mixing=True,surfaces=True),
        strong=dict(mixing=True,surfaces=True,scale=5.),rapid=dict(mixing=True,surfaces=True,rapid=True),
        rapid_mixing=dict(mixing=True,rapid=True),supply=dict(mixing=True,surfaces=True,rapid=True,rapid_surface=False),
        warm=dict(mixing=True,surfaces=True,warm_surface=True),
        warm_supply=dict(mixing=True,surfaces=True,rapid=True,warm_surface=True),
        route10=dict(mixing=True,surfaces=True),route_max=dict(mixing=True,surfaces=True),
        cooling=dict(mixing=True,surfaces=True,warm_surface=True,reverse_surface=True),
        route_cooling=dict(mixing=True,surfaces=True,warm_surface=True,reverse_surface=True),
        reverse=dict(mixing=True,surfaces=True,reverse_surface=True),
        route_reverse=dict(mixing=True,surfaces=True,reverse_surface=True),
        flow=dict(mixing=True,surfaces=True,warm_surface=True,reverse_surface=True,warm_mixing=True),
        tempered=dict(mixing=True,surfaces=True,reverse_surface=True,warm_mixing=.1),
        route_tempered=dict(mixing=True,surfaces=True,reverse_surface=True,warm_mixing=.1),
        route_drift=dict(mixing=True,surfaces=True,reverse_surface=True,ambient_mixing=True))
    labels=dict(control='Исходные газовые связи',mixing='Горячий газ–газ',surfaces='Горячий газ–поверхности',
        both='Обе горячие связи',strong='Обе связи: усиленный отклик',rapid='Конвективный отклик после 450 K',
        rapid_mixing='Быстрый газовый обмен без усиления поверхности',supply='Быстрый подвод; проверенная теплоотдача',
        warm='Теплоотдача тёплого потока',warm_supply='Тёплый поток; быстрый подвод',
        route10='Проходы к C3/ISS ×10: диагностика',route_max='Предельные проходы к C3/ISS: диагностика',
        cooling='Газовая граница: нагрев и обратная теплоотдача',
        route_cooling='Диагностические проходы; обратная теплоотдача',
        reverse='Прежний нагрев; обратная теплоотдача',
        route_reverse='Диагностические проходы; прежний нагрев и обратная теплоотдача',
        flow='Тёплая конвекция и газовая граница; неизменная геометрия',
        tempered='Умеренная тёплая конвекция; прежний нагрев; обратная теплоотдача',
        route_tempered='Диагностические проходы; умеренная тёплая конвекция',
        route_drift='Диагностические проходы; затухание конвекции к комнатной температуре')
    try:
        for name in args.cases:
            if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()!=core:raise RuntimeError('Native binary changed')
            scene=averaged_scene(candidate(original,**specs[name]),args.duration)
            # The archived control stops at 950 s. Extending duration alone
            # would leave the entire late cooling tail without observations.
            scene['failure_sample_times_s']=sorted(set(scene['failure_sample_times_s']) |
                set(dataio.observation_times(data,dataio.event(data),args.duration)))
            if name!='control' and args.implicit_mixing:scene['gas_network']['mixing_integrator']='implicit'
            if name!='control' and args.free_gas_only:scene['gas_network']['free_gas_response_only']=True
            if name!='control' and args.joint_surfaces:scene['gas_network']['mixing_integrator']='implicit_surfaces'
            if name!='control' and args.minimum_step is not None:scene['simulation']['minimum_dt_s']=args.minimum_step
            if name=='control' and args.wall_mass_factor is not None:
                raise ValueError('Archived control cannot be used as a fresh wall-mass trial')
            if args.wall_mass_factor is not None:scene=wall_mass_scene(scene,args.wall_mass_factor)
            transport_probe=name in ('route10','route_max','route_cooling','route_reverse','route_tempered','route_drift')
            if transport_probe:scene,manifest=local_maximum(scene,heat_factor=1.,gas_factor=1000. if name=='route_max'else 10.)
            folder=out/name;folder.mkdir();cal.save(folder/'scenario.json',scene)
            if transport_probe:cal.save(folder/'transport-probe.json',manifest)
            state['active_case']=name;cal.save(out/'status.json',state);print('START',name,flush=True)
            if name=='control':
                source=args.reference.resolve()
                with gzip.open(source/'replay/run.json.gz','rt')as stream:run=json.load(stream)
                cal.save(folder/'archived-source.json',dict(source=str(source),native_physics='unchanged; only probe restart interface rebuilt'))
            else:
                request=dict(diagnostic_transport_probe=transport_probe,stop_after_first_rupture=False)
                if args.wall_mass_factor is None:request.update(restart_from=str(seed/'sealed-state.bin'),
                    restart_contract=contract,restart_gas_response_scenario=str(seed/'scenario.json'),
                    )
                run=native(folder/'scenario.json',folder/'replay',request,1600.,
                    prune_fn=lambda partial:reject_reason(partial,dict(data=data,event=dataio.event(data))))
            report=summary(scene,run)
            report['temperature_phases']=phase_report(run,data,args.duration)
            report['vessel_heat_budget']=wall_inventory(scene,run)
            cal.save(folder/'report.json',report)
            state['cases'][name]=report;runs[name]=run;cal.save(out/'status.json',state)
            # Keep only plotting sections: gas arrays can be released per case.
            runs[name]={k:run[k]for k in ('time_s','temperature_k','gas_film_trace')}
            del run
            plot(out,runs,data,labels=labels,title='Холодные связи и твёрдые материалы фиксированы; горячая газовая связность')
            plot_cooling(out,runs,data,labels)
            print('RESULT',name,report['speed'],json.dumps(report['C3_ISS_samples']),flush=True)
        state['status']='complete'
    except KeyboardInterrupt:
        state.update(status='stopped',error='Interrupted; any partial replay remains explicitly incomplete')
        raise
    except Exception as e:
        state.update(status='failed',error=str(e));raise
    finally:
        state.pop('active_case',None);cal.save(out/'status.json',state)
    return 0


if __name__=='__main__':raise SystemExit(main())
