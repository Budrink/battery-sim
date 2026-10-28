#!/usr/bin/env python3
"""Bounded adaptive gas-response search, preserving the latest geometry/core.

Conditional suffixes share one exact-core sealed checkpoint. Cooling acceptance
is not final physical validation; a selected profile still needs a fresh long run.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal

import calibrate_upv_r123 as cal
from check_upv_restart import native
from refine_upv_gas_access import temperature_pruner
from probe_upv_gas_access import plot
from upv_cooling_review import plot_cooling,phase_report
from upv_cooling_fit import rejection_reason
from upv_gas_response_refinement import candidate,assess,front_prune_reason,CRITERIA_VERSION
import upv_r123_data as dataio


def load(path):
    with gzip.open(path,'rt')as stream:return json.load(stream)


def publish_best(out,row,data):
    run=load(out/row['trial']/'replay/run.json.gz');runs={'Кандидат':run}
    plot_cooling(out,runs,data,{'Кандидат':'Лучший допустимый кандидат'})
    plot(out,runs,data,filename='burst-comparison.png')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True,help='Completed fresh geometry directory')
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--end',type=float,default=1100.)
    ap.add_argument('--max-trials',type=int,default=12)
    ap.add_argument('--resume',action='store_true',help='Audit and rescore all completed trials, preserving old assessments')
    a=ap.parse_args();source=a.reference.resolve();out=a.output.resolve();out.mkdir(exist_ok=a.resume)
    if a.end<1100. or a.end>1800.:raise ValueError('Cooling search requires 1100..1800 s')
    prior=json.loads((source/'status.json').read_text())
    cal.PROBE=a.probe.resolve();core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    if prior['status']!='complete'or prior['native_sha256']!=core:
        raise ValueError('Require completed exact-core fresh reference')
    original=json.loads((source/'trial-000/scenario.json').read_text())
    prefix=json.loads((source/'prefix-scenario.json').read_text())
    warm=load(source/'prefix/run.json.gz')
    if not prior['trials'][0]['fresh_from_ambient']or warm['started_at_s']!=0. or warm['advanced_to_s']!=835. or warm.get('first_rupture_s')is not None:
        raise ValueError('Require verified fresh sealed 835 s prefix')
    def physical(scene):
        scene=json.loads(json.dumps(scene));scene.pop('metadata',None)
        scene.pop('failure_sample_times_s',None);scene['simulation'].pop('duration_s',None)
        return scene
    if physical(original)!=physical(prefix):raise ValueError('Fresh reference geometry changed across checkpoint')
    data,record=dataio.load('no-cooling');event=dataio.event(data)
    strict_prune=temperature_pruner(data)
    state=dict(status='running',pid=os.getpid(),reference=str(source),native_sha256=core,
        source_sha256=hashlib.sha256((source/'trial-000/scenario.json').read_bytes()).hexdigest(),
        objective_code_sha256=hashlib.sha256((Path(__file__).parent/'upv_cooling_fit.py').read_bytes()).hexdigest(),
        response_code_sha256=hashlib.sha256((Path(__file__).parent/'upv_gas_response_refinement.py').read_bytes()).hexdigest(),
        experimental_sha256=record['canonical_sha256'],end_s=a.end,minimum_regular_dt_s=original['simulation']['minimum_dt_s'],
        geometry_frozen=True,chemistry_frozen=True,materials_frozen=True,solid_contacts_frozen=True,
        numerical_tolerances_frozen=True,accepted_calibration=False,records=[],best=None)
    state['criteria_version']=CRITERIA_VERSION
    state['runner_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if a.resume:
        previous=json.loads((out/'status.json').read_text())
        if previous['status']=='running':raise ValueError('Stop the existing controller before resuming')
        for key in ('reference','native_sha256','source_sha256','end_s','minimum_regular_dt_s','experimental_sha256'):
            if previous[key]!=state[key]:raise ValueError('Resume changes frozen contract: '+key)
        resumes=previous.get('resumes',[])
        audit=f'assessment-before-resume-{len(resumes):03d}.json'
        cal.save(out/audit,previous)
        state['resumes']=resumes+[dict(previous_status=previous['status'],previous_assessment=audit,criteria=CRITERIA_VERSION)]
        state['records']=previous['records']
        for row in state['records']:
            if row['status']!='complete':continue
            run=load(out/row['trial']/'replay/run.json.gz')
            result=assess(run,data,a.end);row.update(result)
            cal.save(out/row['trial']/('assessment-'+CRITERIA_VERSION+'.json'),result)
            if result['preserved_front']['accepted']and(state['best']is None or row['objective']<state['best']['objective']):
                state['best']=dict(row)
        recorded={r['trial']for r in state['records']}
        for folder in sorted(out.glob('trial-*')):
            if not folder.is_dir()or folder.name in recorded:continue
            scene=json.loads((folder/'scenario.json').read_text());p=scene['metadata']['gas_response_refinement']
            state['records'].append(dict(trial=folder.name,status='interrupted',mixing_scale=p['mixing_scale'],
                damage_delta=p['casing_reverse_damage_delta'],error='Controller stopped; partial evidence preserved, not scored'))
    cal.save(out/'status.json',state)
    if state['best']:
        cal.save(out/'best-candidate.json',state['best']);publish_best(out,state['best'],data)
    queue=[(1.,0.,'extended-control'),(.3,0.,'mixing-contrast'),(.1,0.,'mixing-contrast'),
        (1.,-4.,'reverse-damage-contrast'),(.3,-4.,'combined-contrast')]
    seen={(round(math.log(r['mixing_scale']),8),round(r['damage_delta'],8))for r in state['records']}
    if a.resume and state['best']:
        scale=state['best']['mixing_scale']
        queue=[(scale,-4.,'reverse-damage-contrast'),(1.,-4.,'reverse-damage-contrast'),
            (scale,-6.,'reverse-damage-bracket'),(.6,-4.,'combined-bracket')]+queue
    runs={};radius=.5
    first=max([int(p.name.split('-')[1])for p in out.glob('trial-*')if p.is_dir()]+[-1])+1
    def stop(_sig,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    def prune(run):return strict_prune(run)or rejection_reason(run,data,event)or front_prune_reason(run,data)
    try:
        for i in range(first,a.max_trials):
            while queue and (round(math.log(queue[0][0]),8),round(queue[0][1],8))in seen:queue.pop(0)
            if not queue:state['status']='local_bracket_exhausted';break
            scale,damage,origin=queue.pop(0);seen.add((round(math.log(scale),8),round(damage,8)))
            folder=out/f'trial-{i:03d}';folder.mkdir();scene=candidate(original,scale,damage,a.end)
            cal.save(folder/'scenario.json',scene)
            state.update(active_trial=folder.name);cal.save(out/'status.json',state)
            print('START',folder.name,'mixing',scale,'damage delta',damage,flush=True)
            row=dict(trial=folder.name,mixing_scale=scale,damage_delta=damage,origin=origin,conditional_restart=True)
            try:
                if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()!=core:raise RuntimeError('Native binary changed')
                run=native(folder/'scenario.json',folder/'replay',dict(restart_contract=core,
                    restart_from=str(source/'sealed-state.bin'),restart_gas_response_scenario=str(source/'prefix-scenario.json'),
                    stop_after_first_rupture=False),max(900.,2.2*(a.end-835.)),prune_fn=prune)
                result=assess(run,data,a.end);cal.save(folder/'assessment.json',result)
                row.update(status='complete',**result,realtime_ratio=(a.end-835.)/run['wall_s'],
                    wall_s=run['wall_s'],balances={k:v for k,v in run.items()if k.startswith('max_')and'residual'in k})
                cal.save(folder/'phases.json',phase_report(run,data,a.end))
                # Keep the physically sharp incumbent, not merely a smaller
                # cooling scalar obtained by erasing the burst.
                if result['preserved_front']['accepted']:
                    old=state['best'];improved=old is None or result['objective']<.99*old['objective']
                    if improved:
                        state['best']=dict(row);cal.save(out/'best-candidate.json',row)
                        runs={'Кандидат':run};plot_cooling(out,runs,data,{'Кандидат':'Лучший резкий кандидат'})
                        plot(out,runs,data,filename='burst-comparison.png')
                        if old is not None:
                            # Follow the improving axis before widening; steps
                            # are parameter brackets, NOT physical time steps.
                            if scale!=old['mixing_scale']:
                                sign=1. if scale>old['mixing_scale']else -1.
                                queue.insert(0,(min(3.,max(.03,scale*math.exp(sign*radius))),damage,'continue-mixing'))
                            if damage!=old['damage_delta']:
                                sign=1. if damage>old['damage_delta']else -1.
                                queue.insert(0,(scale,min(6.,max(-6.,damage+sign*2.)),'continue-damage'))
                        for s,d in [(scale*math.exp(-radius),damage),(scale*math.exp(radius),damage),(scale,damage-2.),(scale,damage+2.)]:
                            if .03<=s<=3. and -6.<=d<=6.:queue.append((s,d,'local-bracket'))
                    else:radius=max(.08,radius*.8)
                if result['accepted']:
                    state['status']='cooling_tolerances_met_pending_fresh';state['records'].append(row)
                    cal.save(folder/'result.json',row);break
            except (RuntimeError,TimeoutError)as exc:row.update(status='rejected',error=str(exc))
            state['records'].append(row);cal.save(folder/'result.json',row);cal.save(out/'status.json',state)
            print('RESULT',json.dumps({k:row[k]for k in ['trial','status','objective','realtime_ratio','error']if k in row}),flush=True)
        else:state['status']='trial_budget_not_converged'
    except KeyboardInterrupt:state['status']='stopped'
    except BaseException as exc:state.update(status='failed',error=str(exc));raise
    finally:state.pop('active_trial',None);cal.save(out/'status.json',state)


if __name__=='__main__':main()
