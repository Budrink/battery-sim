"""Continue from the calibrated heater toward all observed cascade waves.

Only shared R2 and gas transport parameters are searched; no timing shifts,
sensor-specific forcing, or synthetic failures. All candidates are fresh
20 mm runs. Passing temperature-wave gates is not an independent validation.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path

from calibrate_fsri30_bayes import atomic_save
from calibrate_fsri30_sequence import trial
from fsri30_data import save
from fsri30_sequence_model import initial, suggest
from fsri30_wave_fit import reference


def search(fitted,out,state,data,probe,warmup_metrics):
    base=deepcopy(fitted)
    # Represent a 0.01..20 multiplier of the accepted seed within the existing
    # shared R2 search encoding (.001..2). No chemical inventory is changed.
    for cell in base['cells']:
        if cell.get('role')=='battery':cell['conversion']['reaction_rate_per_s']*=10.
    anchor=initial(base);anchor['r2_rate_scale']=.1
    ref=reference(data);rows=[];state.update(stage='cascade-search',cascade_trials=[])
    def warmup_guard(run):
        if run['time_s'][-1]<1500.-1e-7:return None
        m=warmup_metrics(run,data)
        state['fine_warmup_metrics']=m
        atomic_save(out/'status.json',state)
        if m and (m['rmse_k']>8. or m['max_abs_k']>20.):
            return 'Candidate loses calibrated warmup shape'
        return None
    def update(row):
        state.update(stage='cascade-search',active_trial=row['trial'],progress={k:row.get(k) for k in
                     ('assessment','wall_s','live_advanced_to_s','physical_error')})
        atomic_save(out/'status.json',state)
    while True:
        parameters,proposal=suggest(rows,base,anchor)
        number=len(rows);folder=out/f'cascade-{number:04d}'
        row=trial(folder,base,parameters,20,ref,data,update,probe=probe,extra_guard=warmup_guard,
                  committed_restart_at=1800.)
        row['proposal']=proposal;row['r2_multiplier_vs_heater_candidate']=10.*parameters['r2_rate_scale']
        row['secondary_conversion_events']=[]
        if row.get('run_path'):
            path=Path(row['run_path'])
            with gzip.open(path,'rt') as stream:run=json.load(stream)
            for cell in fitted['cells']:
                if cell.get('role')!='battery' or cell['id']=='B10':continue
                threshold=.1*cell['conversion']['mass_kg']*cell['conversion']['gas_yield']
                hit=next((t for t,x in zip(run['time_s'],run['chemistry'][cell['id']]['r2_generated_gas_kg']) if x>=threshold),None)
                if hit is not None:row['secondary_conversion_events'].append(dict(cell=cell['id'],time_s=hit))
            m=warmup_metrics(run,data)
            row['warmup_metrics']=m
            if m and m['final_time_s']>=1499. and (m['rmse_k']>8. or m['max_abs_k']>20.):
                row['assessment']['accepted']=False;row['assessment']['constraint']=max(1.,row['assessment'].get('constraint',0.))
                row['assessment']['reason']='Candidate loses calibrated warmup shape'
        assessment=row['assessment'];second=ref['arrival_groups'][1]['time_s'] if len(ref['arrival_groups'])>1 else None
        near=[e for e in row['secondary_conversion_events'] if second is not None and abs(e['time_s']-second)<=100.]
        qualified=bool(row['status']=='complete' and assessment.get('accepted') and
                       assessment.get('phase_window_complete') and len(near)>=2 and not row.get('physical_error'))
        row['independent_secondary_reactions_near_second_wave']=len(near)
        row['qualified_candidate']=qualified
        if assessment.get('accepted') and not qualified:
            assessment['accepted']=False;assessment['constraint']=max(1.,assessment.get('constraint',0.))
            assessment['reason']='Temperature-wave fit lacks secondary R2 reactions near second wave'
        rows.append(row);save(folder/'search-result.json',row)
        state.update(cascade_trials=rows,best_candidate=str(folder) if qualified else state.get('best_candidate'))
        atomic_save(out/'status.json',state)
        if qualified:
            state.update(status='cascade-candidate-qualified',best_candidate=str(folder),
                         note='All-wave candidate; TC mapping and independent validation remain unresolved')
            return
        if row['status']=='numerical-failed':
            state.update(status='numerical-review-needed',reason=row.get('error'));return
