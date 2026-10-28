#!/usr/bin/env python3
"""Measure a promising candidate at 40 then 20 mm, WITHOUT parameter refitting.

This targeted numerical check may refine a primary-aligned prefix before its
entire coarse cascade passes. That is not sequence acceptance or validation.
Every run starts from zero; raw times and failed prefixes remain available.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import tarfile

import calibrate_upv_r123 as calibration
from calibrate_fsri30_bayes import atomic_save,event
from calibrate_fsri30_sequence import BASE,BASE_HASH,CORE,PROBE,mesh_evidence,trial
from fsri30_data import ROOT,load,save,sha
from fsri30_multilevel import remesh,inventory
from fsri30_sequence_model import candidate
from fsri30_wave_fit import reference


def primary(row):
    groups=row.get('assessment',{}).get('model',{}).get('pulse_groups',[])
    return groups[0] if groups else None


def worth_finer_check(row):
    """Eligibility for a numerical probe, not for a calibrated profile."""
    if row.get('physical_error') or row['status'] in ('numerical-failed','interrupted'):
        return False
    p=primary(row);offset=row.get('assessment',{}).get('primary_offset_s')
    return bool(p and offset is not None and abs(offset)<=250. and
                '3500 K' not in str(row.get('temperature_rejection')))


def paired_events(coarse,fine):
    if any(r.get('physical_error') or r['status']=='numerical-failed' for r in (coarse,fine)):
        return None
    a,b=primary(coarse),primary(fine)
    if not a or not b or a['channels']!=b['channels']:return None
    if coarse['parameters']!=fine['parameters']:raise ValueError('Cannot infer mesh error across changed parameters')
    if coarse['native_sha256']!=fine['native_sha256']:raise ValueError('Cannot infer mesh error across changed cores')
    return dict(coarse_level_mm=coarse['level_mm'],fine_level_mm=fine['level_mm'],
                component_ids=a['channels'],coarse_primary_s=a['time_s'],fine_primary_s=b['time_s'],
                coarse_minus_fine_s=a['time_s']-b['time_s'],parameters=coarse['parameters'],
                coarse_history_sha256=coarse['run_sha256'],fine_history_sha256=fine['run_sha256'],
                scope='Same-parameter primary event only; no shift of measured or computed curves')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    source=json.loads((args.source/'status.json').read_text())
    if source['native_sha256']!=CORE or sha(PROBE)!=CORE or sha(BASE)!=BASE_HASH:
        raise ValueError('Pinned core or retained baseline changed')
    if not worth_finer_check(source):raise ValueError('Source has no suitable observed primary event')
    if sha(Path(source['run_path']))!=source['run_sha256']:raise ValueError('Source history changed')
    base=json.loads(BASE.read_text());parameters=dict(source['parameters'])
    if json.loads((args.source/'scenario.json').read_text())!=candidate(remesh(base,80),parameters):
        raise ValueError('Source is not the same physical scene at 80 mm')
    evidence=mesh_evidence(base);data,manifest=load();ref=reference(data)
    out.mkdir(parents=True,exist_ok=False)
    calibration.PROBE=PROBE
    paths=sorted((ROOT/'scripts').glob('*.py'))
    save(out/'contract.json',dict(native_sha256=CORE,base_sha256=BASE_HASH,
         source_trial=str(args.source.resolve()),source_status_sha256=sha(args.source/'status.json'),
         source_history_sha256=source['run_sha256'],parameters=parameters,
         csv_sha256=manifest['csv_sha256'],levels=[40,20],mesh_discrepancy=evidence,
         source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
         parameter_refitting=False,checkpoint_transfer=False,calibrated=False,validation_passed=False,
         scope='User-requested numerical refinement of a primary-aligned coarse candidate; all-wave acceptance still required'))
    with tarfile.open(out/'source-policy.tar.gz','w:gz') as archive:
        for path in paths:archive.add(path,arcname=str(path.relative_to(ROOT)))
    save(out/'source-observations.json',source)
    state=dict(status='running',pid=os.getpid(),parameters=parameters,calibrated=False,
               validation_passed=False,parameter_refitting=False,source_trial=str(args.source.resolve()))
    rows=[source];pairs=[]
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    def update(row):
        state.update(level_mm=row['level_mm'],active_trial=row['trial'],progress={
            k:row[k] for k in ('assessment','wall_s','live_advanced_to_s','minimum_regular_step_s') if k in row})
        atomic_save(out/'status.json',state)
    try:
        for level in (40,20):
            state.update(level_mm=level,active_trial=str(out/f'h-{level}mm'),progress={})
            atomic_save(out/'status.json',state)
            mesh=remesh(base,level)
            save(out/f'inventory-{level}mm.json',inventory(mesh))
            row=trial(out/f'h-{level}mm',mesh,parameters,level,ref,data,update,evidence=evidence)
            rows.append(row)
            for previous in rows[:-1]:
                pair=paired_events(previous,row)
                if pair:pairs.append(pair)
            atomic_save(out/'comparison.json',dict(pairs=pairs,observations=rows,calibrated=False,
                validation_passed=False,parameters_changed=False))
            event(out,'refinement-result',level_mm=level,status=row['status'],assessment=row['assessment'])
            if level==40 and not worth_finer_check(row):
                state.update(status='refinement-rejected',reason='40 mm primary or numerical checks failed; 20 mm not launched')
                return
        state.update(status='refinement-failed-numerically' if rows[-1]['status']=='numerical-failed' else 'refinement-finished',pairs=pairs,
                     fine_sequence_accepted=bool(rows[-1]['status']=='complete' and not rows[-1].get('physical_error')
                                                 and rows[-1]['assessment'].get('accepted')),
                     note='Actual fine-grid evidence, not a completed calibration or independent validation')
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:
        state.update(status='failed',error=str(exc));raise
    finally:
        atomic_save(out/'status.json',state)
        print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()
