#!/usr/bin/env python3
"""Explicit constitutive revision from one ACTUAL completed training scene.

Project its pre-event electrical trajectory onto stored charging precursor.
This is an initialization, NOT inherited calibration or a held-out fit.
All other actual fitted physics is copied exactly, never rebuilt from defaults.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

import calibrate_upv_r123 as cal


def seed(scene, run, end_s=850.):
    result=copy.deepcopy(scene)
    fits={}
    time=np.asarray(run['time_s'])
    for cell in result['cells']:
        law=cell.get('electrical')
        if not law or law.get('charge_log_slope',0.)==0.:
            continue
        if law.get('charge_state_input')=='stored_precursor':
            raise ValueError('Scene already uses stored precursor; refusing a second migration')
        ident=cell['id'];chem=run['chemistry'][ident]
        a=np.asarray(chem['virgin_kg']);bc=np.asarray(chem['healthy_charged_kg'])
        u=np.divide(bc,a+bc,out=np.zeros_like(bc),where=a+bc>0.)
        mixture=(a+bc+law['product_conductivity_ratio']*(
            np.asarray(chem['r2_precursor_kg'])-bc+np.asarray(chem['residue_kg'])))/cell['conversion']['mass_kg']
        resistance=np.asarray(run['electrical'][ident]['resistance_ohm'])
        temperature=np.asarray(run['cell_mean_temperature_k'][ident])
        use=(time>=40.)&(time<=end_s)&(resistance>0.)&(mixture>0.)
        if use.sum()<10:
            raise ValueError('Insufficient pre-event training electrical trajectory')
        design=np.column_stack((np.ones(use.sum()),temperature[use]-law['reference_temperature_k'],u[use]))
        target=np.log(resistance[use]*mixture[use])
        fit=lsq_linear(design,target,bounds=([np.log(.001),0.,0.],[np.log(.0195),.02,20.]))
        if not fit.success:
            raise RuntimeError('Electrical projection failed')
        before={key:law[key]for key in ('resistance_ohm','temperature_log_slope_per_k','charge_log_slope')}
        law.update(resistance_ohm=float(np.exp(fit.x[0])),temperature_log_slope_per_k=float(fit.x[1]),
                   charge_log_slope=float(fit.x[2]),charge_state_input='stored_precursor')
        fits[ident]=dict(previous=before,initialization_log_resistance_rmse=float(np.sqrt(np.mean((design@fit.x-target)**2))),
                         initialization_only=True,pre_event_end_s=end_s)
    if not fits:
        raise ValueError('No old throughput-dependent charging resistance found')
    result.setdefault('metadata',{})['stored_charge_revision']=dict(
        model='stored-charge-polarization-v1',calibrated=False,validation_used=False,
        fitted_materials_reset=False,previous_training_metrics_inherited=False,
        polarization='Bc/(A+Bc), local committed state; not total SOC',projection=fits)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise ValueError('Refusing to overwrite an existing source scene')
    source=args.trial/'scenario.json'
    with gzip.open(args.trial/'replay/run.json.gz','rt')as stream:run=json.load(stream)
    if run.get('partial'):raise ValueError('A partial trajectory is not an initialization reference')
    scene=seed(json.loads(source.read_text()),run)
    scene['metadata']['stored_charge_revision'].update(source_scene=str(source.resolve()),
        source_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    cal.save(args.output,scene)
    print(json.dumps(scene['metadata']['stored_charge_revision'],indent=2))


if __name__=='__main__':main()
