#!/usr/bin/env python3
"""Isolated hot-casing CUDA smoke test; NOT an experimental calibration.

Production profile keeps ALL original initial states. Only the two explicitly
manufactured test fixtures start the ABS hot, to exercise cooling in seconds.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import calibrate_upv_r123 as calibration
from check_upv_restart import native
from fsri30_data import ROOT,save,sha
from fsri30_radiation import add_exterior_radiation

BASE=ROOT/'data/development/fsri30-abs-seams-20261025-r2/scenario.json'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    calibration.PROBE=args.probe.resolve();core=sha(calibration.PROBE);base=json.loads(BASE.read_text())
    # Reusable production seed: source initial conditions/chemistry unchanged.
    save(out/'scenario.json',add_exterior_radiation(base,.1))
    state=dict(status='running',native_sha256=core,source_sha256=sha(BASE),
        smoke_fixture=True,experimental_replay=False,calibrated=False,validation_passed=False,results={})
    save(out/'status.json',state)
    try:
        for emissivity in (0.,.1):
            folder=out/f'hot-wall-eps-{emissivity:g}';folder.mkdir()
            scene=add_exterior_radiation(base,emissivity)
            for c in scene['cells']:
                if c['id'].startswith('ABS:'):c['thermal']['initial_temperature_k']=900.
            scene['simulation']['duration_s']=10.;scene.pop('failure_sample_times_s',None)
            scene['gas_film_trace']=False;scene['thermal_path_trace']=False
            scene['metadata']['manufactured_hot_wall_fixture']=True
            save(folder/'scenario.json',scene)
            def prune(prefix):
                if any(max(v)>1500. for v in prefix['cell_mean_temperature_k'].values()):
                    return 'Manufactured cooling test left its declared temperature range'
            run=native(folder/'scenario.json',folder/'replay',dict(restart_contract=core,stop_after_first_rupture=False),
                90.,prune_fn=prune,minimum_speed=.1,rolling_minimum=.1)
            if run.get('max_oxidizer_residual_kg',0.)>1e-10:raise ValueError('Oxygen ledger failed')
            objects={c['id']:c for c in run['applied_objects']};fields=run['final_fields_k']
            capacity=sum(c['capacity_j_k'] for key,c in objects.items()if key.startswith('ABS:'))
            temperature=sum(c['capacity_j_k']*np.mean(fields[key])for key,c in objects.items()if key.startswith('ABS:'))/capacity
            result=dict(mean_abs_temperature_k=float(temperature),wall_s=run['wall_s'],
                radiative_boundary_energy_out_j=run['process_totals']['thermal_boundary_energy_out_j'][-1],
                max_thermal_residual_j=run['max_thermal_residual_j'],max_mass_residual_kg=run['max_mass_residual_kg'],
                minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'],run_sha256=sha(folder/'replay/run.json.gz'))
            if result['minimum_regular_step_s']<.01-1e-9:raise ValueError('Physical step floor violated')
            state['results'][str(emissivity)]=result;save(out/'status.json',state)
        a=state['results']['0.0'];b=state['results']['0.1']
        if not b['mean_abs_temperature_k']<a['mean_abs_temperature_k'] or b['radiative_boundary_energy_out_j']<=0.:
            raise ValueError('Radiation did not remove heat from the manufactured hot casing')
        state.update(status='complete',accepted_smoke_test=True,
            mean_abs_temperature_decrease_k=a['mean_abs_temperature_k']-b['mean_abs_temperature_k'])
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)
    if state['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
