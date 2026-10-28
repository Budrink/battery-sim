#!/usr/bin/env python3
"""Fresh controlled ABS topology test; retain the compatible full20 open seed."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import signal

import calibrate_upv_r123 as calibration
from calibrate_fsri30_bayes import atomic_save
from check_upv_restart import native
from check_fsri30_resolved_gas import compact
from check_fsri30_source_reconstruction import metrics
from check_fsri30_open_boundary import cooling_metrics,plot
from fsri30_data import ROOT,load,save,sha
from fsri30_abs_seams import join_abs_panels
from fsri30_early_stop import rejection

BASE=ROOT/'data/development/fsri30-open-boundary-20261025-r2/open-0.01m3s/scenario.json'
PROBE=ROOT/'build/calibration-probes/battery_sim_casing_probe-ambient-20261025-r1'
CORE='f6b72f5930e0f3e891c3127b8825acf6a6e847190adbb762d1d9a055824e7629'


def analytic_check(out):
    # Two equal homogeneous volumes, no gas/chemistry/source. Interface Rc is
    # negligible; material half-cell resistances give G=k*A/L. Compare the
    # actual CUDA temperatures with the closed two-lump exponential solution.
    cells=[]
    for name,x,t in (('ABS:a',0.,600.),('ABS:b',.01,300.)):
        cells.append(dict(id=name,role='passive',material_id='ABS',
            geometry=dict(kind='prismatic',dimensions_m=[.01,.02,.02]),
            pose=dict(position_m=[x,0.,0.]),thermal=dict(heat_capacity_j_per_k=.1,
            initial_temperature_k=t,divisions=[1,1,1],conductivity_w_per_m_k=[.18]*3),
            runaway=dict(energy_j=0.,duration_s=1.)))
    scene=dict(name='analytic ABS continuity',cells=cells,ports=[],interfaces=[],boundaries=[],
        measurement_probes=[],observations={},thermal_path_trace=True,
        simulation=dict(duration_s=10.,max_dt_s=.01,min_dt_s=.01,output_interval_s=1.,stability_factor=.25))
    scene=join_abs_panels(scene,expected_pairs=1)
    save(out/'analytic.json',scene)
    run=native(out/'analytic.json',out/'analytic-replay',dict(restart_contract=CORE),90.,enforce_speed=False)
    g=.18*.02*.02/.01/(1+1e-6)
    expected=[450.+150.*math.exp(-2*g*10./.1),450.-150.*math.exp(-2*g*10./.1)]
    actual=[run['final_fields_k'][key][0] for key in ('ABS:a','ABS:b')]
    error=max(abs(a-b)for a,b in zip(actual,expected))
    result=dict(expected_temperature_k=expected,actual_temperature_k=actual,error_k=error,
        max_thermal_residual_j=run['max_thermal_residual_j'],accepted=error<.1)
    save(out/'analytic-result.json',result)
    if not result['accepted']:raise ValueError('Analytic ABS continuity CUDA check failed')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    calibration.PROBE=PROBE
    if sha(PROBE)!=CORE:raise ValueError('Frozen core changed')
    scene=join_abs_panels(json.loads(BASE.read_text()))
    if scene['metadata']['air_mesh']['spacing_m']!=.02:raise ValueError('20mm mesh required')
    # Read-only observers; no solver law or gas topology is changed.
    scene['thermal_path_trace']=True;scene['gas_film_trace']=True
    save(out/'scenario.json',scene)
    state=dict(status='running',phase='analytic-cuda-check',pid=os.getpid(),
        native_sha256=CORE,source=str(BASE),source_sha256=sha(BASE),
        scenario_sha256=sha(out/'scenario.json'),fresh_from_zero=True,mesh_mm=20.,
        chemistry_materials_gas_frozen=True,calibrated=False,validation_passed=False)
    save(out/'contract.json',dict(state,controller_sha256=sha(Path(__file__)),
        geometry_adapter_sha256=sha(Path(__file__).with_name('fsri30_abs_seams.py'))))
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        atomic_save(out/'status.json',state);analytic_check(out)
        state.update(phase='fresh-full-5945s');atomic_save(out/'status.json',state)
        data,_=load()
        def prune(prefix):
            decision=rejection(scene,prefix,data)
            if decision:
                save(out/'early-stop.json',decision)
                return decision['reason']
            return None
        for key,name in (('BATTERY_AVERAGED_FAILURE_SNAPSHOT','pressure-failure.json'),
            ('BATTERY_MIXING_FAILURE_SNAPSHOT','mixing-failure.json'),('BATTERY_GAS_FAILURE_SNAPSHOT','constituent-failure.json')):
            os.environ[key]=str(out/name)
        run=native(out/'scenario.json',out/'replay',dict(restart_contract=CORE,
            stop_after_first_rupture=False,save_committed_restart=str(out/'post-cascade-3000.bin'),
            committed_restart_at_s=3000.,gas_frames_path=str(out/'gas-frames.bin')),
            5945.*5.2+60.,minimum_speed=.2,rolling_minimum=.2,prune_fn=prune)
        if run.get('max_oxidizer_residual_kg',0.)>1e-10:raise ValueError('Oxygen balance failed')
        data,_=load();result=metrics(scene,run,data)
        result.update(cooling=cooling_metrics(run,data),run_sha256=sha(out/'replay/run.json.gz'))
        if result['minimum_regular_step_s']<.01-1e-9:raise ValueError('Physical floor violated')
        save(out/'summary.json',result)
        state.update(phase='comparison',result=result);atomic_save(out/'status.json',state)
        candidate=compact(run);del run
        with gzip.open(BASE.parent/'replay/run.json.gz','rt')as f:reference=json.load(f)
        before=cooling_metrics(reference,data)
        save(out/'comparison.json',dict(control_cooling=before,candidate_cooling=result['cooling'],
            cooling_rmse_change_k=result['cooling']['envelope_rmse_k']-before['envelope_rmse_k'],
            control_run_sha256=sha(BASE.parent/'replay/run.json.gz'),
            automatically_promoted=False,scope='Native ABS seam continuity only, not independent validation'))
        plot(out,{'disconnected ABS (control)':compact(reference),'continuous ABS (candidate)':candidate},data)
        state.update(status='complete',phase='observed-comparison',automatically_promoted=False)
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='pruned'if str(exc).startswith('Pruned: ')else'failed',error=str(exc))
    finally:atomic_save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()
