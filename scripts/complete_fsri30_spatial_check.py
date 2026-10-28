#!/usr/bin/env python3
"""Continue a successful reactive mesh pilot with the FULL published horizon.

Does not fit parameters, reuse checkpoints across meshes or promote validation.
An unsuccessful pilot stops the chain and preserves its diagnostic evidence.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from calibrate_fsri30_bayes import atomic_save
from fsri30_data import ROOT,sha
from model_evidence import record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot',required=True,type=Path)
    ap.add_argument('--probe',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    pilot=args.pilot.resolve();probe=args.probe.resolve();state=dict(status='waiting-for-pilot',pid=os.getpid(),
        pilot=str(pilot),native_sha256=sha(probe),validation_passed=False,calibrated=False)
    child=None
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    atomic_save(out/'status.json',state)
    try:
        while True:
            p=json.loads((pilot/'status.json').read_text())
            if p['status']!='running':break
            os.kill(p['pid'],0)  # Do not wait forever on an abandoned status.
            time.sleep(10.)
        if p['native_sha256']!=state['native_sha256']:raise ValueError('Pilot core changed')
        if p['status']!='complete' or not p['spatially_converged']:
            raise ValueError('Pilot did not establish reactive spatial convergence: '+p['status'])
        base=Path(p['base_scenario'])
        if sha(base)!=p['base_sha256']:raise ValueError('Frozen base changed')
        levels=p['levels_mm'];index=levels.index(p['selected_mesh_mm'])
        if index<1:raise ValueError('Mesh selection needs a coarser reference')
        full=out/'full-horizon';state.update(status='running-full-horizon',levels_mm=levels[index-1:],duration_s=5945.)
        command=[sys.executable,str(ROOT/'scripts/check_fsri30_resolved_gas.py'),'--base',str(base),'--probe',str(probe),
            '--output',str(full),'--duration','5945','--stirring',str(p['stirring_prior']),
            '--stirring-length-mm',str(p['stirring_length_mm']),
            '--levels-mm',*[str(h) for h in levels[index-1:]]]
        if p['pressure_patterns']:command.append('--pressure-patterns')
        for key,option in (('vent_spacing_mm','--vent-spacing-mm'),('wall_spacing_mm','--wall-spacing-mm')):
            if p.get(key) is not None:command.extend([option,str(p[key])])
        state['command']=command;atomic_save(out/'status.json',state)
        with (out/'full-controller.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            child.wait()
        result=json.loads((full/'status.json').read_text())
        if child.returncode or result['status']!='complete' or not result['spatially_converged']:
            raise ValueError('Full-horizon mesh check did not converge: '+result['status'])
        mm=result['selected_mesh_mm'];folder=full/f'h-{mm:g}mm'
        scene=json.loads((folder/'scenario.json').read_text());meta=scene['metadata']
        evidence=dict(result_path=str(full/'status.json'),result_sha256=sha(full/'status.json'),
            full_warmup_validation=True,numerical_checks=dict(accepted=True,comparisons=result['comparisons']),
            independent_validation=dict(accepted=False),experimental_sensor_mapping_confirmed=False,
            interpretation='Numerically checked development candidate, NOT an experimentally validated material law')
        records={}
        records['geometry']=record('geometry_prior','fsri30-source-resolved-gas',
            dict(apparatus_geometry=meta['geometry_reconstruction'],air_mesh=meta['air_mesh'],
                 heater=scene['heating_protocols'],scene_path=str(folder/'scenario.json'),scene_sha256=sha(folder/'scenario.json')),
            evidence,compatibility=dict(fixture='fsri30-source-table3-fig17',stage='fresh-full-horizon',
                atmosphere='finite-1m3',core_geometry='equal-volume-prismatic-surrogate'),
            producer_core=state['native_sha256'],activate_development=True)
        records['stirring']=record('constitutive_law','free-gas-thermal-stirring-v2',meta['thermal_stirring_prior'],evidence,
            compatibility=dict(model='conservative-gas-counterexchange',scope='free-gas-not-internal-pores',
                law=meta['thermal_stirring_prior']['law']),producer_core=state['native_sha256'],activate_development=True)
        state.update(status='complete',full_result=str(full/'status.json'),development_records=records,
            selected_scene=str(folder/'scenario.json'),selected_mesh_mm=mm)
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=15.)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state['status']='stopped'
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:
        atomic_save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()
