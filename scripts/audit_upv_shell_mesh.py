#!/usr/bin/env python3
"""Time-refinement diagnostics for the fixed 64/32 mm casing comparison."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import compare_upv_shell_mesh as mesh
from calibrate_upv import save

model=mesh.model


def main():
    cli=argparse.ArgumentParser(__doc__);cli.add_argument('--output',type=Path,default=mesh.OUT)
    folder=cli.parse_args().output;scene,prior=model.geometry.inputs('no-cooling')
    manifest=model.geometry.read(folder/'manifest.json')
    fp=hashlib.sha256(Path(mesh.__file__).read_bytes()+Path(model.__file__).read_bytes()+Path(model.protocol.__file__).read_bytes()+model.fit.fingerprint(model.fit.PROBE).encode()).hexdigest()
    if fp!=manifest['fingerprint']:raise ValueError('Original mesh experiment belongs to another model')
    audit_folder=folder/'time-refinement';audit_folder.mkdir(exist_ok=True)
    audit_manifest=dict(parent_fingerprint=fp,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),refinement=.5,seed=model.SEEDS[0])
    path=audit_folder/'manifest.json'
    if path.exists() and model.geometry.read(path)!=audit_manifest:raise ValueError('Changed audit protocol')
    save(path,audit_manifest)
    runner=model.fit.native.Runner(model.fit.PROBE,audit_folder);rows={}
    try:
        for size in mesh.SIZES_MM:
            stem=f'mesh-{size:g}mm-seed{model.SEEDS[0]}'
            coarse=model.geometry.read(folder/(stem+'.json'));path=audit_folder/(stem+'.json')
            if path.exists():fine=model.geometry.read(path)
            else:
                print('time refinement',stem,flush=True)
                fine,candidate=model.replay(runner,scene,prior,model.BASE,model.SEEDS[0],refinement=.5,grid=mesh.grid_for_size(scene,size))
                save(path,fine);save(audit_folder/(stem+'-scenario.json'),candidate)
            delta=model.protocol.refinement_metrics(coarse,fine)
            rows[str(size)]=dict(delta,passed=model.protocol.refinement_passes(delta),
                 pressure_rmse_change_bar=fine['metrics']['pressure_0_30_rmse_bar']-coarse['metrics']['pressure_0_30_rmse_bar'],
                 temperature_3_30_rmse_change_k=fine['metrics']['phases']['3-30']['rmse_k']-coarse['metrics']['phases']['3-30']['rmse_k'],
                 steps=fine['result']['steps'],wall_s=fine['result']['wall_s'])
            save(audit_folder/'summary.json',dict(rows=rows,complete=len(rows)==len(mesh.SIZES_MM),calibration_accepted=False))
            print('time refinement result',size,rows[str(size)],flush=True)
        runs={size:[model.geometry.read(folder/f'mesh-{size:g}mm-seed{seed}.json') for seed in model.SEEDS] for size in mesh.SIZES_MM}
        features={str(size):{key:mesh.stats([r['features'][key] for r in group])
                  for key in ('first_open_s','sampled_max_source_pressure_bar','sampled_chamber_peak_bar','net_export_g','fracture_work_j')}
                  for size,group in runs.items()}
        mean_pressure={size:np.mean([np.asarray(r['result']['gas']['chamber']['pressure_pa'])/1e5 for r in group],axis=0) for size,group in runs.items()}
        mean_temperature={size:{k:np.mean([r['result']['temperature_k'][k] for r in group],axis=0) for k in model.fit.WITNESSES} for size,group in runs.items()}
        a,b=mesh.SIZES_MM
        comparison=dict(features=features,max_mean_pressure_curve_difference_bar=float(np.max(np.abs(mean_pressure[a]-mean_pressure[b]))),
                        max_mean_temperature_curve_difference_k={k:float(np.max(np.abs(mean_temperature[a][k]-mean_temperature[b][k]))) for k in model.fit.WITNESSES},
                        scope='Sampled histories, three realizations; internal pressure peaks are sampled, not event-exact maxima.',
                        calibration_accepted=False)
        save(folder/'mesh-differences.json',comparison)
        print('mesh differences',comparison,flush=True)
    finally:runner.close()


if __name__=='__main__':main()
