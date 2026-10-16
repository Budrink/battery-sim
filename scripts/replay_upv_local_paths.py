#!/usr/bin/env python3
"""Generate/replay local gas topology with frozen thermal and chemical inputs.

This is a geometry diagnostic, not a calibration or validation claim.
"""
import argparse
import hashlib
from pathlib import Path
import calibrate_upv_area as area
import upv_local_mass_paths as local
import gas_void_geometry as voids
from calibrate_upv import save


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration',type=float,default=3.)
    p.add_argument('--refinement',type=float,default=.5)
    p.add_argument('--generate-only',action='store_true')
    args=p.parse_args()
    if args.duration<=0 or args.duration>30 or not 0<args.refinement<=1:p.error('Duration must be in (0,30], refinement in (0,1]')
    args.output.mkdir(parents=True,exist_ok=True)
    s,prior=area.old.geometry.inputs('no-cooling')
    s=local.attach(area.build(s,area.BASE,17))
    s['simulation']['duration_s']=args.duration
    s['simulation']['stability_factor']*=args.refinement
    s['failure_sample_times_s']=sorted({0.,args.duration,*[t for t in s['failure_sample_times_s'] if t<args.duration]})
    fp=hashlib.sha256(Path(__file__).read_bytes()+Path(local.__file__).read_bytes()+Path(voids.__file__).read_bytes()+
                      Path(area.__file__).read_bytes()+Path(area.old.__file__).read_bytes()+area.old.fit.fingerprint(area.old.fit.PROBE).encode()).hexdigest()
    manifest=dict(fingerprint=fp,duration=args.duration,refinement=args.refinement,calibration_accepted=False)
    m=args.output/'manifest.json'
    if m.exists() and area.old.geometry.read(m)!=manifest:raise ValueError('Model changed: use a fresh output folder')
    save(m,manifest);save(args.output/'scenario.json',s)
    print('local gas nodes',len(s['gas_network']['compartments']),'channels',len(s['gas_network']['channels']),flush=True)
    if args.generate_only:return
    runner=area.old.fit.native.Runner(area.old.fit.PROBE,args.output)
    try:
        run=runner.run(s,args.refinement);save(args.output/'run.json',run)
        print('steps',run['steps'],'wall_s',run['wall_s'],'thermal_residual_j',run['max_thermal_residual_j'],
              'mass_residual_kg',run['max_mass_residual_kg'],flush=True)
    finally:runner.close()


if __name__=='__main__':main()
