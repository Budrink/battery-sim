#!/usr/bin/env python3
"""Resolve the first 30 s more densely without using extra synthesized observations."""
import argparse
import copy
import json
import numpy as np
import calibrate_upv_gas_energy as g
from calibrate_upv import save


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=g.Path,default=g.OUT)
    p.add_argument('--probe',type=g.Path,default=g.Path('/tmp/battery-sim-cuda-v1/battery_sim_casing_probe'));a=p.parse_args()
    profile=json.loads((a.output/'profile.json').read_text())
    if profile['fingerprint']!=g.fingerprint(a.probe,profile.get('heat_directory')):raise ValueError('Frozen model changed')
    dest=a.output/'dense-check';dest.mkdir(exist_ok=True);runner=g.t.Runner(a.probe,dest);summary={}
    try:
        for mode,name in [('fit','no-cooling'),('validate','active-cooling')]:
            if not (a.output/(mode+'.json')).exists():continue
            s=json.loads((a.output/(name+'-'+mode+'-scenario.json')).read_text())
            s['simulation']['duration_s']=30.
            s['failure_sample_times_s']=np.unique(np.r_[np.arange(0.,5.,.01),np.arange(5.,30.,.1),30.,[x for x in s['failure_sample_times_s'] if x<30]]).tolist()
            r=runner.run(s);fine_s=copy.deepcopy(s);fine_s['simulation']['stability_factor']*=.5;fine=runner.run(fine_s,.5)
            def peak(v):
                pressure=np.asarray(v['gas']['chamber']['pressure_pa'])/1e5;i=int(np.argmax(pressure))
                return dict(peak_bar=float(pressure[i]),peak_time_s=v['time_s'][i])
            summary[name]=dict(coarse=peak(r),refined=peak(fine),max_pressure_difference_bar=float(np.max(np.abs(np.asarray(r['gas']['chamber']['pressure_pa'])-fine['gas']['chamber']['pressure_pa']))/1e5),
                note='Numerical dense replay only; no interpolated experimental values used for fitting')
            if name=='no-cooling':
                delayed=copy.deepcopy(s);delayed['gas_network']['channels'][0]['opening_pressure_pa']*=100
                alternate=runner.run(delayed)
                summary[name]['threshold_x100_sensitivity']=dict(**peak(alternate),
                    rupture_events=[e for e in alternate['events'] if e['kind']=='shell_rupture'],
                    note='Training-only diagnostic after freezing, not a plausible identified casing strength or new fitted candidate')
                save(dest/'no-cooling-threshold-x100.json',alternate)
            save(dest/(name+'.json'),dict(summary=summary[name],coarse=r,refined=fine));print(name,summary[name],flush=True)
        save(dest/'assessment.json',summary)
    finally:runner.close()


if __name__=='__main__':main()
