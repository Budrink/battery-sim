"""CUDA regression: accepted states identical with the optional movie observer."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from fsri30_insulation_delay_probe import candidate, SOURCE, PROBE, ROOT


def check(out):
    out.mkdir(parents=True,exist_ok=False)
    scene=candidate(json.loads(SOURCE.read_text()))
    scene['simulation']['duration_s']=5.
    scene['failure_sample_times_s']=[0.,5.]
    path=out/'scenario.json'
    path.write_text(json.dumps(scene))
    old=ROOT/'build/calibration-probes/battery_sim_casing_probe-wall-damage-20261027-r1'
    runs=[]
    for name,binary,request in [('old',old,{}),('off',PROBE,{}),
        ('on',PROBE,{'visual_recording_path':str(out/'visual')})]:
        p=subprocess.run([str(binary),str(path)],input=json.dumps(request)+'\n',
            text=True,capture_output=True,timeout=120)
        run=json.loads(p.stdout)
        assert p.returncode==0 and 'error' not in run,(p.stderr,run)
        (out/(name+'.json')).write_text(json.dumps(run))
        runs.append(run)
    fields=['cell_mean_temperature_k','chemistry','gas','channels','thermal_paths','events','steps']
    for key in fields:assert runs[0][key]==runs[1][key]==runs[2][key],key
    m=json.loads((out/'visual/manifest.json').read_text())
    assert not m['partial'] and m['time_s']==runs[2]['time_s']
    assert np.allclose(m['time_s'],[0.,5.],rtol=0,atol=1e-12)
    width=4*len(m['objects'])+12*len(m['channels'])+6*len(m['gas'])
    data=np.fromfile(out/'visual/frames.bin',dtype='<f4').reshape((2,width))
    for i,name in enumerate(m['objects']):
        if name not in runs[2]['cell_mean_temperature_k']:continue
        assert np.array_equal(data[:,4*i],np.asarray(runs[2]['cell_mean_temperature_k'][name],dtype=np.float32)),name
    assert np.isfinite(data).all()
    result=dict(original_core_states_bitwise_equal=True,observer_states_bitwise_equal=True,
        full_capture=True,recorded_temperatures_match=True,frames=2,frame_floats=width,
        calibrated=False,validation_passed=False)
    (out/'checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    check(p.parse_args().output.resolve())
