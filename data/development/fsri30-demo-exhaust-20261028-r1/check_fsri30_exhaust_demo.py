"""Topology/immutability checks and short CUDA verification of the demo fixture."""
import argparse
import json
from pathlib import Path
import subprocess
from fsri30_exhaust_demo import candidate,SOURCE,PROBE


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=json.loads(SOURCE.read_text());scene=candidate(source)
    assert scene['cells']==source['cells'] and scene['shells']==source['shells']
    assert scene['interfaces']==source['interfaces'] and scene['simulation']==source['simulation']
    net=scene['gas_network'];nodes={n['id']:n for n in net['compartments']}
    owners=[c['id'] for c in scene['cells'] if c.get('role')=='battery']
    for owner in owners:
        allowed={f'demo-exhaust-{owner}-capture',f'demo-exhaust-{owner}-outlet'}
        incident=[c for c in net['channels'] if c['from'] in allowed or c['to'] in allowed]
        assert len(incident)==4
        for c in incident:
            if c['kind']=='rupture':assert c['port_id'].startswith(owner+':')
            else:assert c['from'] in allowed and c['to'] in allowed
        assert 'ambient_boundary' in nodes[f'demo-exhaust-{owner}-outlet']
    scene['simulation']['duration_s']=5.;scene['failure_sample_times_s']=[0.,5.]
    path=out/'scenario.json';path.write_text(json.dumps(scene))
    request=dict(visual_recording_path=str(out/'visual'))
    proc=subprocess.run([str(PROBE),str(path)],input=json.dumps(request)+'\n',
        capture_output=True,text=True,timeout=120)
    (out/'cuda.log').write_text(proc.stderr)
    result=json.loads(proc.stdout);(out/'run.json').write_text(json.dumps(result))
    assert proc.returncode==0 and 'error' not in result,result
    limits={'max_thermal_residual_j':1e-4,'max_chemical_residual_j':1e-4,
        'max_mass_residual_kg':1e-10,'max_oxidizer_residual_kg':1e-10}
    for key,limit in limits.items():assert result[key]<=limit,(key,result[key])
    m=json.loads((out/'visual/manifest.json').read_text())
    assert not m['partial'] and len(m['gas'])==len(nodes)
    assert len([n for n in m['gas'] if n.startswith('demo-exhaust-')])==60
    checks=dict(sealed_independent_routes=30,unmodified_battery_physics=True,
        five_second_cuda_pass=True,recorded_duct_nodes=60,
        ledgers={k:result[k] for k in limits},cascade_suppression_not_yet_tested=True)
    (out/'checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
