"""Five-second CUDA regression, not a cascade/calibration run."""
import json, subprocess
from pathlib import Path
base=Path('/home/konstantin/work/battery-sim')
work=Path('/tmp/battery-wall-response-QXQ6Vd')
old=base/'build/calibration-probes/battery_sim_casing_probe-cap-material-20261027-r1'
new=work/'build/cap_probe'
def run(scene,tag,binary):
    s=json.loads(scene.read_text());s['simulation']['duration_s']=5.;s['failure_sample_times_s']=[0.,5.]
    p=work/(tag+'.json');p.write_text(json.dumps(s))
    result=subprocess.run([str(binary),str(p)],input=json.dumps({'stop_after_first_rupture':False}),capture_output=True,text=True,timeout=90)
    r=json.loads(result.stdout);assert result.returncode==0 and 'error' not in r,(result.stderr,r)
    (work/(tag+'-result.json')).write_text(json.dumps(r));return r
baseline=base/'data/development/fsri30-cap-r3-20261027-r1/scenario.json'
candidate=base/'data/development/fsri30-wall-response-20261027-r1/scenario.json'
a=run(baseline,'old',old);b=run(baseline,'new-default',new);c=run(candidate,'wall',new)
for field in ['cell_mean_temperature_k','chemistry','gas','channels','thermal_paths']:
    assert a[field]==b[field],field
assert c['applied_objects']==b['applied_objects']
changed=[]
for name,edge in c['thermal_paths']['interfaces'].items():
    before=b['thermal_paths']['interfaces'][name]
    if edge['conductance_w_per_k'][0]!=before['conductance_w_per_k'][0]:
        assert edge['from'].startswith('B') and edge['to'].startswith(edge['from']+':case:'),name
        assert edge['conductance_w_per_k'][0]<before['conductance_w_per_k'][0]
        changed.append(name)
assert len(changed)==180,len(changed)
assert c['max_mass_residual_kg']<1e-10
for k in ['max_chemical_residual_j','max_thermal_residual_j']:assert c[k]<1e-4
report={'unconfigured_core_bitwise_equal':True,'changed_inner_contacts':len(changed),'other_initial_contact_conductances_unchanged':True,'all_object_properties_unchanged':True,'cuda_5s_ledgers_pass':True,'full_run_started':False}
(work/'checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
