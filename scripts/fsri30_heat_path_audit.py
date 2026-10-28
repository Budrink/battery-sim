#!/usr/bin/env python3
"""Audit accepted gas reaction/film ledgers, never infer heat from hot colors.

Unobserved gas nodes are explicitly accounted for by the global-minus-observed
remainder. A hot gas node without a surface contact is not automatically a
topology error: an intervening barrier may be real. Report it, do not bypass it.
"""
import argparse
import gzip
import json
from pathlib import Path

from fsri30_data import save,sha


def group(name):
    for prefix in ('body-','top-','base-','plume-','inside-'):
        if name.startswith(prefix):return prefix[:-1]
    return name


def audit(scene,run):
    film={}
    for value in run.get('gas_film_trace',{}).values():
        solid=value['solid'].split(':case:')[0]
        row=film.setdefault(solid,dict(net_j=0.,positive_j=0.,negative_j=0.))
        for key,field in (('net_j','heat_to_solid_j'),('positive_j','positive_step_heat_j'),
                          ('negative_j','negative_step_heat_j')):
            row[key]+=value[field][-1]-value[field][0]
    groups={};nodes=[];observed=0.
    contacts={c['compartment'] for c in scene['gas_network']['thermal_contacts']}
    for name,value in run['gas'].items():
        heat=value['released_energy_j'][-1]-value['released_energy_j'][0]
        observed+=heat
        row=groups.setdefault(group(name),dict(reaction_heat_j=0.,remaining_fuel_j=0.,nodes=0))
        row['reaction_heat_j']+=heat;row['remaining_fuel_j']+=value['chemical_j'][-1];row['nodes']+=1
        nodes.append(dict(id=name,reaction_heat_j=heat,remaining_fuel_j=value['chemical_j'][-1],
            peak_temperature_k=max(value['temperature_k']),direct_solid_contact=name in contacts))
    total=run.get('process_totals',{}).get('gas_released_energy_j')
    global_heat=None if total is None else total[-1]-total[0]
    return dict(interval_s=[run['time_s'][0],run['time_s'][-1]],partial=run.get('partial',False),
        conditioned_transport_diagnostic=run.get('diagnostic_conditioned_transport_suffix',False),
        accepted_film_heat=film,observed_gas_groups=groups,global_reaction_heat_j=global_heat,
        unobserved_reaction_heat_j=None if global_heat is None else global_heat-observed,
        hot_reacting_nodes_without_direct_solid_contact=sorted(
            [n for n in nodes if not n['direct_solid_contact'] and n['reaction_heat_j']>1.],
            key=lambda n:-n['reaction_heat_j']),
        interpretation='Accepted integrals; absent contact is diagnostic, not permission to cross a solid barrier',
        calibrated=False,validation_passed=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();path=args.trial/'replay/run.json.gz'
    with gzip.open(path,'rt') as stream:run=json.load(stream)
    scene_path=args.trial/'scenario.json'
    result=audit(json.loads(scene_path.read_text()),run)
    result['provenance']=dict(run=str(path.resolve()),run_sha256=sha(path),
        scenario_sha256=sha(scene_path))
    save(args.output,result)
    print(json.dumps({k:result[k] for k in ('interval_s','global_reaction_heat_j','unobserved_reaction_heat_j','observed_gas_groups')},indent=2))


if __name__=='__main__':main()
