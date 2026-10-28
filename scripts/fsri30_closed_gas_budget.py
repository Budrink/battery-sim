"""Partition CLOSED-boundary ledgers without mistaking old outside fuel for a leak."""
import numpy as np


def budget(scene,run):
    outside=[n['id'] for n in scene['gas_network']['compartments']
             if n['id']=='hood' or n['id'].startswith('plume-')]
    if not outside or any(k not in run['gas'] for k in outside):
        raise ValueError('Every outside volume must be observed for the closed-module budget')
    columns=(('mass_kg','gas_mass_kg'),('oxidizer_kg','gas_oxidizer_kg'),
             ('chemical_j','gas_chemical_j'),('released_energy_j','gas_released_energy_j'))
    module={};external={}
    for local,total in columns:
        global_value=np.asarray(run['process_totals'][total])
        outer=sum(np.asarray(run['gas'][k][local]) for k in outside)
        inner=global_value-outer;module[local]=inner;external[local]=outer
    coefficients={n.get('oxidizer_heat_j_per_kg',13.1e6) for n in scene['gas_network']['compartments']}
    if len(coefficients)!=1:raise ValueError('Exact oxygen heat closure requires a shared heat-per-oxygen coefficient')
    heat_per_oxygen=next(iter(coefficients))
    released=module['released_energy_j']-module['released_energy_j'][0]
    oxygen_consumed=module['oxidizer_kg'][0]-module['oxidizer_kg']
    def ends(values):return {k:dict(initial=float(v[0]),final=float(v[-1]),change=float(v[-1]-v[0])) for k,v in values.items()}
    return dict(module=ends(module),outside=ends(external),
        initial_module_oxygen_heat_ceiling_j=float(module['oxidizer_kg'][0]*heat_per_oxygen),
        maximum_oxygen_heat_closure_error_j=float(max(abs(released-oxygen_consumed*heat_per_oxygen))),
        note='Global minus ALL outside nodes; includes every unobserved internal pore. Ceiling assumes no module oxygen source/inflow.',
        conditioned_transport_diagnostic=True,full_warmup_validation=False)


def main():
    import argparse,gzip,json
    from pathlib import Path
    from fsri30_data import save,sha
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();path=args.trial/'replay/run.json.gz'
    with gzip.open(path,'rt') as stream:run=json.load(stream)
    if not run.get('diagnostic_closed_exhausts'):raise ValueError('Not an explicitly closed-exhaust diagnostic')
    result=budget(json.loads((args.trial/'scenario.json').read_text()),run)
    result['provenance']=dict(run=str(path.resolve()),run_sha256=sha(path),
        script_sha256=sha(Path(__file__)),known_until_s=run['time_s'][-1],partial=run.get('partial',False))
    save(args.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
