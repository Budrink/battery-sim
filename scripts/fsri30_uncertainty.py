"""Read-only numerical audit and an analytical throughflow/splitting diagnostic.

The stirred-reactor example isolates operator splitting; it is NOT a battery
experiment or a replacement CPU solver. No fitted physical parameters here.
"""
import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from fsri30_data import save


def throughflow(step, residence=.0005, reaction_rate=2000., duration=1., supply=1.):
    """Constant-T CSTR: dF/dt=supply-(1/tau+k)*F, F(0)=0.

    Compare exact continuous evolution with resident R3 half-stages and an
    implicit whole-window transit stage, as in the current split integrator.
    Both close their chemical ledger; conservation alone does not ensure the
    right partition into reacted versus unburned exiting fuel.
    """
    count = math.ceil(duration/step)
    h = duration/count
    stock = heat = escaped = 0.
    for _ in range(count):
        before = stock*(-math.expm1(-reaction_rate*h/2))
        heat += before
        stock -= before
        stock = (stock+supply*h)/(1+h/residence)
        escaped += h/residence*stock
        after = stock*(-math.expm1(-reaction_rate*h/2))
        heat += after
        stock -= after
    rate = reaction_rate+1/residence
    exact_stock = supply*(-math.expm1(-rate*duration))/rate
    exact_heat = reaction_rate*supply/rate*(duration-(-math.expm1(-rate*duration))/rate)
    return dict(step_s=h, residence_s=residence, rate_per_s=reaction_rate,
        split_heat_j=heat, continuous_heat_j=exact_heat, split_escaped_j=escaped,
        split_remaining_j=stock, continuous_remaining_j=exact_stock,
        relative_heat_error=abs(heat-exact_heat)/exact_heat,
        chemical_closure_j=stock+heat+escaped-supply*duration)


def audit(folders):
    cases = []
    for folder in folders:
        with gzip.open(folder/'replay/run.json.gz', 'rt') as f:
            r = json.load(f)
        scene = json.loads((folder/'scenario.json').read_text())
        gas = scene['gas_network']
        recorded = sum(g['released_energy_j'][-1] for g in r['gas'].values())
        r2 = sum(r['released_energy_j_history'][-30:])
        present = set(r['gas'])
        omitted = [g['id'] for g in gas['compartments'] if g['id'] not in present]
        cases.append(dict(folder=str(folder), duration_s=r['advanced_to_s'],
            max_thermal_step_s=scene['simulation']['max_dt_s'],
            gas_requested_interval_s=gas['coupled_transport_interval_s'],
            pressure_iteration_cap=gas['averaged_pressure_iterations'],
            payload_iteration_cap=gas['averaged_payload_iterations'],
            recorded_r3_j=recorded, condensed_r2_j=r2, recorded_coupled_heat_j=r2+recorded,
            recorded_gas_nodes=len(present), total_gas_nodes=len(gas['compartments']),
            omitted_nodes=len(omitted), complete_global_r3_available=not omitted,
            note='Time interval AND algebraic caps changed together: not an isolated timestep experiment'))
    return dict(cases=cases, analytical_diagnostic=[throughflow(h) for h in (.05, .025, .0125, .01, .005)],
        interpretation='Throughflow can lose reaction opportunity under splitting despite exact ledgers; this is a diagnosed mechanism, not proof of the unique FSRI discrepancy cause')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', type=Path, action='append', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.case)
    save(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
