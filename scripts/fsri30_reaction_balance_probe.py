"""Fresh diagnostic of fuel energy and R3 rate; does not alter R2 directly."""
import json
from copy import deepcopy
from pathlib import Path
from fsri30_data import save


def candidate(source, fuel=2., r3=.4):
    if not (0 < fuel <= 10 and 0 < r3 <= 10):raise ValueError('Invalid diagnostic multipliers')
    s=deepcopy(source);g=s['gas_network']
    g['vapor_fuel_heat_j_kg']*=fuel
    for c in s['cells']:
        if c.get('role')=='battery':c['conversion']['volatile_phase']['chemical_j_per_kg']*=fuel
    for n in g['compartments']:
        if 'reaction_rate_per_s' in n:n['reaction_rate_per_s']*=r3
    s['metadata']['reaction_balance_probe']=dict(volatile_fuel_multiplier=fuel,r3_rate_multiplier=r3,
        meaning='additional initial liquid fuel reserve; existing R2 gas fuel reserve remains separate',
        caveat='double counting and total gas fuel energy must be audited before physical interpretation',
        measured=False,validation_passed=False)
    return s


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--fuel',type=float,default=2.);p.add_argument('--r3',type=float,default=.4);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    s=candidate(json.loads(a.source.read_text()),a.fuel,a.r3);save(a.output,s);print(s['metadata']['reaction_balance_probe'])
