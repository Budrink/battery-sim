"""Fresh diagnostic for lower valve threshold and richer, faster burning fuel."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from fsri30_data import save


def candidate(source, valve=.75, fuel=2., rate=1.5):
    if not all(math.isfinite(v) and v>0 for v in (valve,fuel,rate)) or valve>1:
        raise ValueError('Invalid diagnostic multipliers')
    scene=deepcopy(source);network=scene['gas_network']; changed=0
    for edge in network['channels']:
        v=edge.get('damage_opening',{}).get('valve')
        if v is not None:v['pressure_factor']*=valve;changed+=1
    if not changed:raise ValueError('Missing safety valve')
    # This is a fuel inventory owned by liquid and advected with its vapour;
    # keep liquid and gas species descriptions identical for ledger closure.
    network['vapor_fuel_heat_j_kg']*=fuel
    for cell in scene['cells']:
        if cell.get('role')=='battery':cell['conversion']['volatile_phase']['chemical_j_per_kg']*=fuel
    for gas in network['compartments']:
        if 'reaction_rate_per_s' in gas:gas['reaction_rate_per_s']*=rate
    scene['metadata']['fuel_valve_probe']=dict(valve_pressure_factor_multiplier=valve,
        volatile_fuel_chemical_energy_multiplier=fuel,r3_rate_multiplier=rate,
        assumption='unmeasured energetic fuel inventory; conserved, not heat added during transfer',
        valve_count=changed,calibrated=False,validation_passed=False)
    return scene


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    s=candidate(json.loads(a.source.read_text()));save(a.output,s);print(s['metadata']['fuel_valve_probe'])
