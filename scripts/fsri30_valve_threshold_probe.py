"""Single-factor fresh-state probe of safety-valve opening pressure."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from fsri30_data import save


def candidate(source, factor):
    if not math.isfinite(factor) or not 0 < factor <= 4:raise ValueError('Expected valve multiplier in (0, 4]')
    result=deepcopy(source);count=0
    for edge in result['gas_network']['channels']:
        valve=edge.get('damage_opening',{}).get('valve')
        if valve is not None:valve['pressure_factor']*=factor;count+=1
    if not count:raise ValueError('No safety valve in source')
    result['metadata']['valve_threshold_probe']=dict(multiplier=factor,valves=count,
        changes='valve pressure factor only; all heat, gas and chemistry settings unchanged',
        calibrated=False,validation_passed=False)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--factor',type=float,default=.5);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    scene=candidate(json.loads(a.source.read_text()),a.factor);save(a.output,scene)
    print(scene['metadata']['valve_threshold_probe'])
