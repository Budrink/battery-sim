#!/usr/bin/env python3
"""Compare two production CUDA snapshot auditors on the same hot state."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--scenario',type=Path,required=True);ap.add_argument('--snapshot',type=Path,required=True)
    args=ap.parse_args();env=dict(os.environ,BATTERY_GAS_SCENARIO=str(args.scenario.resolve()),
                                 BATTERY_AVERAGE_PRESSURE_ITERATIONS='8192')
    def run(path):
        return json.loads(subprocess.check_output([str(path.resolve()),str(args.snapshot.resolve()),'averaged'],env=env))
    a,b=run(args.baseline),run(args.candidate)
    if a.get('error_code') or b.get('error_code'):raise RuntimeError(f'Snapshot rejected: {a.get("error_code")}, {b.get("error_code")}')
    if [x['id'] for x in a['edges']]!=[x['id'] for x in b['edges']]:raise ValueError('Edge topology changed')
    def stats(field):
        x=[v[field] for v in a['edges']];y=[v[field] for v in b['edges']]
        total=sum(abs(v) for v in x)
        return dict(max_absolute=max((abs(i-j) for i,j in zip(x,y)),default=0.),
                    total_absolute_difference=sum(abs(i-j) for i,j in zip(x,y)),
                    normalized_l1=sum(abs(i-j) for i,j in zip(x,y))/max(total,1e-300))
    print(json.dumps(dict(baseline_pressure_ms=a['pressure_wall_ms'],candidate_pressure_ms=b['pressure_wall_ms'],
        baseline_iterations=a['pressure_iterations_used'],candidate_iterations=b['pressure_iterations_used'],
        baseline_residual=a['pressure_relative_residual'],candidate_residual=b['pressure_relative_residual'],
        baseline_balances=[a['mass_residual_kg'],a['energy_residual_j']],
        candidate_balances=[b['mass_residual_kg'],b['energy_residual_j']],
        mass_packets=stats('accepted_mass_kg'),energy_packets=stats('pressure_energy_from_to_j')),indent=2))


if __name__=='__main__':main()
