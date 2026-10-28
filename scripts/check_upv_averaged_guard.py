#!/usr/bin/env python3
"""Audit an exact rejected transport window; no thermal/chemical recalibration."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import calibrate_upv_r123 as cal

AUDIT = Path('/tmp/battery-sim-cuda-v1/battery_sim_gas_limiter_snapshot')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--scenario', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    source = json.loads(args.snapshot.read_text())
    source['scenario_path'] = str(args.scenario.resolve())
    path = out / 'input.json'
    cal.save(path, source)
    state = dict(diagnostic_only=True, calibrated=False, snapshot=str(args.snapshot.resolve()),
                 attempts=source['attempts'], cases=[])
    cases = [('pressure', n) for n in (96, 128, 192, 256, 512)]
    cases += [('payload', n) for n in (16, 32, 64, 128)]
    for kind, iterations in cases:
        key = 'BATTERY_AVERAGE_PRESSURE_ITERATIONS' if kind == 'pressure' else 'BATTERY_AVERAGE_PAYLOAD_ITERATIONS'
        env = dict(os.environ, **{key: str(iterations)})
        result = subprocess.run([str(AUDIT), str(path), 'averaged'], env=env,
                                text=True, capture_output=True, timeout=120.)
        if result.returncode:
            raise RuntimeError(result.stderr)
        report = json.loads(result.stdout)
        cal.save(out / f'{kind}-{iterations}.json', report)
        row = dict(varied=kind, iterations=iterations, error_code=report.get('error_code', 0),
                   pressure_iterations_used=report.get('pressure_iterations_used'),
                   pressure_relative_residual=report.get('pressure_relative_residual'))
        if row['error_code']:
            row['attempts'] = report['attempts']
            row['lowest_entropy_nodes'] = sorted(report['rejected_nodes'],
                key=lambda n: n['entropy_change_j_per_k'])[:12]
        else:
            row.update(net_mass_scale=report['net_mass_scale'], entropy_backtracks=report['entropy_backtracks'],
                       mass_residual_kg=report['mass_residual_kg'], energy_residual_j=report['energy_residual_j'])
        state['cases'].append(row)
        cal.save(out / 'report.json', state)
        print(json.dumps({k: v for k, v in row.items() if k != 'lowest_entropy_nodes'}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
