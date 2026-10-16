#!/usr/bin/env python3
"""Compare matching replay samples, inventories and cost; never fit physics."""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_upv import save
from report_upv_hybrid import difference


def compare(a, b):
    result = difference(a, b)

    def field(section, quantity):
        if a[section].keys() != b[section].keys():
            raise ValueError('Different topology: ' + section)
        return max(float(np.max(np.abs(np.asarray(v[quantity]) -
                                      b[section][key][quantity])))
                   for key, v in a[section].items())

    for quantity in ('mass_kg', 'chemical_j', 'oxidizer_kg'):
        result['gas_' + quantity] = field('gas', quantity)
    for quantity in ('opening_fraction', 'transferred_mass_kg',
                     'transferred_chemical_j', 'transferred_oxidizer_kg',
                     'fracture_work_j'):
        result['channel_' + quantity] = field('channels', quantity)
    if 'final_fields_k' in a and 'final_fields_k' in b:
        result['final_field_temperature_k'] = max(
            float(np.max(np.abs(np.asarray(v) - b['final_fields_k'][key])))
            for key, v in a['final_fields_k'].items())
    ledger_keys = ('max_thermal_residual_j', 'max_mass_residual_kg',
                   'max_chemical_residual_j', 'max_oxidizer_residual_kg')
    return dict(
        duration_s=a['time_s'][-1], samples=len(a['time_s']),
        partial=bool(a.get('partial') or b.get('partial')),
        differences=result,
        wall_s=dict(reference=a['wall_s'], candidate=b['wall_s']),
        wall_time_ratio=a['wall_s'] / b['wall_s'],
        profiles=dict(reference=a.get('solver_profile'), candidate=b.get('solver_profile')),
        conservation={name: {k: run[k] for k in ledger_keys}
                      for name, run in (('reference', a), ('candidate', b))},
        calibration_accepted=False,
        caveat='Differences at saved samples, not continuous extrema; not experimental validation. '
               'A time ratio is a speedup only for comparable controls and execution conditions.')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = compare(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
    if args.output:
        save(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
