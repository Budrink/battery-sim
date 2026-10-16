#!/usr/bin/env python3
"""Summarize completed short audits and the explicitly incomplete UPV attempt."""
import argparse
import json
from pathlib import Path
import numpy as np
from calibrate_upv import save


def difference(a,b):
    if not np.allclose(a['time_s'],b['time_s'],rtol=0,atol=1e-9):
        raise ValueError('Use matching observation times, not interpolated peaks')
    return dict(
        witness_temperature_k=max(float(np.max(np.abs(np.asarray(v)-b['temperature_k'][k]))) for k,v in a['temperature_k'].items()),
        gas_temperature_k=max(float(np.max(np.abs(np.asarray(v['temperature_k'])-b['gas'][k]['temperature_k']))) for k,v in a['gas'].items()),
        gas_pressure_pa=max(float(np.max(np.abs(np.asarray(v['pressure_pa'])-b['gas'][k]['pressure_pa']))) for k,v in a['gas'].items()),
        chamber_pressure_pa=float(np.max(np.abs(np.asarray(a['gas']['chamber']['pressure_pa'])-b['gas']['chamber']['pressure_pa']))))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('directory',type=Path);args=p.parse_args();root=args.directory
    def read(path):return json.loads((root/path).read_text())
    strict=read('coarse-window/run.json');relaxed=read('relaxed-smoke/run.json');fine=read('relaxed-refined-smoke/run.json')
    partial=read('baseline/partial.json')
    report=dict(calibration_accepted=False,
        short_duration_s=relaxed['time_s'][-1],
        short_wall_s=dict(strict=strict['wall_s'],relaxed=relaxed['wall_s'],refined=fine['wall_s']),
        pressure_tolerance_sensitivity=difference(strict,relaxed),
        coupled_short_refinement=difference(relaxed,fine),
        caveat='Short checks are before casing breakthrough; not full destruction convergence.',
        full_attempt=read('baseline/status.json'),
        partial_steps=partial['steps'],partial_wall_s=partial['wall_s'],
        partial_conservation={k:partial[k] for k in ('max_thermal_residual_j','max_mass_residual_kg',
            'max_chemical_residual_j','max_oxidizer_residual_kg')},
        heating_regression=read('heating-regression.json'))
    save(root/'integration-summary.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
