#!/usr/bin/env python3
"""Refine both the CFL limit and max_dt; a max_dt-only check can be a no-op."""
import argparse
import copy
from pathlib import Path

import numpy as np

from calibrate_upv import save
import refit_upv_training as refit


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--probe', type=Path, default=refit.model.previous.PROBE)
    args = cli.parse_args(); out = args.output/'heat'
    target = out/'numerics-effective.json'
    if target.exists():
        raise ValueError('Do not overwrite refinement proof')
    scene = refit.read(out/'no-cooling-fitted-scenario.json')
    runner = refit.model.native.Runner(args.probe, out)
    try:
        coarse = runner.run(scene)
        refined = copy.deepcopy(scene)
        refined['simulation']['stability_factor'] *= .5
        fine = runner.run(refined, .5)
        change = max(float(np.max(np.abs(np.asarray(v)-fine['temperature_k'][k])))
                     for k, v in coarse['temperature_k'].items())
        proof = dict(max_temperature_change_k=change, coarse_steps=coarse['steps'],
                     refined_steps=fine['steps'], coarse_energy_residual_j=coarse['energy_residual_j'],
                     refined_energy_residual_j=fine['energy_residual_j'],
                     controls='Both stability_factor and max_dt halved; not max_dt alone',
                     scene_sha256=refit.model.digest(out/'no-cooling-fitted-scenario.json'),
                     probe_sha256=refit.model.digest(args.probe))
        save(target, proof)
        if fine['steps'] < 1.5*coarse['steps'] or change > .25:
            raise ValueError('Refinement ineffective or too large a numerical error')
        print(proof, flush=True)
    finally:
        runner.close()


if __name__ == '__main__':
    main()
