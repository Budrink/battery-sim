#!/usr/bin/env python3
"""One bounded joint proposal from the large fixed-energy training perturbations.

This inexpensive residual interpolant only proposes a candidate. The reported
fit always comes from a new full native replay, not the interpolant. No claim
of optimizer convergence, identifiability or validated destruction physics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import scan_upv_rupture_regimes as scan
from calibrate_upv import save


def features(x):
    fraction, exposure, rate = x
    return np.array([fraction, fraction*fraction, exposure, fraction*exposure, rate])


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('--screen', type=Path, default=scan.model.ROOT/'data/calibration/upv-geometric-regimes-v12')
    cli.add_argument('--output', type=Path, default=scan.model.ROOT/'data/calibration/upv-geometric-joint-v12')
    cli.add_argument('--probe', type=Path, default=scan.model.previous.PROBE)
    args = cli.parse_args()
    m = scan.model; fp = m.fingerprint(args.probe)
    manifest = json.loads((args.screen/'manifest.json').read_text())
    if manifest['fingerprint'] != fp or manifest['validation_used_for_selection']:
        raise ValueError('Stale or non-training screen')
    if (args.output/'profile.json').exists():
        raise ValueError('Frozen output exists')
    seed = manifest['seed']; total = manifest['fixed_total_energy_j']
    base_fraction = seed['gas_energy_j']/total
    scene, prior = m.inputs('no-cooling')
    reports = {name: json.loads((args.screen/(name+'.json')).read_text()) for name in
               ['seed']+[case['id'] for case in manifest['cases']]}
    base = m.shell.score(reports['seed']['result'], prior)
    points, deltas = [], []
    for case in manifest['cases']:
        p = reports[case['id']]['parameters']
        x = [p['gas_energy_j']/total-base_fraction,
             np.log(p['exposure_scale']/seed['exposure_scale']),
             np.log(p['solid_rate']/seed['solid_rate'])]
        points.append(features(x))
        deltas.append(m.shell.score(reports[case['id']]['result'], prior)-base)
    design = np.asarray(points)
    if design.shape != (5, 5) or np.linalg.matrix_rank(design) != 5:
        raise ValueError('Expected the complete five-case experimental design')
    coefficients = np.linalg.solve(design, np.asarray(deltas))
    def residual(x):
        return base + features(x) @ coefficients
    lower = [.3-base_fraction, 0., 0.]
    upper = [.8-base_fraction, np.log(3.), np.log(5.)]
    # Several inexpensive starts of the interpolant, not native model searches.
    proposals = [least_squares(residual, start, bounds=(lower, upper)) for start in
                 [np.zeros(3), (np.array(lower)+np.array(upper))/2, np.array(upper)*.99]]
    proposal = min(proposals, key=lambda result: np.sum(result.fun**2))
    x = proposal.x
    case = dict(id='joint-proposal', gas_fraction=float(base_fraction+x[0]),
                exposure_factor=float(np.exp(x[1])), rate_factor=float(np.exp(x[2])))
    p = scan.parameters(seed, case)
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output/'proposal.json', dict(case=case, parameters=p, fingerprint=fp,
         fixed_total_energy_j=total, predicted_cost=float(proposal.fun@proposal.fun),
         scope='Residual interpolant of training runs; not a simulation or accepted fit',
         screen_manifest_sha256=m.digest(args.screen/'manifest.json'), script_sha256=m.digest(__file__)))
    print('joint proposal', json.dumps(case), 'predicted cost', float(proposal.fun@proposal.fun), flush=True)
    runner = m.native.Runner(args.probe, args.output)
    try:
        run = m.run_candidate(runner, scene, p)
        report = m.report('no-cooling', p, run, prior)
        save(args.output/'joint-proposal.json', report)
        reports['joint-proposal'] = report
        best_id = min(reports, key=lambda name: reports[name]['metrics']['objective_sum_squares'])
        best = reports[best_id]; p = best['parameters']
        history = json.loads((args.screen/'history.json').read_text()) + [scan.entry('joint-proposal', report['parameters'], run, prior)]
        for i, row in enumerate(history):
            row['candidate'] = i
        save(args.output/'history.json', history)
        # Freeze BEFORE checking the held-out case. A diagnostic freeze does not
        # promote the profile to an accepted model or change viewer defaults.
        save(args.output/'profile.json', dict(parameters=p, fingerprint=fp, protocol=m.PROTOCOL,
             validation_used_for_selection=False, calibration_accepted=False,
             fixed_total_energy_j=total, selected_case=best_id,
             search=dict(method='Five broad fixed-energy regimes and one interpolant-proposed native trial',
                         converged=False, best_cost=best['metrics']['objective_sum_squares']),
             screen_manifest_sha256=m.digest(args.screen/'manifest.json'), script_sha256=m.digest(__file__)))
        save(args.output/'fit.json', best)
        save(args.output/'no-cooling-fit-scenario.json', m.bake(scene, p))
        for name in ('heat-replay', 'numerics'):
            save(args.output/(name+'.json'), json.loads((m.OUT/(name+'.json')).read_text()))
        print('actual proposal', json.dumps(report['metrics']), flush=True)
        print('frozen diagnostic candidate', best_id, json.dumps(best['metrics']), flush=True)
    finally:
        runner.close()


if __name__ == '__main__':
    main()
