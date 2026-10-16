#!/usr/bin/env python3
"""Expensive-model trust-region fit with a Broyden Jacobian update.

One initial finite-difference sensitivity screen; subsequent accepted points
update that Jacobian by a secant. Every proposed coefficient set is evaluated
by the full native simulation, never accepted on surrogate predictions alone.
The final Jacobian is approximate and is NOT a parameter-confidence estimate.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares
import calibrate_upv_geometric_rupture as model
from calibrate_upv import save


class SecantJacobian:
    def __init__(self, residual, lower, upper, step=.05):
        self.residual = residual
        self.lower, self.upper, self.step = lower, upper, step
        self.x = self.r = self.matrix = None
        self.initial_matrix = None

    def __call__(self, x):
        r = self.residual(x)
        if self.matrix is None:
            columns = []
            for k in range(len(x)):
                dx = self.step if x[k]+self.step <= self.upper[k] else -self.step
                y = x.copy(); y[k] += dx
                columns.append((self.residual(y)-r)/dx)
            self.matrix = np.column_stack(columns)
            self.initial_matrix = self.matrix.copy()
        else:
            delta = x-self.x; norm2 = float(delta@delta)
            if norm2 > 1e-12:
                self.matrix += np.outer(r-self.r-self.matrix@delta, delta)/norm2
        self.x, self.r = x.copy(), r.copy()
        # scipy may scale its Jacobian in place. Never expose stored secant state.
        return self.matrix.copy()


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('--output', type=Path, default=model.OUT)
    cli.add_argument('--probe', type=Path, default=model.previous.PROBE)
    cli.add_argument('--max-evaluations', type=int, default=16)
    args = cli.parse_args(); out = args.output
    fp = model.fingerprint(args.probe)
    if (out/'profile.json').exists():
        raise ValueError('Profile is frozen; use another output directory')
    for name in ('heat-replay', 'numerics'):
        if json.loads((out/(name+'.json')).read_text())['fingerprint'] != fp:
            raise ValueError('Repeat heat/numerics proof after model changes')
    scene, prior = model.inputs('no-cooling'); initial = model.initial_parameters()
    names = model.NAMES
    origin = np.log([initial[k] for k in names])
    lower, upper = np.log(np.array([model.LIMITS[k] for k in names]).T)-origin
    baseline = json.loads((out/'baseline.json').read_text())
    if baseline['parameters'] != initial:
        raise ValueError('Numerical baseline parameters changed')
    cache = {tuple(np.zeros(len(names))): model.shell.score(baseline['result'], prior)}
    best = [float(cache[tuple(np.zeros(len(names)))]@cache[tuple(np.zeros(len(names)))]), initial, baseline['result']]
    history = [dict(candidate=0, cost=best[0], parameters=initial, metrics=baseline['metrics'], reused_baseline=True)]
    runner = model.native.Runner(args.probe, out); started = time.monotonic()
    try:
        def residual(x):
            key = tuple(x)
            if key in cache:
                return cache[key]
            p = {k:float(np.clip(v, *model.LIMITS[k])) for k, v in zip(names, np.exp(x+origin))}
            run = model.run_candidate(runner, scene, p)
            r = model.shell.score(run, prior); metrics = model.shell.metrics(run, prior)
            entry = dict(candidate=len(history), cost=float(r@r), parameters=p,
                         metrics=metrics, energy_ledger=model.previous.ledger(run), wall_s=run['wall_s'], steps=run['steps'])
            history.append(entry); cache[key] = r
            if entry['cost'] < best[0]:
                best[:] = [entry['cost'], p, run]
                save(out/'checkpoint.json', entry); save(out/'best-run.json', run)
            save(out/'history.json', history)
            print(f"candidate {entry['candidate']} cost={entry['cost']:.6f} best={best[0]:.6f} T={metrics['witness']['rmse_k']:.4f} K peak={metrics['predicted_peak_bar']:.4f} bar native={run['wall_s']:.1f}s elapsed={time.monotonic()-started:.0f}s", flush=True)
            return r

        jacobian = SecantJacobian(residual, lower, upper)
        initial_jacobian = jacobian(np.zeros(len(names)))
        save(out/'sensitivity.json', dict(kind='initial full finite difference, log step .05',
             parameters=initial, column_norms=dict(zip(names, np.linalg.norm(initial_jacobian, axis=0).tolist())),
             singular_values=np.linalg.svd(initial_jacobian, compute_uv=False).tolist()))
        print('initial sensitivities', json.dumps(dict(zip(names, np.linalg.norm(initial_jacobian, axis=0).tolist()))), flush=True)
        result = least_squares(residual, np.zeros(len(names)), bounds=(lower, upper),
                               jac=jacobian, x_scale='jac', max_nfev=args.max_evaluations,
                               ftol=5e-4, xtol=1e-4, gtol=1e-4)
        p, run = best[1:]
        profile = dict(parameters=p, fitted=list(names), fixed_priors=dict(fracture_work_j_m2=model.FIXED_WORK),
                       fingerprint=fp, protocol=model.PROTOCOL, validation_used_for_selection=False,
                       search=dict(method='bounded trust-region least squares with initial finite differences and Broyden updates',
                                   success=bool(result.success), message=result.message, nfev=result.nfev,
                                   candidates=len(history)-1, reused_baseline=True, initial_cost=history[0]['cost'], best_cost=best[0],
                                   jacobian_note='Final secant approximation is not an identifiability/confidence estimate'),
                       input_sha256={str(pth.relative_to(model.ROOT)):model.digest(pth) for pth in
                                     (Path(__file__), model.PRIOR/'profile.json', model.PRIOR/'fit.json',
                                      model.PRIOR/'no-cooling-fit-scenario.json', model.shell.HEAT/'profile.json')})
        save(out/'profile.json', profile)
        save(out/'fit.json', model.report('no-cooling', p, run, prior))
        save(out/'no-cooling-fit-scenario.json', model.bake(scene, p))
        print('frozen training fit', json.dumps(profile), flush=True)
    finally:
        runner.close()


if __name__ == '__main__':
    main()
