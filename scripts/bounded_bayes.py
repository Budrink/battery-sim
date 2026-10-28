"""Small sequential, constrained, cost-aware Gaussian-process optimizer.

Only NumPy/SciPy are needed. It proposes unit-cube parameters; it neither
implements physics nor edits observations. A serialized observation list and
seed completely determine the next point, including after a process restart.
"""
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import qmc


def matern52(a, b, length):
    distance = np.sqrt(np.maximum(0., np.sum(((a[:, None, :]-b[None, :, :])/length)**2, axis=2)))
    r = math.sqrt(5.)*distance
    return (1.+r+r*r/3.)*np.exp(-r)


class GaussianProcess:
    """Exact GP with regularized length scale and a small fixed noise floor.

    Nine free ARD length scales are not identifiable from a dozen simulations;
    fit one isotropic scale instead. Unit-cube/log physical transforms make
    that choice explicit. Posterior uncertainty drives exploration, not a
    reported physical confidence interval on this incomplete model.
    """
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.x.ndim != 2 or not len(y) or self.x.shape[0] != len(y) or not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(y)):
            raise ValueError('Invalid GP observations')
        self.center = float(np.mean(y))
        self.scale = max(float(np.std(y)), .05)
        self.y = (y-self.center)/self.scale

        def factor(theta):
            length, amplitude = np.exp(theta)
            covariance = amplitude*matern52(self.x, self.x, length)
            covariance.flat[::len(y)+1] += 1e-5
            return cho_factor(covariance, lower=True, check_finite=False)

        def likelihood(theta):
            try:
                f = factor(theta)
                value = .5*float(self.y@cho_solve(f, self.y)) + np.log(np.diag(f[0])).sum()
                return value+.15*((theta[0]-math.log(.5))/1.2)**2+.05*theta[1]**2
            except np.linalg.LinAlgError:
                return 1e12

        initial = np.log([.5, 1.])
        fit = minimize(likelihood, initial, bounds=[(math.log(.06), math.log(3.)), (math.log(.05), math.log(20.))],
                       method='L-BFGS-B', options=dict(maxiter=60))
        theta = fit.x if fit.success and np.all(np.isfinite(fit.x)) else initial
        self.length, self.amplitude = np.exp(theta)
        self.factor = factor(theta)
        self.alpha = cho_solve(self.factor, self.y)

    def predict(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        k = self.amplitude*matern52(self.x, x, self.length)
        mean = self.center+self.scale*(k.T@self.alpha)
        v = solve_triangular(self.factor[0], k, lower=True, check_finite=False)
        variance = np.maximum(1e-10, self.amplitude-np.sum(v*v, axis=0))
        return mean, self.scale*np.sqrt(variance)


def expected_improvement(mean, sd, best):
    sd = np.maximum(1e-12, np.asarray(sd))
    improvement = best-np.asarray(mean)
    z = improvement/sd
    return np.maximum(0., improvement*ndtr(z)+sd*np.exp(-.5*z*z)/math.sqrt(2.*math.pi))


def trust_region(observations):
    radius = .75
    success = failure = 0
    best = math.inf
    for row in observations:
        value = row.get('objective') if row.get('complete') and row.get('constraint', 1.) <= 0. else None
        improved = value is not None and (not math.isfinite(best) or value < best-max(.01, .01*abs(best)))
        if improved:
            best = value
            success += 1
            failure = 0
            if success >= 3:
                radius = min(1., radius*1.4)
                success = 0
        else:
            failure += 1
            success = 0
            if failure >= 4:
                radius *= .6
                failure = 0
        if radius < .08:
            radius = .75  # Exhausted local region: explicitly restart broadly.
    return radius


def propose(observations, dimension, initial, *, seed=20261023, initial_design=12, pool_size=4096, admissible=None):
    """Return point plus acquisition provenance; infeasible/censored != good.

    Partial/pruned runs train only the feasibility model, never a fake final
    objective. Full feasible observations train log(1+loss). Global Sobol
    exploration is retained even after the trust region becomes local.
    """
    if dimension < 1 or len(initial) != dimension:
        raise ValueError('Invalid search dimension')
    step = len(observations)
    if not step:
        if admissible is not None and not admissible(initial):
            raise ValueError('Declared initial point violates a known prior')
        return np.array(initial, dtype=float), dict(method='declared-initial-point')
    used = np.asarray([r['x'] for r in observations], dtype=float)
    rng_seed = int(seed+104729*step) % (2**32)
    sobol = qmc.Sobol(dimension, scramble=True, seed=rng_seed)
    pool = sobol.random_base2(int(math.ceil(math.log2(pool_size))))
    feasible = [r for r in observations if r.get('complete') and r.get('constraint', 1.) <= 0. and r.get('objective') is not None]
    radius = trust_region(observations)
    if feasible and step >= initial_design:
        best_row = min(feasible, key=lambda r: r['objective'])
        center = np.asarray(best_row['x'])
        lo, hi = np.maximum(0., center-radius/2), np.minimum(1., center+radius/2)
        local_n = 3*len(pool)//4
        pool[:local_n] = lo+(hi-lo)*pool[:local_n]
    # Exclude duplicates in both initial and Bayesian designs.
    if admissible is not None:
        pool = pool[np.array([admissible(p) for p in pool], dtype=bool)]
        if not len(pool):
            raise ValueError('No proposal satisfies known parameter priors')
    distance = np.min(np.linalg.norm(pool[:, None, :]-used[None, :, :], axis=2), axis=1)
    pool, distance = pool[distance > 1e-5], distance[distance > 1e-5]
    if not len(pool):
        raise ValueError('No distinct bounded proposal')
    if step < initial_design:
        # Maximin space-filling Sobol design; does not look at the measured loss.
        index = int(np.argmax(distance))
        return pool[index], dict(method='sobol-maximin', minimum_distance=float(distance[index]))
    probability = np.ones(len(pool))
    if any(r.get('constraint', 1.) > 0. for r in observations):
        gp = GaussianProcess(used, [float(r.get('constraint', 1.)) for r in observations])
        mean, sd = gp.predict(pool)
        probability = np.clip(ndtr(-mean/sd), .005, 1.)
    if not feasible:
        acquisition = probability*(.05+distance)
        index = int(np.argmax(acquisition))
        return pool[index], dict(method='feasibility-exploration', feasible_probability=float(probability[index]), radius=radius)
    x = np.array([r['x'] for r in feasible])
    y = np.log1p([max(0., r['objective']) for r in feasible])
    gp = GaussianProcess(x, y)
    mean, sd = gp.predict(pool)
    ei = expected_improvement(mean, sd, float(np.min(y)))
    cost = np.ones(len(pool))
    if len(feasible) >= 4 and all(r.get('wall_s', 0.) > 0. for r in feasible):
        cost_mean, _ = GaussianProcess(x, np.log([r['wall_s'] for r in feasible])).predict(pool)
        cost = np.exp(np.clip(cost_mean, -5., 15.))
    acquisition = ei*probability/cost**.25
    # Every eighth Bayesian point deliberately explores high posterior variance.
    explore = (step-initial_design) % 8 == 7
    if explore:
        acquisition = sd*probability*(.1+distance)/cost**.25
    index = int(np.argmax(acquisition))
    return pool[index], dict(method='constrained-gp-uncertainty' if explore else 'constrained-gp-expected-improvement',
        feasible_probability=float(probability[index]), expected_improvement=float(ei[index]),
        predicted_log_loss=float(mean[index]), posterior_sd=float(sd[index]),
        predicted_wall_s=float(cost[index]), radius=radius, kernel_length=float(gp.length))
