"""Select ONE fixed numerical regime before Bayesian physical calibration.

Refinement is driven by numerical discrepancies, not experimental residuals.
No physical parameter bounds, minimum time step or observation weights change.
"""
from fsri30_objective import numerical_comparison
import numpy as np


def check_pair(a, b):
    result = numerical_comparison(a['run'], b['run'])
    ca, cb = a['score']['cascade'], b['score']['cascade']
    result['cascade_counts'] = [ca['reacted_components'], cb['reacted_components']]
    result['same_cascade_count'] = ca['reacted_components'] == cb['reacted_components']
    # Overall waveform RMS must not hide a drift in cascade chronology.
    result['same_conversion_order'] = [e['component'] for e in ca['events']] == [e['component'] for e in cb['events']]
    times_a = {e['component']: e['time_s'] for e in ca['events']}
    times_b = {e['component']: e['time_s'] for e in cb['events']}
    result['same_converted_components'] = times_a.keys() == times_b.keys()
    result['maximum_conversion_time_difference_s'] = max(
        (abs(times_a[k]-times_b[k]) for k in times_a), default=0.) if result['same_converted_components'] else None
    # Symmetry-related near-simultaneous failures may swap in a list; component
    # chronology is compared by identity, without imposing a fictitious order.
    result['accepted'] &= result['same_cascade_count'] and result['same_converted_components']
    if result['same_converted_components']:
        result['accepted'] &= result['maximum_conversion_time_difference_s'] <= 20.
    result['reference'] = a['folder']
    result['refined'] = b['folder']
    return result


def select(evaluate_one, max_accuracy=3, on_comparison=None):
    """evaluate_one(accuracy, layers) -> {run, score, folder}.

    Keep algebraic work bounded, gas interval >= preserved .01s floor and max thermal step
    >=.025s in this prescribed ladder. The global .01s floor never changes.
    At each level compare with the NEXT level, not a lossy old baseline.
    A final chosen regime is frozen for all proposed physical parameters.
    """
    cache = {}
    comparisons = []

    def record(comparison):
        comparisons.append(comparison)
        if on_comparison is not None:
            on_comparison(list(comparisons))

    def get(level, layers):
        key = (level, layers)
        if key not in cache:
            value = evaluate_one(level, layers)
            if not value['score']['complete'] or value['score']['constraint'] > 0.:
                raise RuntimeError('Numerical control could not complete: '+value['folder'])
            cache[key] = value
        return cache[key]

    selected = None
    for level in range(max_accuracy):
        comparison = dict(kind='time', accuracy_level=level, layers=1,
                          **check_pair(get(level, 1), get(level+1, 1)))
        record(comparison)
        if comparison['accepted']:
            selected = level
            break
    if selected is None:
        return dict(accepted=False, reason='Time refinement did not converge within declared safe ladder',
                    comparisons=comparisons)
    for layers, next_layers in ((1, 2), (2, 4)):
        comparison = dict(kind='space', accuracy_level=selected, layers=layers,
                          **check_pair(get(selected, layers), get(selected, next_layers)))
        record(comparison)
        if comparison['accepted']:
            # Space refinement can make time stiffness worse: check the chosen
            # spatial representation against the next time level as well.
            if layers != 1:
                joint = dict(kind='time-after-space', accuracy_level=selected, layers=layers,
                    **check_pair(get(selected, layers), get(selected+1, layers)))
                record(joint)
                if not joint['accepted']:
                    return dict(accepted=False, reason='Joint time/space refinement has not converged', comparisons=comparisons)
            return dict(accepted=True, accuracy_level=selected, layers=layers,
                reference=cache[(selected, layers)]['folder'], comparisons=comparisons,
                note='Fixed numerical policy, not an optimization variable; repeat on best physical candidate')
    return dict(accepted=False, reason='Near-field space refinement did not converge', comparisons=comparisons)


def exploration_policy(controls, strict):
    """Permit bounded DEVELOPMENT search without asserting R3 convergence.

    Keep the strict 5% branch test untouched. An unresolved minor branch may
    inform provisional proposals only when whole temperature envelopes,
    identities/count/timing and the L1 perturbation of recorded coupled heat
    meet their existing bounds. Missing full gas histories and untested space
    remain explicit. A candidate from this policy is NOT numerically accepted.
    """
    eligible = []
    for level in range(3):
        if (level, 1) not in controls or (level+1, 1) not in controls:
            continue
        a, b = controls[(level, 1)], controls[(level+1, 1)]
        if any(not x['score']['complete'] or x['score']['constraint'] > 0. for x in (a, b)):
            continue
        comparison = check_pair(a, b)
        def budget(run):
            return np.array([sum(run['released_energy_j_history'][-30:]),
                             sum(g['released_energy_j'][-1] for g in run['gas'].values())])
        ea, eb = budget(a['run']), budget(b['run'])
        coupled = float(np.abs(ea-eb).sum()/max(100., ea.sum(), eb.sum()))
        temperature = comparison.get('envelope_rms_relative_kelvin', float('inf'))
        timing = comparison.get('maximum_conversion_time_difference_s')
        if (temperature <= .05 and coupled <= .05 and comparison['same_converted_components']
                and timing is not None and timing <= 20.):
            eligible.append((a['score']['wall_s'], level, a, comparison, coupled))
    if not eligible:
        return dict(strict, search_allowed=False, provisional=True)
    _, level, reference, comparison, coupled = min(eligible, key=lambda x: x[0])
    return dict(accepted=False, search_allowed=True, provisional=True, numerically_verified=False,
        accuracy_level=level, layers=1, reference=reference['folder'],
        strict_preflight=strict, supporting_comparison=comparison,
        recorded_coupled_heat_l1_relative=coupled,
        warnings=['R3 branch convergence remains UNCONFIRMED; strict 5% test has NOT passed',
                  'Recorded gas histories are a subset, not the full global R3 budget',
                  'Spatial refinement is deferred to promising-candidate checks'],
        note='Permission for exploratory parameter search ONLY, never acceptance of cascade/calibration/validation')
