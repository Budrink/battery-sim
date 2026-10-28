#!/usr/bin/env python3
"""Propose one bounded minimax film refinement from complete native trials.

This is a local optimizer, NEVER a substitute for a native replay. Every
temperature/pressure residual remains unshifted and unscaled in the reports.
Only the five selected observations are fitted; C2/ISS is not consulted.
"""
import argparse
import copy
import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import calibrate_upv_r123 as cal
import upv_r123_data as dataio
from upv_final_fit import only_gas_contacts_changed
from upv_observation_policy import CALIBRATION_CHANNELS

AXES = ('reverse_profile_power', 'backing_fraction', 'wall_hot_scale')


def encode(parameters):
    return np.array([parameters[AXES[0]], math.log(parameters[AXES[1]]),
                     math.log(parameters[AXES[2]])])


def decode(vector, reference):
    result = copy.deepcopy(reference)
    result.update(zip(AXES, [float(vector[0]), math.exp(vector[1]), math.exp(vector[2])]))
    return result


def residual_vector(run, data, assessment, end=1100.):
    event = dataio.event(data)['time_s']
    if run.get('partial') or run['time_s'][-1] < end - 1e-7:
        raise ValueError('Never train the surrogate on an incomplete trajectory')
    grid = np.r_[np.arange(math.ceil(event+60.), end, 1.), end]
    observed = dataio.observed(data, grid)
    blocks = []
    for channel in CALIBRATION_CHANNELS:
        known = np.isfinite(observed[channel])
        error = np.interp(grid[known], run['time_s'], run['temperature_k'][channel])-observed[channel][known]
        tolerance = 10. if channel.startswith('C2_') else 5.
        blocks.append(error / (tolerance * math.sqrt(len(error))))
    pressure = np.interp(grid, run['time_s'], np.asarray(dataio.chamber_pressure_pa(run))/1e5)
    blocks.append((pressure-observed['pressure_bar'])/(.1*math.sqrt(len(grid))))
    peak = assessment['front']['features']['pressure']
    extras = [(peak['predicted']['peak']/peak['observed']['peak']-1.)/.05]
    extras.extend(assessment['arrival_times'][key]['error_s']/.5 for key in ('pressure','108'))
    return np.r_[np.concatenate(blocks), extras], [len(block) for block in blocks]


def propose(parameters, vectors, block_sizes, base_index=0, arrival_exploration_s=.5):
    """Convex local residual model with native-only final acceptance.

    Temperature and pressure-tail RMS are balanced by minimax loss; the
    pressure peak and two physical arrivals are separate linear constraints.
    A small trust box prevents large excursions beyond the measured probes.
    """
    if not .5<=arrival_exploration_s<=2.:
        raise ValueError('Arrival exploration must remain within [0.5,2] seconds')
    fixed = {k:v for k,v in parameters[base_index].items() if k not in AXES}
    if any({k:v for k,v in p.items()if k not in AXES}!=fixed for p in parameters):
        raise ValueError('Untracked parameters changed between derivative probes')
    coordinates = np.array([encode(p) for p in parameters])
    values = np.asarray(vectors)
    centre = coordinates[base_index]
    reference = values[base_index]
    design = coordinates-centre
    if np.linalg.matrix_rank(design) != len(AXES):
        raise ValueError('Need independent completed probes of all three film axes')
    jacobian, _, _, _ = np.linalg.lstsq(design, values-reference, rcond=None)
    offsets = np.r_[0, np.cumsum(block_sizes)]
    if values.shape[1] != offsets[-1]+3:
        raise ValueError('Residual block shape mismatch')

    def predict(x):
        return reference+(x-centre)@jacobian

    def norms(vector):
        return np.array([np.linalg.norm(vector[a:b]) for a,b in zip(offsets,offsets[1:])])

    def loss(x):
        score = norms(predict(x))
        return float(np.max(score)+.02*np.dot(score,score))

    radius = np.array([.5, .6, .25])
    # Power=2 already entered a sustained C3/ISS critical zone. Stay clear
    # of that measured failed bracket, rather than trusting an extrapolation.
    absolute_lo = np.array([2.5, math.log(.005), math.log(.6)])
    absolute_hi = np.array([5., math.log(.08), math.log(1.6)])
    bounds = list(zip(np.maximum(absolute_lo, centre-radius), np.minimum(absolute_hi, centre+radius)))
    limits=np.array([1.,arrival_exploration_s/.5,arrival_exploration_s/.5])
    constraints = dict(type='ineq', fun=lambda x: limits-np.abs(predict(x)[offsets[-1]:]))
    result = minimize(loss, centre, method='SLSQP', bounds=bounds, constraints=constraints,
                      options=dict(maxiter=500,ftol=1e-10))
    predicted = predict(result.x)
    feasible = bool(np.all(np.abs(predicted[offsets[-1]:])<=limits+1e-6))
    if not result.success or not feasible:
        raise ValueError('No feasible local refinement: '+str(result.message))
    proposal = decode(result.x, parameters[base_index])
    report = dict(axes=list(AXES), source_points=len(parameters), accepted_calibration=False,
        predicted_normalized_rmse=dict(zip([*CALIBRATION_CHANNELS,'pressure_tail'],map(float,norms(predicted)))),
        baseline_normalized_rmse=dict(zip([*CALIBRATION_CHANNELS,'pressure_tail'],map(float,norms(reference)))),
        predicted_peak_and_arrival_residuals=predicted[offsets[-1]:].tolist(),
        arrival_exploration_s=arrival_exploration_s,final_arrival_tolerance_s=.5,
        requires_physical_retiming=bool(np.any(np.abs(predicted[-2:])>1.)),
        trust_bounds=bounds, jacobian_condition=float(np.linalg.cond(design)),
        measured_local_model_residual=float(np.linalg.norm(design@jacobian-(values-reference))),
        requires_native_replay=True, requires_fresh_final_verification=True,
        objective_before=loss(centre),objective_predicted=loss(result.x))
    return proposal, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exploration-arrival-s',type=float,default=.5,
                        help='Only a search bracket, [0.5,2]; native final tolerance remains 0.5 s')
    args = parser.parse_args()
    data, _ = dataio.load('no-cooling')
    parameters, vectors, sizes, scenes, provenance = [], [], [], [], None
    for trial in args.trials:
        status=json.loads((trial.parent/'status.json').read_text())
        identity=tuple(status[k]for k in ('native_sha256','reference','overlay_base_sha256'))
        if provenance is not None and identity!=provenance:
            raise ValueError('Cannot mix cores, checkpoints or overlay pivots')
        provenance=identity
        scene = json.loads((trial/'scenario.json').read_text())
        if scene['metadata']['experiment']!='no-cooling':
            raise ValueError('Held-out data is not a calibration probe')
        if scenes and not only_gas_contacts_changed(scenes[0],scene):
            raise ValueError('Cannot combine different solid/source/gas-route physics')
        scenes.append(scene)
        parameters.append(scene['metadata']['final_calibration_trial']['parameters'])
        with gzip.open(trial/'replay/run.json.gz','rt') as stream:
            run = json.load(stream)
        assessment = json.loads((trial/'assessment.json').read_text())
        vector, blocks = residual_vector(run,data,assessment)
        if sizes and sizes != blocks:
            raise ValueError('Observation masks changed')
        sizes = blocks; vectors.append(vector)
    proposal, report = propose(parameters,vectors,sizes,arrival_exploration_s=args.exploration_arrival_s)
    report['source_trials'] = [str(path.resolve()) for path in args.trials]
    report['native_sha256'],report['checkpoint_reference'],report['overlay_base_sha256']=provenance
    args.output.mkdir()
    cal.save(args.output/'points.json',[proposal])
    cal.save(args.output/'prediction.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
