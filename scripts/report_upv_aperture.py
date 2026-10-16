#!/usr/bin/env python3
"""Plots and diagnostics of a frozen aperture fit; never selects parameters."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import calibrate_upv_aperture as fit
from calibrate_upv import save


def sensitivity_plot(output):
    screen = json.loads((output/'scan.json').read_text())
    trials = {x['label']: x for x in screen['trials']}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, labels, parameter, ylabel in [
        (axes[0], ['h-100.0', 'initial', 'h-1000000.0', 'h-10000000.0'], 'surface_h', 'Objective sum of squares'),
        (axes[1], ['work-1.0', 'initial', 'work-1000000.0'], 'fracture_work_j_m2', 'Maximum witness change from prior [K]')]:
        selected = [trials[k] for k in labels]
        y = [r['metrics']['objective_sum_squares'] if parameter == 'surface_h'
             else r['change_from_initial']['max_witness_change_k'] for r in selected]
        ax.semilogx([r['parameters'][parameter] for r in selected], y, 'o-', color='#277da8')
        ax.set(xlabel=parameter, ylabel=ylabel); ax.grid(alpha=.2)
    fig.suptitle('Training-only sensitivity; exploratory bounds, other parameters fixed')
    fig.savefig(output/'sensitivity.png', dpi=160); plt.close(fig)
    return screen


def objective_blocks(report):
    """Same fixed residual blocks as the optimizer, for transparent weighting."""
    run = report['result']; at = np.asarray(run['time_s'])
    predicted = np.asarray(run['gas']['chamber']['pressure_pa'])/1e5
    observed = np.asarray(report['observed_pressure_bar'], dtype=float)
    blocks = {}
    for name, mask, scale in [('pressure_early', (at>0)&(at<=30), .25),
                               ('pressure_late', at>30, .05)]:
        mask &= np.isfinite(observed)
        blocks[name] = float(np.mean(((predicted[mask]-observed[mask])/scale)**2))
    for channel, metric in report['metrics']['witness']['per_channel'].items():
        blocks[channel] = (metric['rmse_k']/5)**2
    return blocks


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--output', type=Path, default=fit.OUT)
    ap.add_argument('--probe', type=Path, default=fit.PROBE)
    ap.add_argument('--dense', action='store_true')
    ap.add_argument('--baseline-validation', action='store_true')
    ap.add_argument('--dense-baseline', action='store_true')
    args = ap.parse_args()
    profile = json.loads((args.output/'profile.json').read_text())
    if profile['fingerprint'] != fit.fingerprint(args.probe):
        raise ValueError('Frozen model changed')
    pressure, paxes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    thermal, taxes = plt.subplots(4, 2, figsize=(12, 11), constrained_layout=True)
    energy, eaxes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    routing, raxes = plt.subplots(4, 2, figsize=(12, 11), constrained_layout=True)
    source_paths = [fit.ENERGY_PRIOR, fit.SHELL_PRIOR, fit.shell.HEAT/'profile.json']
    for mode, case in [('fit', 'no-cooling'), ('validate', 'active-cooling')]:
        source_paths += [fit.shell.PRIOR/(mode+'.json'),
                         fit.shell.PRIOR/(case+'-'+mode+'-scenario.json'),
                         fit.ROOT/'data/experiments/upv'/(case+'.csv')]
    summary = dict(profile=profile, experiments={}, sensitivity=sensitivity_plot(args.output),
                   frozen_shell_parameters=json.loads(fit.SHELL_PRIOR.read_text())['parameters'],
                   heat_replay=json.loads((args.output/'heat-replay.json').read_text()),
                   input_sha256={str(p.relative_to(fit.ROOT)): fit.shell.digest(p) for p in source_paths})
    runner = fit.gas.t.Runner(args.probe, args.output)
    try:
        for col, (mode, case) in enumerate([('fit', 'no-cooling'), ('validate', 'active-cooling')]):
            report = json.loads((args.output/(mode+'.json')).read_text())
            run = report['result']; at = np.asarray(run['time_s'])
            legacy_path = args.output/(case+'-legacy-v9.json')
            if case == 'no-cooling':
                legacy = json.loads((args.output/'legacy-v9-run.json').read_text())
            elif args.baseline_validation:
                scene, _ = fit.shell.inputs(case)
                legacy = runner.run(fit.bake(scene, fit.initial_parameters(), enabled=False))
                save(legacy_path, legacy)
            else:
                legacy = json.loads(legacy_path.read_text())
            dense_path = args.output/(case+'-dense.json')
            if args.dense:
                scene = json.loads((args.output/(case+'-'+mode+'-scenario.json')).read_text())
                scene['simulation']['duration_s'] = 30.
                scene['failure_sample_times_s'] = np.unique(np.r_[np.arange(0, 5, .01), np.arange(5, 30, .1),
                                                                  30., at[at<30]]).tolist()
                dense = runner.run(scene); save(dense_path, dense)
                fine_scene = copy.deepcopy(scene)
                fine_scene['simulation']['stability_factor'] *= .5
                dense_fine = runner.run(fine_scene, .5)
                save(args.output/(case+'-dense-refined.json'), dense_fine)
            else:
                dense = json.loads(dense_path.read_text()) if dense_path.exists() else run
                fine_path = args.output/(case+'-dense-refined.json')
                dense_fine = json.loads(fine_path.read_text()) if fine_path.exists() else None
            legacy_dense_path = args.output/(case+'-legacy-v9-dense.json')
            if args.dense_baseline:
                source, _ = fit.shell.inputs(case)
                source = fit.bake(source, fit.initial_parameters(), enabled=False)
                source['simulation']['duration_s'] = dense['time_s'][-1]
                source['failure_sample_times_s'] = dense['time_s']
                legacy_dense = runner.run(source); save(legacy_dense_path, legacy_dense)
            else:
                legacy_dense = json.loads(legacy_dense_path.read_text()) if legacy_dense_path.exists() else legacy
            for row in range(2):
                ax = paxes[row, col]; shown = dense if row == 0 else run
                ax.plot(at, report['observed_pressure_bar'], color='#343434', marker='o' if row == 0 else None,
                        markersize=3, label='Experiment (scan samples)')
                ax.plot(shown['time_s'], np.asarray(shown['gas']['chamber']['pressure_pa'])/1e5, color='#c43c39', label='Aperture v10')
                old_shown = legacy_dense if row == 0 else legacy
                ax.plot(old_shown['time_s'], np.asarray(old_shown['gas']['chamber']['pressure_pa'])/1e5, '--', color='#277da8', label='Previous v9')
                ax.set(xlim=(0, 10 if row==0 else 1800), xlabel='Time [s]', ylabel='Chamber pressure [bar abs]', title=case)
                ax.grid(alpha=.2)
            paxes[0, col].legend(fontsize=8)
            for row, (key, label) in enumerate([('101','C1 / ISS'), ('102','C1 / TIM'), ('107','C3 / TIM'), ('108','C3 / ISS')]):
                ax = taxes[row, col]; mask = at >= 300
                ax.plot(at[mask], np.asarray(report['observed_temperature_k'][key], dtype=float)[mask]-273.15, color='#343434', label='Experiment')
                ax.plot(at[mask], np.asarray(run['temperature_k'][key])[mask]-273.15, color='#c43c39', label='Aperture v10')
                ax.plot(at[mask], np.asarray(legacy['temperature_k'][key])[mask]-273.15, '--', color='#277da8', label='Previous v9')
                ax.set(title=case+' — '+label, xlabel='Time [s]', ylabel='Temperature [C]'); ax.grid(alpha=.2)
            taxes[0, col].legend(fontsize=8)
            for name, values in dense['channels'].items():
                for row, key in enumerate(('surface_energy_j', 'fracture_work_j')):
                    eaxes[row, col].plot(dense['time_s'], values[key], label=name.replace('C2-progressive-', ''))
            for row, label in enumerate(('Cumulative surface heat [J]', 'Cumulative opening work [J]')):
                eaxes[row, col].set(xlabel='Time [s]', ylabel=label, title=case); eaxes[row, col].grid(alpha=.2)
            eaxes[0, col].legend(ncol=3, fontsize=8)
            for key, label in [('inside-C2', 'Inside cell (unmeasured)'), ('chamber', 'Chamber')]:
                raxes[0, col].plot(dense['time_s'], np.asarray(dense['gas'][key]['pressure_pa'])/1e5, label=label)
                raxes[1, col].plot(dense['time_s'], dense['gas'][key]['temperature_k'], label=label)
            for row, (key, label, factor) in enumerate([('effective_area_m2', 'Opening area [mm2]', 1e6), ('mass_flow_kg_s', 'Flow [g/s]', 1e3)], start=2):
                values = sum(np.asarray(c[key]) for c in dense['channels'].values())*factor
                raxes[row, col].plot(dense['time_s'], values, label=label)
            for row, ylabel in enumerate(('Pressure [bar abs]', 'Gas temperature [K]', 'Opening area [mm2]', 'Mass flow [g/s]')):
                raxes[row, col].set(xlim=(0, 5), xlabel='Time [s]', ylabel=ylabel, title=case)
                raxes[row, col].grid(alpha=.2); raxes[row, col].legend(fontsize=8)
            _, prior = fit.shell.inputs(case)
            p = np.asarray(dense['gas']['chamber']['pressure_pa'])/1e5
            summary['experiments'][case] = dict(metrics=report['metrics'], previous_metrics=fit.shell.metrics(legacy, prior),
                energy_ledger=fit.ledger(run), dense_peak_bar=float(p.max()), dense_peak_time_s=dense['time_s'][int(p.argmax())],
                objective_blocks=objective_blocks(report),
                modeled_internal_peak_bar=float(np.max(dense['gas']['inside-C2']['pressure_pa'])/1e5),
                conservation={k: v for k, v in run.items() if 'residual' in k})
            if case == 'no-cooling':
                ablation_path = args.output/'frozen-parameters-without-aperture.json'
                if args.baseline_validation:
                    source, _ = fit.shell.inputs(case)
                    ablation = runner.run(fit.bake(source, profile['parameters'], enabled=False))
                    save(ablation_path, ablation)
                elif ablation_path.exists():
                    ablation = json.loads(ablation_path.read_text())
                else:
                    ablation = None
                if ablation is not None:
                    summary['experiments'][case]['frozen_parameter_ablation'] = dict(
                        metrics=fit.shell.metrics(ablation, prior),
                        changes=fit.changes(run, ablation),
                        note='Same frozen v10 coefficients, both new features disabled; diagnostic only, no refit')
            if dense_fine is not None:
                fine_p = np.asarray(dense_fine['gas']['chamber']['pressure_pa'])/1e5
                summary['experiments'][case]['dense_refinement'] = dict(
                    peak_bar=float(fine_p.max()), peak_time_s=dense_fine['time_s'][int(fine_p.argmax())],
                    max_pressure_change_bar=float(np.max(np.abs(fine_p-p))))
            print(case, json.dumps(summary['experiments'][case]), flush=True)
        for fig, filename, title in [(pressure, 'pressure-comparison.png', 'Chamber pressure: calibration / held-out replay'),
                                     (thermal, 'witness-comparison.png', 'Late witness temperatures, 300–1800 s; intact materials frozen'),
                                     (energy, 'energy-routing.png', 'Directed surface heat and irreversible opening work'),
                                     (routing, 'opening-transient.png', 'Modeled opening transient; internal pressure / flow are not measured')]:
            fig.suptitle(title); fig.savefig(args.output/filename, dpi=160); plt.close(fig)
        refinement = args.output/'refinement.json'
        if refinement.exists():
            summary['refinement'] = json.loads(refinement.read_text())
        save(args.output/'assessment.json', summary)
    finally:
        runner.close()


if __name__ == '__main__':
    main()
