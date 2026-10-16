#!/usr/bin/env python3
"""Report a frozen fit and held-out replay; no optimization or parameter choice."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import calibrate_upv_geometric_rupture as fit
from calibrate_upv import save

LABELS = {'101':'C1 / ISS', '102':'C1 / TIM', '107':'C3 / TIM', '108':'C3 / ISS'}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=fit.OUT)
    parser.add_argument('--probe', type=Path, default=fit.previous.PROBE)
    parser.add_argument('--dense', action='store_true')
    args = parser.parse_args(); out = args.output
    profile = json.loads((out/'profile.json').read_text())
    if profile['fingerprint'] != fit.fingerprint(args.probe):
        raise ValueError('Frozen model changed')
    summary = dict(profile=profile, experiments={}, heat_replay=json.loads((out/'heat-replay.json').read_text()),
                   refinement=json.loads((out/'refinement.json').read_text()),
                   calibration_accepted=False,
                   limitations=['Conditional primary failure, not prediction of ignition or cascade',
                                'Held-out experiment was inspected before this fit; not blind validation',
                                'Internal pressure, opening area and fracture work are unmeasured model predictions',
                                'Burst thermocouple temperatures excluded from the fit; no sensor offsets or fitted time shifts',
                                'Fixed work/seam priors are not identified material properties'])
    thermal, taxes = plt.subplots(4, 2, figsize=(12, 11), constrained_layout=True)
    pressure, paxes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    rupture, raxes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    runner = fit.native.Runner(args.probe, out)
    try:
        for col, (mode, name) in enumerate([('fit','no-cooling'), ('validate','active-cooling')]):
            report = json.loads((out/(mode+'.json')).read_text()); run = report['result']
            at = np.asarray(run['time_s']); dense_path = out/(name+'-dense.json')
            if args.dense:
                scene, _ = fit.inputs(name); scene = fit.bake(scene, profile['parameters'])
                scene['simulation']['duration_s'] = 30.
                scene['failure_sample_times_s'] = np.unique(np.r_[np.arange(0,3,.01), np.arange(3,30,.1),
                                                                  at[at<30], 30.]).tolist()
                dense = runner.run(scene); fit.audit(dense, scene); save(dense_path, dense)
            else:
                dense = json.loads(dense_path.read_text()) if dense_path.exists() else run
            for row, (key, label) in enumerate(LABELS.items()):
                ax = taxes[row,col]; mask = at>=300
                observed = np.asarray(report['observed_temperature_k'][key], dtype=float)
                ax.plot(at[mask], observed[mask]-273.15, color='#252b36', label='Experiment')
                ax.plot(at[mask], np.asarray(run['temperature_k'][key])[mask]-273.15, color='#d6493e', label='Geometric rupture v11')
                ax.set(title=name+' — '+label, xlabel='Time after event [s]', ylabel='Temperature [°C]'); ax.grid(alpha=.2)
            taxes[0,col].legend(fontsize=8)
            for row in range(2):
                ax = paxes[row,col]; shown = dense if row==0 else run
                ax.plot(at, report['observed_pressure_bar'], color='#252b36', marker='o' if row==0 else None,
                        markersize=3, label='Experiment (original scans)')
                ax.plot(shown['time_s'], np.asarray(shown['gas']['chamber']['pressure_pa'])/1e5,
                        color='#d6493e', label='Geometric rupture v11')
                ax.set(xlim=(0,5 if row==0 else 1800), title=name, xlabel='Time [s]', ylabel='Chamber pressure [bar abs]')
                ax.grid(alpha=.2)
            paxes[0,col].legend(fontsize=8)
            for gas, label in [('inside-C2','Internal, unmeasured'), ('chamber','Chamber')]:
                raxes[0,col].plot(dense['time_s'], np.asarray(dense['gas'][gas]['pressure_pa'])/1e5, label=label)
            area = sum(np.asarray(c['effective_area_m2']) for c in dense['channels'].values())*1e6
            raxes[1,col].plot(dense['time_s'], area, color='#277da8')
            for key, values in dense['channels'].items():
                raxes[2,col].plot(dense['time_s'], np.asarray(values['surface_energy_j'])/1000,
                                 label=key.replace('C2-progressive-',''))
            for row, label in enumerate(('Pressure [bar abs]', 'Total opening [mm²]', 'Cumulative surface heat [kJ]')):
                raxes[row,col].set(xlim=(0,5), title=name, xlabel='Time [s]', ylabel=label); raxes[row,col].grid(alpha=.2)
            raxes[0,col].legend(fontsize=8); raxes[2,col].legend(ncol=3, fontsize=8)
            dp = np.asarray(dense['gas']['chamber']['pressure_pa'])/1e5
            summary['experiments'][name] = dict(metrics=report['metrics'], energy_ledger=report['energy_ledger'],
                dense_peak_bar=float(dp.max()), dense_peak_time_s=float(dense['time_s'][int(dp.argmax())]),
                modeled_internal_peak_bar=float(max(dense['gas']['inside-C2']['pressure_pa'])/1e5),
                final_open_area_mm2=float(area[-1]),
                conservation={k:v for k,v in run.items() if 'residual' in k}, steps=run['steps'], wall_s=run['wall_s'])
            print(name, json.dumps(summary['experiments'][name]), flush=True)
        for fig, path, title in [(thermal,'witness-comparison.png','Late witness temperatures: calibration / held-out replay'),
                                 (pressure,'pressure-comparison.png','Chamber pressure: calibration / held-out replay'),
                                 (rupture,'opening-transient.png','Geometric opening: modeled pressure, area and routed heat')]:
            fig.suptitle(title); fig.savefig(out/path, dpi=160); plt.close(fig)
        summary['parameters_near_bounds'] = {k:v for k,v in profile['parameters'].items()
            if min(np.log(v/fit.LIMITS[k][0]), np.log(fit.LIMITS[k][1]/v))<.02}
        history = json.loads((out/'history.json').read_text())
        fig, ax = plt.subplots(figsize=(9,4), constrained_layout=True)
        ax.plot([x['candidate'] for x in history], [x['metrics']['witness']['rmse_k'] for x in history], '.-', color='#277da8')
        ax.set(xlabel='Full native candidate evaluation', ylabel='Training witness RMSE [K]', title='Training-only calibration; includes sensitivity probes')
        ax.grid(alpha=.2); fig.savefig(out/'search-history.png', dpi=160); plt.close(fig)
        save(out/'assessment.json', summary)
    finally:
        runner.close()


if __name__ == '__main__':
    main()
