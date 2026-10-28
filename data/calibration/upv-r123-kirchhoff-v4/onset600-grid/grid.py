#!/usr/bin/env python3
"""Coarse contact grid on the 600 s freeze. Held-out cooling is not used."""
import json, subprocess, sys, time
from itertools import product
from pathlib import Path
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[4] / 'scripts')]
import upv_r123_data as dataio
import upv_r123_model as model
import calibrate_upv_r123 as cal

OUT = Path(__file__).resolve().parent
SEED = json.loads((OUT.parent / 'onset600-contacts' / 'heat-parameters.json').read_text())['parameters']
DURATION = 626.956
TIMEOUT = 100.
FACES = ('C2_ISS', 'C2_TIM', '101', '102', '107', '108')
GRID = dict(tim_h=(300., 529., 2500.),
            case_h=(180., 477., 1400.),
            case_warm_response=(-4., 0., 4.))


def snapshot(run, meta):
    t = np.asarray(run['time_s']); obs = dataio.observed(meta['data'], t)
    gate = cal.r2_gate_s(meta['data'], meta['event']['time_s'])
    late = t >= gate; end = np.zeros(len(t), bool); end[-1] = True
    def stats(mask):
        rows = {}
        for name in FACES:
            err = np.asarray(run['temperature_k'][name]) - obs[name]
            sel = err[mask & np.isfinite(err)]
            rows[name] = dict(bias=float(np.mean(sel)), rmse=float(np.sqrt(np.mean(sel ** 2))))
        return rows
    iss = run['temperature_k']['C2_ISS'][-1]; tim = run['temperature_k']['C2_TIM'][-1]
    late_rmse = [v['rmse'] for v in stats(late).values()]
    return dict(late=stats(late), end=stats(end),
                late_face_rmse=float(np.sqrt(np.mean(np.square(late_rmse)))),
                iss_tim=float(iss - tim),
                obs_iss_tim=float(obs['C2_ISS'][-1] - obs['C2_TIM'][-1]),
                end_skew=float(max(v['bias'] for v in stats(end).values()) - min(v['bias'] for v in stats(end).values())))


def reject(partial):
    return cal.observed_runaway_rejection(partial, META, protocol='contacts')


history = json.loads((OUT / 'history.json').read_text()) if (OUT / 'history.json').exists() else []
done = {row['label'] for row in history}
META = None
best = None
started = time.monotonic()
for tim, case, warm in product(GRID['tim_h'], GRID['case_h'], GRID['case_warm_response']):
    label = f'tim{tim:g}_case{case:g}_warm{warm:g}'
    if label in done:
        print('SKIP', label, flush=True); continue
    p = dict(SEED, tim_h=tim, case_h=case, case_warm_response=warm)
    scene, META = model.build('no-cooling', p, warmup=True, duration=DURATION)
    print('START', label, flush=True)
    row = dict(label=label, parameters={k: p[k] for k in ('tim_h', 'case_h', 'case_warm_response',
               'iss_h', 'surface_iss', 'surface_tim', 'bulk_slope')})
    try:
        run = cal.replay(scene, OUT, label, TIMEOUT, reject_partial=reject)
        r = cal.residual(run, META, 'heat', False, 'contacts')
        snap = snapshot(run, META)
        row.update(status='complete', objective=float(r @ r), diagnosis=cal.diagnose_heat(run, META), **snap)
        print('DONE', label, 'obj', round(row['objective'], 3), 'late_rmse', round(snap['late_face_rmse'], 2),
              'end_skew', round(snap['end_skew'], 2), 'ISS-TIM', round(snap['iss_tim'], 2), flush=True)
        if best is None or snap['late_face_rmse'] < best[0]:
            best = (snap['late_face_rmse'], p, run, scene, row)
            cal.write_best(OUT, 'heat', scene, run, p, META, row['objective'],
                           dict(protocol='contacts', grid='tim_h,case_h,case_warm_response',
                                selected_on='late_face_rmse', validation_used_for_selection=False))
    except (subprocess.TimeoutExpired, ValueError, RuntimeError) as error:
        row.update(status='pruned' if str(error).startswith('Pruned:') else 'failed', error=str(error)[-400:])
        print(row['status'].upper(), label, row['error'][:160], flush=True)
    history.append(row); cal.save(OUT / 'history.json', history)
    cal.save(OUT / 'status.json', dict(evaluations=len(history), elapsed_s=time.monotonic() - started,
                                       last=label, best_late_rmse=None if best is None else best[0]))

if best:
    print('BEST late_rmse', round(best[0], 3), {k: best[1][k] for k in ('tim_h', 'case_h', 'case_warm_response')}, flush=True)
