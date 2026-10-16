#!/usr/bin/env python3
"""Read-only comparison of saved CUDA temperatures and original thermocouples.

No refit, smoothing, time-shift fitting, peak clipping or synthetic source
warmup prediction. Source face averages are explicitly labeled diagnostics.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from calibrate_upv_failure import load_record, CHANNELS

LABELS = {'101': 'C1 / ISS · ТП 101', '102': 'C1 / TIM · ТП 102',
          '107': 'C3 / TIM · ТП 107', '108': 'C3 / ISS · ТП 108'}
MODEL_COLOR = '#d55e00'
DATA_COLOR = '#172d40'


def read(path):
    return json.loads(path.read_text())


def load_series(root):
    specs = [('no-cooling', 'Калибровка · без водянки', root/'fit.json', root/'warmup.json'),
             ('active-cooling', 'Валидация · G = 10 Вт/К',
              root/'cooling-check/nominal/report.json', root/'cooling-check/nominal/warmup.json')]
    result = []
    for name, title, report_path, warm_path in specs:
        report, warm = read(report_path), read(warm_path)
        data, record = load_record(name)
        event = report['meta']['event']; origin = event['last_pre_event_s']
        if record['canonical_sha256'] != report['meta']['source_csv_sha256']:
            raise ValueError('Report and raw observations have different origins')
        if abs(warm['time_s'][-1]-origin) > 1e-7:
            raise ValueError('Warmup/event time gap')
        post = report['result']
        time = np.r_[np.asarray(warm['time_s'])[:-1]-origin, post['time_s']]
        if not np.all(np.diff(time)>0):
            raise ValueError('Nonmonotonic joined timeline')
        predicted = {k: np.r_[warm['temperature_k'][k][:-1], post['temperature_k'][k]]-273.15
                     for k in LABELS}
        # Keep acquisition gaps as gaps, not lines bridging missing scans.
        observed_time = data['time_s']-origin
        gap = np.r_[False, np.diff(observed_time)>2*np.median(np.diff(observed_time))]
        observed = {}
        for key in CHANNELS:
            values = np.mean([data['T'+c+'_k'] for c in CHANNELS[key]], axis=0)-273.15
            values[gap] = np.nan
            observed[key] = values
        result.append(dict(title=title, report=report, post=post, time=time,
                           predicted=predicted, observed_time=observed_time, observed=observed,
                           raw=data, gap=gap, origin=origin, event=event,
                           input_paths=[report_path, warm_path], csv_sha256=record['canonical_sha256']))
    return result


def decorate(ax, title, window, event, small=False):
    ax.set(title=title, xlim=window, xlabel='Время относительно начала расчёта разрушения, с',
           ylabel='Температура, °C')
    ax.grid(alpha=.2)
    if window[0] <= 0 <= window[1]:
        ax.axvspan(0, event['bracket_width_s'], color='#777777', alpha=.18)
        ax.axvline(0, color='#777777', ls=':', lw=.9)
    ax.tick_params(labelsize=9)
    if small:
        ax.xaxis.label.set_size(9)


def limits(axes, arrays):
    finite = [a[np.isfinite(a)] for a in arrays if len(a)]
    finite = [a for a in finite if len(a)]
    low, high = min(a.min() for a in finite), max(a.max() for a in finite)
    pad = max(2., (high-low)*.06)
    for ax in axes:
        ax.set_ylim(low-pad, high+pad)


def witness_plot(series, folder, window, filename, title, markers=False):
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), constrained_layout=True)
    for row, (key, label) in enumerate(LABELS.items()):
        values = []
        for col, s in enumerate(series):
            ax = axes[row, col]
            t, raw_t = s['time'], s['observed_time']
            pick = (t>=window[0]) & (t<=window[1])
            raw_pick = (raw_t>=window[0]) & (raw_t<=window[1])
            a, b = s['observed'][key][raw_pick], s['predicted'][key][pick]
            ax.plot(raw_t[raw_pick], a, color=DATA_COLOR, lw=1.5, label='Эксперимент',
                    marker='.' if markers else None, markersize=3)
            ax.plot(t[pick], b, color=MODEL_COLOR, lw=1.6, ls='--', label='Расчёт',
                    marker='.' if markers else None, markersize=3)
            decorate(ax, s['title']+'\n'+label, window, s['event'])
            values.extend([a, b])
        limits(axes[row], values)  # same y scale for each corresponding pair
    for ax in axes[0]:
        ax.legend(fontsize=9)
    fig.suptitle(title+'\nБез сглаживания; одинаковая шкала температур слева и справа', fontsize=14)
    fig.savefig(folder/filename, dpi=160)
    plt.close(fig)


def source_plot(series, folder):
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), constrained_layout=True)
    for row, (key, window) in enumerate([('C2-ISS', (0,60)), ('C2-TIM', (0,60)),
                                        ('C2-ISS', (60,1800)), ('C2-TIM', (60,1800))]):
        values = []
        for col, s in enumerate(series):
            ax = axes[row, col]; t = np.asarray(s['post']['time_s'])
            raw_t = s['observed_time']; raw_pick = (raw_t>=window[0]) & (raw_t<=window[1])
            pick = (t>=window[0]) & (t<=window[1])
            for channel in CHANNELS[key]:
                raw = s['raw']['T'+channel+'_k'].copy()-273.15
                raw[s['gap']] = np.nan
                ax.plot(raw_t[raw_pick], raw[raw_pick], lw=.9, alpha=.55, label='ТП '+channel)
                values.append(raw[raw_pick])
            mean = s['observed'][key][raw_pick]
            predicted = np.asarray(s['post']['temperature_k'][key])[pick]-273.15
            ax.plot(raw_t[raw_pick], mean, color=DATA_COLOR, lw=1.5, label='Среднее двух ТП')
            ax.plot(t[pick], predicted, color=MODEL_COLOR, lw=1.6, ls='--', label='Расчёт поверхности')
            decorate(ax, s['title']+'\n'+key.replace('-', ' / '), window, s['event'])
            values.extend([mean, predicted])
        limits(axes[row], values)
    for ax in axes[:2].flat:
        ax.legend(fontsize=8)
    fig.suptitle('Батарея-источник C2: переходный процесс и охлаждение\n'
                 'Показания при выбросе — диагностика; они не входили в подбор параметров', fontsize=14)
    fig.savefig(folder/'source-event-and-after.png', dpi=160); plt.close(fig)


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('root', type=Path)
    cli.add_argument('--output', type=Path)
    args = cli.parse_args(); out = args.output or args.root/'temperature-process'
    if out.exists():
        raise ValueError('Use a new plot directory; do not overwrite reports')
    series = load_series(args.root); out.mkdir(parents=True)
    witness_plot(series, out, (-10,60), 'witness-event.png', 'Соседние батареи · начало разрушения', True)
    witness_plot(series, out, (60,1800), 'witness-after.png', 'Соседние батареи · после первых 60 секунд')
    witness_plot(series, out, (min(s['time'][0] for s in series),1800),
                 'witness-full.png', 'Соседние батареи · весь прогрев, разрушение и охлаждение')
    source_plot(series, out)
    manifest = dict(protocol='Saved runs, original scans, no smoothing/time shift or refitting',
        time_zero='Last acquisition before measured pressure event; shaded band spans to first pressure-event scan',
        cooling='Previously declared nominal G=10 W/K; not selected on validation errors',
        source='Predicted area-averaged casing face vs two individual thermocouples and their mean; no invented pre-event C2 prediction',
        event_note='Thermocouple attachment/gas exposure during rupture is uncertain; readings not silently discarded',
        axes='Same y limits for corresponding training/validation panels, linear scales; early and late windows explicitly separate',
        inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for s in series for p in s['input_paths']},
        csv_hashes=[s['csv_sha256'] for s in series])
    (out/'provenance.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print(out)


if __name__ == '__main__':
    main()
