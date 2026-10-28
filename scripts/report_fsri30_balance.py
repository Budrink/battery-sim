#!/usr/bin/env python3
"""Read-only, snapshot-based plots; observed prefixes are never extrapolated."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from fsri30_data import ROOT, load, save, sha
from fsri30_objective import model_fields, targets
from fsri30_burst_fit import reference, assess


def read_snapshot(folder, search=None):
    for name in ('run.json.gz', 'incomplete-run.json.gz'):
        path = folder/'replay'/name
        if path.exists():
            raw = gzip.decompress(path.read_bytes())
            return json.loads(raw), dict(path=str(path), uncompressed_sha256=hashlib.sha256(raw).hexdigest())
    if search is None:
        raise ValueError('No saved trajectory: '+str(folder))
    state = json.loads((search/'status.json').read_text())
    standalone='active_trial'not in state
    if standalone:
        if search.resolve()!=folder.resolve()or state.get('status')!='running':
            raise ValueError('Standalone snapshot requires its running controller')
        # Associate the temporary observation with THIS native child. Merely
        # finding one temp file is insufficient when another experiment exists.
        pid=state['pid'];children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        paths=[]
        for child in children:
            try:
                argv=Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0')
                if len(argv)>1 and Path(argv[1].decode()).resolve()==(folder/'scenario.json').resolve():
                    paths.append(Path(f'/proc/{child}/fd/1').resolve().parent/'partial.json')
            except FileNotFoundError:continue
        if len(paths)!=1:raise ValueError('Native child of standalone replay is ambiguous')
    elif Path(state['active_trial']).resolve() != folder.resolve():
        raise ValueError('Requested trial is not the live trial; retry its saved archive')
    else:paths = list(Path('/tmp').glob('battery-gas-restart-*/partial.json'))
    if len(paths) != 1:
        raise ValueError('Live checkpoint is ambiguous')
    raw = paths[0].read_bytes()
    after = json.loads((search/'status.json').read_text())
    if (standalone and (after.get('pid')!=state['pid']or after.get('status')!='running'))or (
        not standalone and after['active_trial'] != state['active_trial']):
        raise ValueError('Live trial changed during capture; retry the saved archive')
    return json.loads(raw), dict(path=str(paths[0]), uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                               captured_live=True, trial=str(folder.resolve()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--current',type=Path,default=ROOT/'data/calibration/fsri30-balance-bracket-20261026-r2/trial-0026')
    ap.add_argument('--current-label',default='26')
    ap.add_argument('--previous',type=Path,default=ROOT/'data/calibration/fsri30-balance-bracket-20261026-r1/trial-0025')
    ap.add_argument('--previous-label',default='25')
    ap.add_argument('--search',type=Path,default=ROOT/'data/calibration/fsri30-balance-bracket-20261026-r2')
    ap.add_argument('--reference-horizon',action='store_true',
        help='Keep completed reference curves visible beyond the live prefix; never extend the prefix')
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    search = args.search
    folders = (
        (args.current_label, args.current.resolve(), '#1478b8', '-'),
        (args.previous_label, args.previous.resolve(), '#dc7625', '--'),
        ('23', ROOT/'data/calibration/fsri30-explosion-priority-20261026-r1/trial-0023', '#46875c', ':'),
    )
    unique=[];seen=set()
    for entry in folders:
        path=entry[1].resolve()
        if path not in seen:unique.append(entry);seen.add(path)
    folders=unique
    data, manifest = load()
    truth = targets(data)
    observed = np.array([data[f'T{i}_k'] for i in range(1, 10)]).T
    arrays = dict(experiment_time_s=truth['time'], experiment_temperatures_k=observed)
    rows = []
    for number, folder, color, style in folders:
        run, source = read_snapshot(folder, search if folder == args.current.resolve() else None)
        t, ids, v = model_fields(run)
        if abs(t[0]) > 1e-7:
            raise ValueError('Only fresh-from-zero observations belong in this report')
        grid = truth['time'][(truth['time'] >= t[0]) & (truth['time'] <= t[-1]+1e-7)]
        q = np.quantile(v, [0., .5, .9, 1.], axis=1).T
        interpolated = np.stack([np.interp(grid, t, q[:, j]) for j in range(4)], axis=1)
        smooth = median_filter(interpolated, size=(11, 1), mode='nearest')
        status = json.loads((folder/'status.json').read_text())
        burst=assess(json.loads((folder/'scenario.json').read_text()),run,data)
        arrays[f'trial{number}_time_s'] = t
        arrays[f'trial{number}_temperatures_k'] = v
        source.update(trial=number, scenario_sha256=sha(folder/'scenario.json'),
                      native_sha256=status['native_sha256'], observed_end_s=float(t[-1]),
                      full_trajectory=bool(not run.get('partial') and t[-1] >= 5945.-1e-7),
                      burst=burst, component_ids=ids)
        rows.append(dict(number=number, grid=grid, q=smooth, color=color, style=style,
                         t=t, raw_q=q, source=source))
        del run, v
    current_end=rows[0]['source']['observed_end_s']
    end=max(r['source']['observed_end_s']for r in rows)if args.reference_horizon else current_end
    common_end = min(r['source']['observed_end_s'] for r in rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    mask = truth['time'] <= end+1e-7
    et = truth['time'][mask]
    eq = truth['q'][mask]-273.15
    for ax in axes:
        ax.fill_between(et, eq[:, 0], eq[:, 3], color='#777777', alpha=.16,
                        label='Эксперимент: диапазон 9 термопар')
        ax.plot(et, eq[:, 1], color='#171717', lw=2.1, label='Эксперимент: медиана')
        current = rows[0]
        ax.fill_between(current['grid'], current['q'][:, 0]-273.15,
                        current['q'][:, 3]-273.15, color=current['color'], alpha=.10,
                        label='№'+args.current_label+': диапазон 30 батарей')
        for r in reversed(rows):
            m = r['grid'] <= end+1e-7
            ax.plot(r['grid'][m], r['q'][m, 1]-273.15, color=r['color'],
                    ls=r['style'], lw=2.0 if r['number'] == args.current_label else 1.5,
                    label='№'+r['number']+': медиана')
        ax.set(xlabel='Время, с', ylabel='Температура, °C')
        ax.grid(alpha=.2)
    axes[0].set(title='Весь процесс — без сдвига времени', xlim=(0., end))
    axes[0].legend(fontsize=8, loc='upper left')
    axes[1].set(title='Каскад — увеличенный участок', xlim=(2470., 2700.))
    experimental_peak=reference(data)['peak_time_s']
    gates=[('Эксп.',experimental_peak,'#171717')]
    peak_text=[f'Пик медианы: эксперимент {experimental_peak:.0f} с']
    for r in rows[:2]:
        burst=r['source']['burst']
        if not burst.get('evaluated'):continue
        when=burst['measured']['peak_time_s'];gates.append(('№'+r['number'],when,r['color']))
        peak_text.append(f"№{r['number']}: {when:.0f} с ({when-experimental_peak:+.0f} с)")
    for label, when, color in gates:
        axes[1].axvline(when, color=color, alpha=.5, lw=.9)
    axes[1].text(.02, .97, '\n'.join(peak_text),
                 transform=axes[1].transAxes, va='top', fontsize=9,
                 bbox=dict(facecolor='white', edgecolor='.85', alpha=.92))
    fig.suptitle(f'FSRI, 30 батарей · снимок №{args.current_label} до {current_end:.0f} с\n'
                 '9 термопар ≠ 30 средних температур батарей; привязка датчиков не установлена', fontsize=12)
    fig.savefig(out/'cascade.png', dpi=160)
    fig.savefig(out/'cascade.svg')
    fig.savefig(out/'cascade.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.3), sharex=True, constrained_layout=True)
    labels = ('Минимальная температура', 'Медианная температура',
              '90-й процентиль', 'Максимальная температура')
    for j, ax in enumerate(axes.flat):
        ax.plot(et, eq[:, j], color='#171717', lw=2., label='Эксперимент: 9 термопар')
        for r in reversed(rows):
            if r['source']['observed_end_s']<3600.:continue
            m = r['grid'] <= end+1e-7
            ax.plot(r['grid'][m], r['q'][m, j]-273.15, color=r['color'],
                    ls=r['style'], lw=2.0 if r['number'] == args.current_label else 1.6,
                    label='№'+r['number']+': 30 батарей')
        ax.set(title=labels[j], ylabel='Температура, °C', xlabel='Время, с',
               xlim=(3600., end))
        relevant = [eq[(et >= 3600.), j]]
        relevant.extend(r['q'][(r['grid'] >= 3600.) & (r['grid'] <= end+1e-7), j]-273.15 for r in rows)
        joined = np.concatenate(relevant)
        if len(joined):
            margin = max(15., .08*float(np.ptp(joined)))
            ax.set_ylim(float(joined.min())-margin, float(joined.max())+margin)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    prior_end=rows[1]['source']['observed_end_s']
    fig.suptitle(f'Охлаждение · распределение температур, не совпадающие датчики\n'
                 f'№{args.current_label} рассчитан до {current_end:.0f} с; №{args.previous_label} — до {prior_end:.0f} с. '
                 'Сглаживание 11 с; продолжения не дорисованы', fontsize=12)
    fig.savefig(out/'cooling.png', dpi=160)
    fig.savefig(out/'cooling.svg')
    fig.savefig(out/'cooling.pdf')
    plt.close(fig)
    np.savez_compressed(out/'plot-source.npz', **arrays)
    save(out/'metadata.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        report_sha256=sha(Path(__file__)), csv_sha256=manifest['csv_sha256'],
        plotted_until_s=end, common_observed_until_s=common_end,
        time_shift_s=0., extrapolation=False, smoothing_seconds=11,
        scope='9 TC quantiles versus 30 component means; NOT matched sensors or validation',
        trials=[r['source'] for r in rows],
        source_arrays_sha256=sha(out/'plot-source.npz')))
    print(json.dumps(dict(output=str(out), plotted_until_s=end, common_observed_until_s=common_end,
                          trials=[{k:r['source'][k] for k in ('trial','observed_end_s','full_trajectory')} for r in rows]), indent=2))


if __name__ == '__main__':
    main()
