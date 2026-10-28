#!/usr/bin/env python3
"""Plot the exact viewer recording; no solver, fitting, warping or smoothing."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

from fsri30_data import load


MODEL, EXPERIMENT = '#176fa6', '#d65a28'
ZOOM, FRONT = (1600., 2700.), (2500., 2630.)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def recording(path):
    manifest = json.loads(path.read_text())
    if manifest['format'] != 'BATTVIEW2' or manifest.get('partial', True):
        raise ValueError('A completed BATTVIEW2 capture is required')
    scene = json.loads(Path(manifest['scenario_path']).read_text())
    ids = sorted(c['id'] for c in scene['cells'] if c.get('role') == 'battery')
    times = np.asarray(manifest['time_s'])
    stride = (len(manifest['objects']) * 4 + len(manifest['channels']) *
              manifest['channel_fields'] + len(manifest['gas']) * manifest['gas_fields'])
    binary = path.parent / 'frames.bin'
    if binary.stat().st_size != times.size * stride * 4:
        raise ValueError('Incomplete binary payload')
    frames = np.memmap(binary, dtype='<f4', mode='r', shape=(len(times), stride))
    temperature = np.array(frames[:, [4 * manifest['objects'].index(i) for i in ids]], dtype=float)
    if not np.isfinite(temperature).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid recorded samples')
    if times[0] != 0 or abs(times[-1] - 3000) > 1e-6:
        raise ValueError('This report requires the same 0–3000 s viewer capture')
    # Resolve shell events via the scenario's ports, not event ordering.
    channel_owner = {c['id']: c.get('port_id', '').split(':')[0]
                     for c in scene['gas_network']['channels']}
    events = {i: {} for i in ids}
    for event in manifest['events']:
        kind = event['kind']
        if kind == 'shell_rupture':
            owner, key = channel_owner.get(event['id']), 'first_open_s'
        elif kind in ('runaway_primary', 'runaway_secondary', 'runaway_complete'):
            owner = event['id']
            key = 'spent_s' if kind == 'runaway_complete' else 'runaway_flag_s'
        else:
            continue
        if owner in events:
            events[owner][key] = min(events[owner].get(key, np.inf), event['time_s'])
    if any(set(row) != {'first_open_s', 'runaway_flag_s', 'spent_s'} for row in events.values()):
        raise ValueError('Missing battery event in completed cascade')
    return manifest, ids, times, temperature, events


def temperatures(ax, time, model, etime, observed, window, legend=False):
    mq = np.quantile(model, [0, .5, 1], axis=1).T - 273.15
    eq = np.quantile(observed, [0, .5, 1], axis=1).T - 273.15
    ax.fill_between(etime, eq[:, 0], eq[:, 2], color=EXPERIMENT, alpha=.15,
                    label='Эксперимент: диапазон 9 термопар')
    ax.fill_between(time, mq[:, 0], mq[:, 2], color=MODEL, alpha=.12,
                    label='Модель: диапазон 30 средних температур')
    ax.plot(etime, eq[:, 1], color=EXPERIMENT, lw=2.2, label='Эксперимент: медиана')
    ax.plot(time, mq[:, 1], color=MODEL, lw=2.2, label='Модель: медиана')
    ax.set(xlim=window, ylim=(0, 1250), xlabel='Время от начала опыта, с', ylabel='Температура, °C')
    ax.grid(alpha=.18)
    if legend:
        ax.legend(loc='upper left', fontsize=9, framealpha=.95)


def write_csv(path, columns, rows):
    with path.open('w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace only this report, requiring matching source hashes')
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest, ids, time, model, events = recording(manifest_path)
    experimental, experiment_manifest = load()
    mask = experimental['time_s'] <= 3000
    etime = experimental['time_s'][mask]
    observed = np.stack([experimental[f'T{i}_k'][mask] for i in range(1, 10)], axis=1)
    if not np.isfinite(observed).all():
        raise ValueError('Missing experiment values; explicit missing-data handling required')
    out = args.output.resolve()
    if out.exists():
        if not args.overwrite:
            raise FileExistsError('Report exists; use --overwrite explicitly')
        previous = json.loads((out / 'metadata.json').read_text())
        if (previous['recording_sha256'] != digest(manifest_path) or
                previous['experiment_csv_sha256'] != experiment_manifest['csv_sha256']):
            raise ValueError('Refusing to replace a report from different sources')
    out.mkdir(parents=True, exist_ok=args.overwrite)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    peaks = {}
    for label, t, values in [('model', time, model), ('experiment', etime, observed)]:
        median = np.median(values, axis=1)
        k = int(np.argmax(median))
        peaks[label] = dict(time_s=float(t[k]), temperature_k=float(median[k]))
    def first_500(t, temperature_k):
        crossed = np.flatnonzero(temperature_k >= 773.15)
        if not len(crossed):
            raise ValueError('No 500 °C crossing: cascade comparison must be revised')
        return float(t[crossed[0]])
    front_times = dict(model_median_500_s=first_500(time, np.median(model, axis=1)),
                      experiment_median_500_s=first_500(etime, np.median(observed, axis=1)),
                      tc_first_500_s={f'T{i+1}': first_500(etime, observed[:, i]) for i in range(9)})
    early = [front_times['tc_first_500_s'][f'T{i}'] for i in range(1, 5)]
    early_window = (min(early), max(early))
    csv_runaway = float(experimental['time_s'][np.flatnonzero(experimental['runaway'])[0]])
    footnote = ('Без сглаживания и сдвига времени. 9 термопар ≠ 30 средних температур батарей: '
                'точная привязка датчиков неизвестна.\n'
                'Это та же траектория, что в визуализаторе, а не новая калибровка или независимая валидация.')
    products = []
    with PdfPages(out / 'fsri30-3000s.pdf') as pdf:
        def save(fig, name):
            fig.savefig(out / f'{name}.png', dpi=170)
            fig.savefig(out / f'{name}.svg')
            pdf.savefig(fig)
            products.append(name)
            plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(17, 6.8))
        fig.subplots_adjust(left=.055, right=.98, top=.84, bottom=.19, wspace=.20)
        temperatures(axes[0], time, model, etime, observed, (0, 3000), legend=True)
        temperatures(axes[1], time, model, etime, observed, ZOOM)
        axes[0].axvspan(*ZOOM, color='#f5c764', alpha=.14, zorder=0)
        axes[0].set_title('Весь интервал · 0–3000 с', loc='left', fontweight='bold')
        axes[1].set_title('Ранние события и поздний фронт · 1600–2700 с', loc='left', fontweight='bold')
        axes[1].axvspan(*early_window, color=EXPERIMENT, alpha=.22)
        axes[0].axvline(csv_runaway, color='#6e6e6e', ls=':', lw=1.2)
        axes[0].text(csv_runaway - 25, 720, f'Флаг Thermal Runaway\nв CSV: {csv_runaway:.0f} с',
                     rotation=90, va='center', ha='right', fontsize=9, color='#565656')
        peak_text = []
        for key, label, color in [('model', 'Модель', MODEL), ('experiment', 'Эксперимент', EXPERIMENT)]:
            peak = peaks[key]
            axes[1].axvline(peak['time_s'], color=color, lw=1, ls='--', alpha=.6)
            peak_text.append(f"{label}: {peak['time_s']:.1f} с / {peak['temperature_k']-273.15:.0f} °C")
        axes[1].text(.97, .04, 'Пики несглаженных медиан\n' + '\n'.join(peak_text),
                     transform=axes[1].transAxes, ha='right', va='bottom', fontsize=10,
                     bbox=dict(facecolor='white', alpha=.94, edgecolor='#dddddd'))
        fig.suptitle('FSRI · 30 батарей · та же запись 3000 с\nТемпература: расчёт и эксперимент',
                     fontsize=17, fontweight='bold', y=.97)
        fig.text(.055, .04, footnote, fontsize=10, color='#555555', linespacing=1.6)
        save(fig, 'temperature-3000')

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.18, hspace=.16)
        for i in range(9):
            axes[0].plot(etime, observed[:, i]-273.15, lw=1.2, label=f'T{i+1}')
        axes[0].set_title('Эксперимент · все 9 термопар, без сглаживания', loc='left')
        axes[0].legend(ncol=9, fontsize=9, loc='upper left')
        for i in range(len(ids)):
            axes[1].plot(time, model[:, i]-273.15, color=MODEL, alpha=.25, lw=.8)
        axes[1].plot(time, np.median(model, axis=1)-273.15, color=MODEL, lw=2.3,
                     label='Медиана 30 батарей')
        axes[1].set_title('Модель · средние температуры всех 30 батарей', loc='left')
        axes[1].legend(loc='upper left', fontsize=9)
        for ax in axes:
            ax.set(xlim=ZOOM, ylim=(0, 1250), ylabel='Температура, °C')
            ax.axvspan(*early_window, color=EXPERIMENT, alpha=.23)
            ax.axvline(front_times['model_median_500_s'], color=MODEL, ls=':', lw=1.2)
            ax.grid(alpha=.18)
        axes[0].annotate(f'T1–T4 проходят 500 °C\nза {early_window[0]:.0f}–{early_window[1]:.0f} с',
                         xy=(np.mean(early_window), 700),
                         xytext=(1900, 930), fontsize=10,
                         arrowprops=dict(arrowstyle='->', color=EXPERIMENT))
        axes[1].text(2160, 700, 'Ранняя групповая волна\nне воспроизводится', color=MODEL)
        axes[1].set_xlabel('Время от начала опыта, с')
        fig.suptitle('Порядок событий не совпадает: ранняя волна и поздний максимум',
                     fontsize=16, fontweight='bold', y=.96)
        front_caption = (f"Медиана достигает 500 °C: эксперимент {front_times['experiment_median_500_s']:.0f} с, "
                         f"модель {front_times['model_median_500_s']:.0f} с "
                         f"(разница {front_times['model_median_500_s']-front_times['experiment_median_500_s']:.0f} с).\n")
        fig.text(.08, .045, front_caption +
                 'Это сравнение агрегатов, не задержка конкретной батареи: привязка термопар неизвестна.\n'
                 'Близкие поздние максимумы не подтверждают правильное воспроизведение всей последовательности.',
                 fontsize=10, color='#555555', linespacing=1.5)
        save(fig, 'early-cascade')

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                 gridspec_kw={'height_ratios': [2.4, 1]})
        fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.16, hspace=.16)
        temperatures(axes[0], time, model, etime, observed, FRONT, legend=True)
        axes[0].set_xlabel('')
        for key, color in [('model', MODEL), ('experiment', EXPERIMENT)]:
            axes[0].axvline(peaks[key]['time_s'], color=color, ls='--', lw=1, alpha=.6)
        for key, label, color in [('runaway_flag_s', 'Достигнут флаг runaway ядра', '#e09b22'),
                                  ('first_open_s', 'Есть вскрытие оболочки', '#008d92'),
                                  ('spent_s', 'Состояние spent', '#741e88')]:
            times = np.sort([e[key] for e in events.values()])
            axes[1].step(np.r_[0, times, 3000], np.r_[0, np.arange(1, len(times)+1), len(times)],
                         where='post', color=color, lw=2, label=label)
        axes[1].set(xlim=FRONT, ylim=(-.5, 31), ylabel='Батарей, шт.',
                    xlabel='Время от начала опыта, с', yticks=np.arange(0, 31, 5))
        axes[1].grid(alpha=.2)
        axes[1].legend(loc='upper left', ncol=1, fontsize=9, framealpha=.95)
        last = max(e['spent_s'] for e in events.values())
        axes[1].annotate(f'30 из 30: {last:.2f} с', xy=(last, 30), xytext=(last+10, 17),
                         color='#741e88', fontsize=10, arrowprops=dict(arrowstyle='->', color='#741e88'))
        fig.suptitle('Поздний коллективный фронт · 2500–2630 с (не первый каскад)',
                     fontsize=17, fontweight='bold', y=.96)
        fig.text(.08, .045, 'Снизу — только события модели. Флаг runaway — порог отчётности, а не начало непрерывной R2.\n'
                 'Их нельзя автоматически считать временем физического взрыва каждой батареи в эксперименте.',
                 fontsize=10, color='#555555', linespacing=1.6)
        save(fig, 'burst-front')

        order = sorted(ids, key=lambda name: events[name]['spent_s'])
        values = model[:, [ids.index(name) for name in order]].T - 273.15
        palette = LinearSegmentedColormap.from_list('viewer-blue-yellow-red', [
            (0., (.025, .12, .55)), (.22, (.025, .40, .90)), (.40, (.36, .72, .86)),
            (.60, (1., .90, .06)), (.80, (1., .34, .015)), (1., (.94, .015, .025))])
        # Sample support only: no resampling or smoothing of the thermal map.
        edges = np.r_[time[0], (time[1:]+time[:-1])/2, time[-1]]
        fig, axes = plt.subplots(1, 2, figsize=(17, 11), sharey=True)
        fig.subplots_adjust(left=.07, right=.91, top=.86, bottom=.14, wspace=.08)
        for ax, window, title in zip(axes, [(0, 3000), (2300, 2650)],
                                     ['Вся история каждой батареи', 'Каскад · увеличенный участок']):
            image = ax.pcolormesh(edges, np.arange(len(order)+1)-.5, values,
                                  cmap=palette, norm=Normalize(25, 1000), rasterized=True)
            ax.set(xlim=window, ylim=(len(order)-.5, -.5), xlabel='Время, с',
                    yticks=np.arange(len(order)), yticklabels=order)
            ax.set_title(title, loc='left', fontweight='bold')
            ax.tick_params(axis='y', labelsize=9)
            for y, name in enumerate(order):
                for key, marker, face, label in [
                    ('runaway_flag_s', 'o', 'none', 'Флаг runaway ядра'),
                    ('first_open_s', '^', '#63e4e0', 'Первое вскрытие'),
                    ('spent_s', 'x', 'white', 'Состояние spent')]:
                    x = events[name][key]
                    if not window[0] <= x <= window[1]:
                        continue
                    ax.plot(x, y, marker=marker, color='white', markerfacecolor=face,
                             markeredgecolor='white' if marker != '^' else '#102830',
                             markersize=5.8, markeredgewidth=1.1, ls='none')
            for y in np.arange(.5, len(order), 1):
                ax.axhline(y, color='white', alpha=.16, lw=.5)
        axes[0].set_ylabel('Батареи по времени перехода в spent')
        handles = [plt.Line2D([], [], marker=marker, color='#444444', markerfacecolor=face,
                             ls='none', label=label) for marker, face, label in [
                                 ('o', 'none', 'Флаг runaway ядра'), ('^', '#63e4e0', 'Первое вскрытие'),
                                 ('x', 'white', 'Состояние spent')]]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.47, .91), ncol=3, frameon=False)
        bar = fig.colorbar(image, cax=fig.add_axes([.93, .18, .012, .63]), extend='max')
        bar.set_label('Средняя температура батареи, °C')
        bar.set_ticks([25, 250, 500, 750, 1000])
        bar.set_ticklabels(['25', '250', '500', '750', '≥1000'])
        fig.suptitle('Распространение нагрева по всем 30 батареям', fontsize=17, fontweight='bold', y=.97)
        fig.text(.07, .045, 'Только модель; строки отсортированы по времени перехода в spent, а не по геометрическому положению.\n'
                 'Ранние локальные события сохранены на левом графике. Цветовая шкала совпадает с визуализатором.',
                 fontsize=10, color='#555555', linespacing=1.6)
        save(fig, 'cascade-map')

    write_csv(out / 'model-temperatures-c.csv', ['time_s']+[i+'_c' for i in ids],
              np.column_stack([time, model-273.15]))
    write_csv(out / 'experimental-temperatures-c.csv', ['time_s']+[f'T{i}_c' for i in range(1,10)],
              np.column_stack([etime, observed-273.15]))
    write_csv(out / 'cascade-events.csv', ['battery', 'runaway_flag_s', 'first_open_s', 'spent_s'],
              [[i, events[i]['runaway_flag_s'], events[i]['first_open_s'], events[i]['spent_s']] for i in order])
    np.savez_compressed(out / 'plot-source.npz', model_time_s=time, model_temperature_k=model,
                        battery_ids=np.asarray(ids), experiment_time_s=etime, experiment_temperature_k=observed)
    metadata = dict(recording_manifest=str(manifest_path), recording_sha256=digest(manifest_path),
                    frames_sha256=digest(manifest_path.parent/'frames.bin'),
                    scenario_sha256=digest(Path(manifest['scenario_path'])),
                    experiment_csv_sha256=experiment_manifest['csv_sha256'],
                    script_sha256=digest(Path(__file__)), horizon_s=[0,3000],
                    smoothing=False, time_shift_s=0, new_simulation=False, independent_validation=False,
                    comparison='9 TC versus 30 battery means; no individual sensor mapping',
                    raw_median_peaks=peaks, model_event_definition='native events; not measured explosion times',
                    first_500_crossings=front_times,
                    events=events, battery_display_order=order, products=products)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out), products=products, raw_median_peaks=peaks), indent=2))


if __name__ == '__main__':
    main()
