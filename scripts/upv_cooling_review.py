"""Observation-only phase report: cooling is assessed separately from jet peaks.

No fitted clock shift, temperature gain, missing-scan filling, or simulated
extrapolation. Uniform-time quadrature prevents dense burst scans dominating.
"""
import numpy as np

import upv_r123_data as dataio
from upv_observation_policy import CALIBRATION_CHANNELS, label, metadata as observation_policy


def resolved_samples(times, grid, maximum_gap_s=5.01):
    """Do not turn one late endpoint into an apparently measured trajectory."""
    indices = np.clip(np.searchsorted(times, grid, side='right')-1, 0, len(times)-2)
    exact = (np.abs(grid-times[indices]) < 1e-7) | (np.abs(grid-times[indices+1]) < 1e-7)
    return exact | ((times[indices+1]-times[indices]) <= maximum_gap_s)


def phase_report(run, data, window_end_s=None):
    times = np.asarray(run['time_s'], dtype=float)
    event = dataio.event(data)['time_s']
    data_start, data_end = float(data['time_s'][0]), float(data['time_s'][-1])
    initial = dataio.observed(data,np.asarray([data_start]))
    window_end = min(data_end, event+300. if window_end_s is None else float(window_end_s))
    phases = [('pre_burst', data_start, event-5.),
              ('burst_uncertain', event-5., event+20.),
              ('early_relaxation', event+20., event+60.),
              ('cooling', event+60., window_end)]
    result = dict(event_s=event, known_interval_s=[float(times[0]), float(times[-1])],
                  complete_record=bool(times[-1] >= data_end-1e-7), phases={},
                  record_end_s=data_end, requested_window_end_s=window_end,
                  complete_requested_window=bool(times[-1] >= window_end-1e-7),
                  max_observation_gap_s=float(np.max(np.diff(times))),
                  observation_policy=observation_policy(),
                  temperature_error_units='K; temperature DIFFERENCES have the same numerical value in degC',
                  relative_temperature_reference='Observed absolute thermodynamic temperature in K, never degC',
                  relative_rise_reference='Largest observed rise above the initial temperature within the phase; undefined for rise <=1 K',
                  interpretation='Cooling is the primary diagnostic. Burst RMSE is descriptive, not an exact peak target.')
    for name, left, right in phases:
        lo, hi = max(left, times[0], data_start), min(right, times[-1], data_end)
        if hi <= lo:
            continue
        grid = np.unique(np.r_[lo, np.arange(np.ceil(lo), hi, 1.), hi])
        observed = dataio.observed(data, grid)
        dt = np.diff(grid)
        resolved = resolved_samples(times, .5*(grid[:-1]+grid[1:]))
        channels = {}
        for key in dataio.CHANNELS:
            target = observed[key]
            predicted = np.interp(grid, times, run['temperature_k'][key])
            error = predicted-target
            # Missing scans break quadrature intervals, never bridge them.
            valid = np.isfinite(error[:-1]) & np.isfinite(error[1:]) & resolved
            duration = float(np.sum(dt[valid]))
            if not duration:
                continue
            def integral(values):
                return float(np.sum(.5*(values[:-1]+values[1:])[valid]*dt[valid]))/duration
            rmse=float(np.sqrt(integral(error*error)))
            supported=np.r_[valid,False]|np.r_[False,valid]
            rise=float(np.max(np.abs(target[supported]-initial[key][0])))
            channels[key] = dict(time_weighted_rmse_k=rmse,
                relative_kelvin_rms=float(np.sqrt(integral((error/target)**2))),
                observed_rise_scale_k=rise,
                rmse_over_observed_rise=rmse/rise if rise>1. else None,
                time_weighted_bias_k=integral(error), covered_s=duration,
                coverage_fraction=duration/float(hi-lo),used_for_calibration=key in CALIBRATION_CHANNELS)
        result['phases'][name] = dict(interval_s=[float(lo), float(hi)],
            complete=bool(times[0] <= left+1e-7 and times[-1] >= right-1e-7),
            numerical_coverage_fraction=float(np.sum(dt[resolved]))/float(hi-lo), channels=channels)
    return result


def plot_cooling(out, runs, data, labels, *, pressure_panel=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    event = dataio.event(data)['time_s']
    end = min(float(data['time_s'][-1]), max(r['time_s'][-1] for r in runs.values()))
    start = event+20.
    if end <= start:
        return
    grid = np.unique(np.r_[start, np.arange(np.ceil(start), end, 1.), end])
    observed = dataio.observed(data, grid)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    panels = [*CALIBRATION_CHANNELS, 'pressure_bar'] if pressure_panel else list(dataio.CHANNELS)
    for ax, key in zip(axes.flat, panels):
        pressure = key == 'pressure_bar'
        offset = 0. if pressure else 273.15
        ax.plot(grid, observed[key]-offset, color='#222222', label='Эксперимент')
        for name, run in runs.items():
            t = np.asarray(run['time_s'])
            lo, hi = max(start, t[0]), min(end, t[-1])
            if hi <= lo:
                continue
            x = np.unique(np.r_[lo, t[(t > lo) & (t < hi)], hi])
            values = np.asarray(dataio.chamber_pressure_pa(run))/1e5 if pressure else run['temperature_k'][key]
            y = np.interp(x, t, values)-offset
            # NaN midpoint breaks an unobserved late interval on the plot.
            mid = .5*(x[:-1]+x[1:])
            missing = mid[~resolved_samples(t, mid)]
            if len(missing):
                full = np.sort(np.r_[x, missing])
                y = np.interp(full, x, y)
                y[np.isin(full, missing)] = np.nan
                x = full
            line, = ax.plot(x, y, label=labels[name])
            if len(missing):
                ax.scatter(x[-1], y[-1], s=15, color=line.get_color())
        ax.axvspan(start, event+60., color='#e0b35b', alpha=.12)
        names={'101':'C1 / ISS','102':'C1 / TIM','107':'C3 / TIM','108':'C3 / ISS',
               'C2_TIM':'C2 / TIM','C2_ISS':'C2 / ISS','pressure_bar':'Давление камеры'}
        ax.set(title=label(key,names.get(key,key)), xlabel='Время, с', ylabel='бар' if pressure else '°C', xlim=(start, end))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('После выброса: охлаждение без масштаба взрывного пика; без подгонки времени')
    fig.savefig(out/'cooling-comparison.png', dpi=140)
    plt.close(fig)
