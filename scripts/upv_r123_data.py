"""Immutable UPV inputs and explicit synchronization assumptions; no solver."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/experiments/upv'
SOURCE = 'https://zenodo.org/records/22273467/files/TR_Experimental_Data_Description.pdf'
WITNESSES = ('101', '102', '107', '108')
CHANNELS = dict(**{c: (c,) for c in WITNESSES}, C2_TIM=('103', '104'), C2_ISS=('105', '106'))


def chamber_pressure_pa(run):
    """Read-only chamber observer, with fallback for archived hub-based runs."""
    observer = run.get('gas_observers', {}).get('chamber')
    return observer['pressure_pa'] if observer else run['gas']['chamber']['pressure_pa']


def load(name):
    manifest = json.loads((DATA/'manifest.json').read_text())
    record = manifest['experiments'][name]
    raw = (DATA/record['canonical_csv']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record['canonical_sha256']:
        raise ValueError('Experimental CSV checksum changed')
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    arrays = {k: np.array([float(r[k]) if r[k] else np.nan for r in rows])
              for k in rows[0] if k == 'time_s' or k.startswith(('T', 'U'))}
    return arrays, record


def event(data):
    time = data['time_s']; pressure = 2.5*data['U114_v']
    for i in np.flatnonzero((np.diff(pressure) >= .5) & (np.diff(time) <= 2.)) + 1:
        end = np.searchsorted(time, time[i]+2., side='right')
        if any(np.any(data['T'+c+'_k'][i:end]-data['T'+c+'_k'][i-1] > 50.)
               for c in ('103', '104', '105', '106', '109')):
            return dict(time_s=float(time[i]), previous_s=float(time[i-1]), index=int(i),
                        bracket_s=float(time[i]-time[i-1]))
    raise ValueError('No corroborated venting marker')


def electrical_record():
    member = '01_MiniModule_NoCooling/MiniModule_NoCooling_Electrical_20250701.csv'
    with zipfile.ZipFile(DATA/'source-v1.zip') as archive:
        raw = archive.read(member)
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig')), delimiter=';'))
    def number(text):
        match = re.match(r'\s*([+-]?[\d]+(?:[,.]\d+)?)', text)
        if not match: raise ValueError('Invalid electrical instrument value: '+text)
        return float(match[1].replace(',', '.'))
    def seconds(text):
        h, m, s = text.replace(',', '.').split(':')
        return 3600*float(h)+60*float(m)+float(s)
    origin = seconds(rows[0]['Time'])
    result = np.array([(seconds(r['Time'])-origin, number(r['U actual']),
                        number(r['I actual']), number(r['P actual'])) for r in rows])
    if np.any(np.diff(result[:, 0]) <= 0) or np.any(~np.isfinite(result)):
        raise ValueError('Unordered electrical log')
    return result, dict(member=member, sha256=hashlib.sha256(raw).hexdigest())


def current_drive(name, data, marker):
    if name == 'no-cooling':
        values, meta = electrical_record()
        # The instrument clocks are not synchronized. Continuity loss is the
        # only common physical landmark; its alignment is a declared assumption,
        # NOT a fitted delay or an ignition prescription for the chemistry.
        offset = marker['time_s']-values[-1, 0]
        profile = [[0., 0.], [max(0., offset-1e-3), 0.]]
        profile += [[float(t+offset), float(i)] for t, _, i, _ in values]
        volts = [[0., float(values[0, 1])], [max(0., offset-1e-3), float(values[0, 1])]]
        volts += [[float(t+offset), float(u)] for t, u, _, _ in values]
        return profile, volts, dict(**meta, alignment='last zero-current reading aligned to first pressure step',
                             offset_s=offset, onset_s=offset, synchronization_measured=False,
                             electrical_power_not_used_as_heat=True)
    # Test 3 has voltage, not a power-source current log. 35 A comes from the
    # published abuse protocol. Voltage compliance detects supply shutdown only.
    cut = np.flatnonzero(data['U115_v'] > 60.)
    stop = float(data['time_s'][cut[0]]) if len(cut) else marker['time_s']
    times = data['time_s']; voltage = data['U115_v']
    hold = times <= stop
    volts = [[float(t), float(u)] for t, u in zip(times[hold], voltage[hold])]
    if not volts or volts[-1][0] < stop:
        volts.append([stop, float(np.interp(stop, times, voltage))])
    volts.append([stop+1e-3, float(volts[-1][1])])
    # The DAQ record contains a pre-charge rest. Infer only the electrical
    # switch-on from its measured voltage step; never align thermal curves.
    jumps=np.flatnonzero((np.diff(voltage)>.2)&(times[1:]<stop)&(voltage[:-1]<4.3))
    if not len(jumps):raise ValueError('No identifiable charging-voltage step in validation input')
    onset_index=int(jumps[0])+1
    bracket=[float(times[onset_index-1]),float(times[onset_index])]
    onset=sum(bracket)/2.
    return [[0.,0.],[onset,0.],[onset+1e-3,35.],[stop,35.],[stop+1e-3,0.]], volts, dict(
        current_a=35., source='Published 1C overcharge protocol; current trace unavailable',
        onset_s=onset,onset_bracket_s=bracket,current_start_inferred_not_measured=True,
        onset_source='First measured voltage step >0.2 V from resting cell; midpoint of scan bracket',
        no_temperature_time_shift_or_fit=True,
        cutoff_s=stop, cutoff='First terminal voltage above 60 V: loss of continuity/compliance, not cell heat')


def charge_onset_s(name, data, marker):
    """Scene time when charging current starts. Rest before this is I=0."""
    _, _, meta = current_drive(name, data, marker)
    return float(meta['onset_s'])


def observation_times(data, marker, duration, warmup=False, dense_after=None, dense_span=10.):
    grid = np.unique(np.r_[0., np.arange(5., duration, 5.), duration])
    if dense_after is not None:
        start, stop = float(dense_after), float(dense_after)+float(dense_span)
        native = data['time_s'][(data['time_s'] >= start) & (data['time_s'] <= min(stop, duration))]
        return np.unique(np.r_[grid[grid <= duration], start, min(stop, duration), native]).tolist()
    if warmup:
        return grid.tolist()
    relative = data['time_s']-marker['time_s']
    dense = data['time_s'][(relative >= -5.) & (relative <= 30.)]
    return np.unique(np.r_[grid, dense[dense <= duration]]).tolist()


def observed(data, times):
    at = np.asarray(times); time = data['time_s']; result = {}
    for name, ids in CHANNELS.items():
        y = np.mean([data['T'+i+'_k'] for i in ids], axis=0)
        # Do not interpolate across missing scans: preserve a missing result.
        j = np.clip(np.searchsorted(time, at, side='right')-1, 0, len(time)-2)
        f = (at-time[j])/(time[j+1]-time[j])
        result[name] = np.where(f == 0., y[j], np.where(f == 1., y[j+1], y[j]+f*(y[j+1]-y[j])))
    result['pressure_bar'] = np.interp(at, time, 2.5*data['U114_v'])
    return result
