#!/usr/bin/env python3
"""Measured UPV electrical work and conditional electrothermal decomposition.

No kinetic fit, no CUDA trajectory, no changes to experimental input. Constant
reversible voltages are sensitivity assumptions, NOT measured OCV. Excess
voltage includes polarization; inferred R is not independently measured DCIR.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

# Also works after installation into battery-sim/scripts.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE if (HERE/'upv_r123_data.py').exists()
                       else Path('/home/konstantin/work/battery-sim/scripts')))
import upv_r123_data as dataio
import upv_r123_model as model
from analyze_upv_heat_balance import stack, layer_temperatures, environment_loss


def electrical_integrals(time, current, voltage, query):
    """Exact integrals for two piecewise-linear signals on their recorded span.

    Returns work (J), charge (C), integral I²dt (A²s). No inferred work before
    the first or after the last recorded point. Query-grid independent.
    """
    time, current, voltage, query = map(np.asarray, (time, current, voltage, query))
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError('Strictly increasing electrical timestamps required')
    knots = np.unique(np.r_[time, np.clip(query, time[0], time[-1])])
    i = np.interp(knots, time, current)
    u = np.interp(knots, time, voltage)
    dt = np.diff(knots)
    work = dt*(2*i[:-1]*u[:-1]+i[:-1]*u[1:]+i[1:]*u[:-1]+2*i[1:]*u[1:])/6
    charge = dt*(i[:-1]+i[1:])/2
    squared = dt*(i[:-1]**2+i[:-1]*i[1:]+i[1:]**2)/3
    return tuple(np.interp(query, knots, np.r_[0., np.cumsum(v)])
                 for v in (work, charge, squared))


def conditional_response(voltage, current, reversible_voltage):
    """Quasi-steady excess-voltage proxy; NaN resistance at zero current."""
    voltage, current = np.asarray(voltage), np.asarray(current)
    power = current*(voltage-reversible_voltage)
    resistance = np.full(power.shape, np.nan)
    np.divide(voltage-reversible_voltage, current, out=resistance,
              where=np.abs(current) > 1.)
    return power, resistance


def thermal_budget(scene, data, stop):
    """Same explicit unobserved-layer scenarios as prior heat-balance audit.

    Not rigorous bounds: battery bulk is estimated from surface thermocouples,
    capacities are the fixed initial ones, and lateral loss is a model prior.
    Valid only as a pre-gross-venting diagnostic, not a post-ejection balance.
    """
    t = np.unique(np.r_[data['time_s'][data['time_s'] < stop], stop])
    observed = dataio.observed(data, t)
    observed['ambient'] = np.interp(t, data['time_s'], data['T109_k'])
    capacities = {row[0]: row[1] for row in stack(scene)}
    bands = {}
    for mode in ('observed-only', 'interpolated', 'neighbour-max'):
        temperatures = layer_temperatures(scene, observed, mode)
        stored = sum(capacities[k]*(v-v[0]) for k, v in temperatures.items())
        loss = environment_loss(scene, temperatures, observed)
        lost = np.r_[0., np.cumsum(np.diff(t)*(loss[:-1]+loss[1:])/2)]
        bands[mode] = dict(stored=stored, lost=lost, required=stored+lost)
    return t, bands


def build_report():
    raw, meta = dataio.load('no-cooling')
    marker = dataio.event(raw)
    measured, electrical_meta = dataio.electrical_record()
    offset = marker['time_s']-measured[-1, 0]
    t, u, i, logged_p = (measured[:, 0]+offset, measured[:, 1],
                          measured[:, 2], measured[:, 3])
    scene, _ = model.build('no-cooling', warmup=True)
    stop = scene['simulation']['duration_s']
    tt, bands = thermal_budget(scene, raw, stop)
    floors = (3.65, 4., 4.2)
    checkpoints = np.array([t[0], 100., 300., 500., 700., 800., 850.,
                            stop, 880., 884., 885., 886., t[-2], t[-1]])
    work, charge, squared = electrical_integrals(t, i, u, checkpoints)
    rows = []
    for j, when in enumerate(checkpoints):
        voltage = float(np.interp(when, t, u))
        current = float(np.interp(when, t, i))
        row = dict(time_s=float(when), seconds_before_marker=float(marker['time_s']-when),
                   voltage_v=voltage, current_a=current, power_ui_w=voltage*current,
                   work_ui_j=float(work[j]), charge_c=float(charge[j]),
                   squared_current_a2s=float(squared[j]), conditional={})
        for floor in floors:
            p, r = conditional_response(voltage, current, floor)
            excess = float(work[j]-floor*charge[j])
            condition = dict(excess_power_w=float(p),
                apparent_resistance_ohm=float(r) if np.isfinite(r) else None,
                excess_work_j=excess,
                excess_fraction_of_ui=excess/work[j] if work[j] > 0 else None,
                weighted_apparent_resistance_ohm=excess/squared[j] if squared[j] > 0 else None)
            if when <= stop:
                condition['thermal_residual_scenarios_j'] = {
                    name: float(np.interp(when, tt, band['required'])-excess)
                    for name, band in bands.items()}
            row['conditional'][str(floor)] = condition
        if when <= stop:
            row['thermal_required_scenarios_j'] = {
                name: float(np.interp(when, tt, band['required']))
                for name, band in bands.items()}
        rows.append(row)
    end_work, end_charge, _ = electrical_integrals(t, i, u, [stop, marker['time_s']])
    report = dict(experiment='no-cooling', source=dataio.SOURCE,
        canonical_sha256=meta['canonical_sha256'], electrical=electrical_metadata(electrical_meta, offset),
        pressure_temperature_marker=marker, thermal_audit_stop_s=float(stop),
        assumptions=[
            'Event marker means major gas-release signature, NOT internal damage onset.',
            'Final source OFF record is not proof of a fuse trip; error field is NONE.',
            'Electrical clock aligned by final record to pressure marker; synchronization is unmeasured.',
            'U_rev choices 3.65 / 4.0 / 4.2 V are constant sensitivity priors, not measured OCV bounds.',
            'I*(U-U_rev) is a conditional quasi-steady irreversible-power estimate/ceiling, not measured Joule heat.',
            'Polarization storage, reversible entropic heat, external lead losses and changing chemistry are unresolved.',
            'Apparent R includes polarization; constant-current U/I cannot uniquely identify ohmic R or damage fraction.',
            'Thermal scenarios use surface-to-bulk estimates, initial heat capacities and fixed environment-h prior.',
            'Thermal residual is signed and diagnostic: it is not uniquely R2, nor clipped into positive heat.',
            'No work is inferred outside the electrical recording span.',
            'U/I fields and logger P are separate instrument readings; their maximum discrepancy is reported.',
            'No validation observations used for parameter selection; no parameters fitted here.'],
        thermal_scenarios_are_rigorous_bounds=False,
        model_fits_performed=False,
        ui_minus_logger_power_max_abs_w=float(np.max(np.abs(u*i-logged_p))),
        final_vent_window=dict(start_s=float(stop),end_s=marker['time_s'],
            ui_work_j=float(end_work[1]-end_work[0]),
            conditional_excess_work_j={str(f):float(np.diff(end_work-f*end_charge)[0]) for f in floors}),
        checkpoints=rows)
    return report, (t,u,i), (tt,bands)


def electrical_metadata(meta, offset):
    return dict(**meta, alignment_offset_s=float(offset), synchronization_measured=False,
                alignment='Last electrical OFF record aligned to first corroborated pressure step')


def plot_report(directory, report, electrical, thermal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t, u, i = electrical
    tt, bands = thermal
    event = report['pressure_temperature_marker']['time_s']
    floors = (3.65,4.,4.2)
    colors = ('#56b4e9','#009e73','#e69f00')
    fig, axes = plt.subplots(2,2,figsize=(14,9))
    ax = axes[0,0]
    ax.plot(t,u*i,color='#0072b2',label='Measured U × I')
    for floor,color in zip(floors,colors):
        p,_=conditional_response(u,i,floor)
        ax.plot(t,p,color=color,label=f'Conditional I(U−{floor:g} V)')
    ax.set_ylabel('Power, W'); ax.set_ylim(0,350)
    ax.set_title('Whole record (final spike shown separately)')
    ax = axes[0,1]
    ax.plot(t,u*i,color='#0072b2',label='Measured U × I')
    for floor,color in zip(floors,colors):
        p,_=conditional_response(u,i,floor)
        ax.plot(t,p,color=color,label=f'Conditional I(U−{floor:g} V)')
    ax.set_xlim(event-25,event+.2);ax.set_ylabel('Power, W')
    ax.set_title('Electrical power immediately before source OFF')
    ax = axes[1,0]
    for floor,color in zip(floors,colors):
        _,r=conditional_response(u,i,floor)
        ax.plot(t,r*1000,color=color,label=f'(U−{floor:g} V)/I')
    ax.set_yscale('log');ax.set_ylabel('Apparent resistance, mΩ')
    ax.set_title('Includes polarization; NOT a measured DCIR')
    ax = axes[1,1]
    w,q,_=electrical_integrals(t,i,u,tt)
    ax.fill_between(tt,bands['observed-only']['required']/1000,
                    bands['neighbour-max']['required']/1000,color='#cc79a7',alpha=.18,
                    label='Unobserved-layer scenarios (not strict bounds)')
    ax.plot(tt,bands['interpolated']['required']/1000,color='#cc79a7',lw=2,
            label='Estimated required thermal input')
    ax.plot(tt,w/1000,color='#0072b2',label='Measured electrical work ∫UI dt')
    for floor,color in zip(floors,colors):
        ax.plot(tt,(w-floor*q)/1000,color=color,label=f'Conditional excess work ({floor:g} V)')
    ax.set_ylabel('Cumulative energy, kJ');ax.set_title('Thermal/electrical separation before major venting')
    for ax in axes.flat:
        ax.set_xlabel('Experimental time, s (clock alignment assumed)')
        ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle('UPV no cooling — measured electrical input vs conditional heat estimates')
    fig.tight_layout();fig.savefig(directory/'electrical-thermal-audit.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5))
    for floor,color in zip(floors,colors):
        excess=w-floor*q
        ax.fill_between(tt,(bands['observed-only']['required']-excess)/1000,
                        (bands['neighbour-max']['required']-excess)/1000,color=color,alpha=.12)
        ax.plot(tt,(bands['interpolated']['required']-excess)/1000,color=color,
                label=f'Thermal need − conditional excess work ({floor:g} V)')
    ax.axhline(0,color='0.4',lw=1);ax.grid(alpha=.25);ax.legend(fontsize=9)
    ax.set(xlabel='Experimental time, s (clock alignment assumed)',ylabel='Signed residual, kJ',
           title='Candidate chemical heat budget — NOT uniquely identified R2')
    fig.tight_layout();fig.savefig(directory/'chemical-residual.png',dpi=150);plt.close(fig)


def main():
    cli=argparse.ArgumentParser(__doc__)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    report,electrical,thermal=build_report()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    t,u,i=electrical
    w,q,sq=electrical_integrals(t,i,u,t)
    with (args.output/'electrical-derived.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['time_s','voltage_v','current_a','power_ui_w','work_ui_j','charge_c',
            'squared_current_a2s','Rapp_Urev3.65_ohm','Rapp_Urev4.0_ohm','Rapp_Urev4.2_ohm'])
        resistances=[conditional_response(u,i,f)[1] for f in (3.65,4.,4.2)]
        for j in range(len(t)):
            writer.writerow([t[j],u[j],i[j],u[j]*i[j],w[j],q[j],sq[j]]+
                            [r[j] if np.isfinite(r[j]) else '' for r in resistances])
    plot_report(args.output,report,electrical,thermal)
    print(args.output/'report.json')
    print(args.output/'electrical-thermal-audit.png')


if __name__=='__main__':main()
