#!/usr/bin/env python3
"""Read-only tail heat-path audit; radiation estimate is NOT simulated heat."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from fsri30_data import save,sha

SIGMA=5.670374419e-8


def read(path):
    if path.suffix=='.gz':
        with gzip.open(path,'rt')as f:return json.load(f)
    return json.loads(path.read_text())


def tail_flows(run,start=3300.,end=3500.):
    t=np.array(run['time_s'])
    if t[-1]<end-1e-7:raise ValueError('Requested flow window has not been observed')
    a=int(np.argmin(abs(t-start)));b=int(np.argmin(abs(t-end)));dt=t[b]-t[a]
    channels=[]
    for key,v in run['channels'].items():
        if not key.startswith('vent-'):continue
        channels.append(dict(channel=key,area_m2=v['effective_area_m2'][b],
            mixed_mass_each_way_kg_s=(v['mixed_mass_kg'][b]-v['mixed_mass_kg'][a])/dt,
            pressure_net_mass_kg_s=(v['transferred_mass_kg'][b]-v['transferred_mass_kg'][a])/dt,
            net_thermal_w=(v['delivered_gas_energy_j'][b]-v['delivered_gas_energy_j'][a])/dt))
    totals=run['process_totals']
    return dict(start_s=float(t[a]),end_s=float(t[b]),channels=channels,
        module_vent_net_thermal_w=float(sum(c['net_thermal_w']for c in channels)),
        ambient_thermal_w=float((totals['escaped_energy_j'][b]-totals['escaped_energy_j'][a])/dt),
        r3_w=float((totals['gas_released_energy_j'][b]-totals['gas_released_energy_j'][a])/dt),
        note='Channel delivered energy INCLUDES pressure transport and sensible counterexchange, not advection alone')


def radiative_potential(scene,run):
    if run.get('partial') or run['time_s'][-1]<5945.-1e-7:raise ValueError('Final solid fields require a complete full trajectory')
    ports={p['id']:p for p in scene['ports']};cells={c['id']:c for c in scene['cells']}
    area=0.;center_power=0.;surface_power=0.;ambient=298.15
    for contact in scene['gas_network']['thermal_contacts']:
        ident=contact['port_id']
        if not ident.startswith('gasmesh:ABS:') or ':outside:' not in ident:continue
        port=ports[ident];cell=cells[port['cell_id']];axis='xyz'.index(port['surface'][1])
        thickness=cell['geometry']['dimensions_m'][axis];kappa=cell['thermal']['conductivity_w_per_m_k'][axis]
        half_resistance=thickness/(2*kappa);h=1/(1/8.+half_resistance)
        law=contact['connectivity']
        reference=law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1+np.exp(-law['geometry_reference_logit']))
        patch_area=reference/h;temperature=run['final_fields_k'][cell['id']][contact['thermal_volume']]
        # Surface estimate assuming 8 W/(m2 K) convection to cold ambient.
        # Current hot local gas makes the actual surface closer to its center.
        surface=(temperature+8.*half_resistance*ambient)/(1+8.*half_resistance)
        area+=patch_area;center_power+=patch_area*SIGMA*(temperature**4-ambient**4)
        surface_power+=patch_area*SIGMA*(surface**4-ambient**4)
    return dict(time_s=float(run['time_s'][-1]),external_abs_area_m2=float(area),
        blackbody_power_using_cell_center_w=float(center_power),
        blackbody_power_with_half_slab_surface_estimate_w=float(surface_power),
        assumed_emissivity=None,assumed_view_factor=None,added_to_solver=False,
        scope='Instantaneous potential per unit effective emissivity/view factor; not a predicted cooling trajectory. Adding radiation would change surface temperatures.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene',type=Path,required=True);ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    scene=json.loads(args.scene.read_text());run=read(args.run)
    result=dict(scene_sha256=sha(args.scene),run_sha256=sha(args.run),flows=tail_flows(run),
        radiation=radiative_potential(scene,run),calibrated=False,validation_passed=False)
    save(args.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
