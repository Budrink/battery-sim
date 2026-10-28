"""Lumped vessel audit from accepted heat records, not a fitted heat source."""
import numpy as np


def wall_inventory(scene, run):
    walls = [c for c in scene['cells'] if c['id'] == 'chamber-wall']
    if len(walls) != 1:
        raise ValueError('Expected one vessel thermal proxy')
    wall = walls[0]
    if wall.get('conversion') or wall.get('electrical') or wall.get('runaway', {}).get('energy_j', 0.):
        raise ValueError('Vessel audit assumes a passive, constant-capacity wall')
    ports = {p['id']: p for p in scene.get('ports', [])}
    for link in scene.get('interfaces', []):
        if any(ports[link[key]]['cell_id'] == wall['id'] for key in ('port_a', 'port_b')):
            raise ValueError('Vessel has an unaccounted solid interface')
    times = np.asarray(run['time_s'])
    temperature = np.asarray(run['thermal_paths']['objects']['chamber-wall'][
        'reference_capacity_weighted_temperature_k'])
    records = [np.asarray(c['heat_to_solid_j']) for c in run['gas_film_trace'].values()
               if c['solid'] == 'chamber-wall']
    if not records:
        raise ValueError('Accepted gas/wall heat trace required')
    heat = np.sum(records, axis=0)
    heat -= heat[0]
    capacity = wall['thermal']['heat_capacity_j_per_k']
    stored = capacity*(temperature-temperature[0])
    # Only gas contacts and the explicit room boundary touch this wall.
    # This is inferred room removal, not an independently measured heat flux.
    room = heat-stored
    samples = []
    for at in sorted({times[0], 750., 930., 950., times[-1]}):
        if not times[0] <= at <= times[-1]:
            continue
        samples.append(dict(time_s=float(at), temperature_c=float(np.interp(at, times, temperature))-273.15,
            net_heat_from_gas_j=float(np.interp(at, times, heat)),
            stored_energy_change_j=float(np.interp(at, times, stored)),
            inferred_heat_to_room_j=float(np.interp(at, times, room))))
    return dict(capacity_j_per_k=capacity, measured_vessel_mass=False,
                wall_mass_probe=scene.get('metadata', {}).get('wall_mass_probe'), snapshots=samples)
