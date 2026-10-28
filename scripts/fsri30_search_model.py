"""Bounded shared parameters and a conservative near-field gas reconstruction.

No material capacities, chemical inventories, ignition schedules or numerical
controls are optimization variables. This is an exploratory FSRI adapter, not
an assertion that the nine published sensors identify thirty component cells.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from generate_fsri30 import BOX

SCHEMA = Path(__file__).resolve().parents[1]/'schemas/scenario-v0.schema.json'


@lru_cache(maxsize=1)
def iteration_bounds():
    """Use the input contract, not an independent drifting Python maximum."""
    properties = json.loads(SCHEMA.read_text())['$defs']['gas_network']['properties']
    return {key: (properties[key]['minimum'], properties[key]['maximum'])
            for key in ('averaged_pressure_iterations', 'averaged_payload_iterations')}


def validate_controls(scene):
    """Cheap, early check of search-generated controls; no solver evolution."""
    gas, sim = scene['gas_network'], scene['simulation']
    for key, (lower, upper) in iteration_bounds().items():
        value = gas[key]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f'{key}={value} outside native input contract [{lower}, {upper}]')
    floor = sim['minimum_dt_s']
    if not math.isfinite(floor) or floor < .005 or not math.isfinite(sim['max_dt_s']) or sim['max_dt_s'] < floor:
        raise ValueError('Invalid search physical time-step bounds')
    interval = gas['coupled_transport_interval_s']
    if not math.isfinite(interval) or interval < floor:
        raise ValueError('Requested gas interval is below the preserved physical floor')


@dataclass(frozen=True)
class Parameter:
    name: str
    lower: float
    upper: float
    initial: float
    log: bool
    meaning: str

    def decode(self, x):
        if not math.isfinite(x) or not 0. <= x <= 1.:
            raise ValueError('Parameter outside unit cube: '+self.name)
        return math.exp(math.log(self.lower)+x*math.log(self.upper/self.lower)) if self.log else self.lower+x*(self.upper-self.lower)

    def encode(self, value):
        if not self.lower <= value <= self.upper:
            raise ValueError('Parameter outside physical bounds: '+self.name)
        return math.log(value/self.lower)/math.log(self.upper/self.lower) if self.log else (value-self.lower)/(self.upper-self.lower)


# All ranges are declared engineering priors, NOT measured confidence bounds.
# The finite bounds prevent absent ABS/radiation physics being hidden in an
# arbitrarily large fuel reserve or a per-cell/contact/time-specific parameter.
SPACE = (
    Parameter('heater_power_w', 4.5, 9., 6., False, 'Unmeasured heater power; published ramp is 0.1 K/s'),
    Parameter('casing_contact_h', 200., 2000., 500., True, 'Shared contents-to-steel contact, W/(m2 K)'),
    Parameter('gas_contact_h', 10., 300., 20., True, 'Shared gas-to-can directed-contact scale, W/(m2 K)'),
    Parameter('passage_open_fraction', .2, 1., 1., False, 'Accessible fraction of reconstructed interstitial passages'),
    Parameter('exhaust_area_scale', .25, 2., 1., True, 'Unmeasured enclosure exit area relative to reconstruction'),
    Parameter('mixing_flow_fraction', .05, .9, .25, False, 'Shared flow-associated mixing coefficient'),
    Parameter('r2_rate_450_per_s', 5e-6, 8e-5, 2e-5, True, 'Shared R2 Arrhenius rate at 450 K; activation fixed'),
    Parameter('r3_rate_600_per_s', .2, 30., 1., True, 'Shared R3 Arrhenius rate at 600 K'),
    Parameter('r3_activation_j_mol', 45000., 100000., 80000., False, 'Shared R3 activation energy'),
)

COLD_R3_PRIOR = dict(temperature_k=298.15, maximum_rate_per_s=1e-6,
    measured=False, rationale='Do not fit hot fire by making initially cold external gas rapidly self-oxidize')


def admissible(point):
    """Cheap, declared kinetic prior; no CUDA run needed to reject this case."""
    p = decode(point)
    rate = p['r3_rate_600_per_s']*math.exp(p['r3_activation_j_mol']/8.314462618*(1/600.-1/298.15))
    return rate <= COLD_R3_PRIOR['maximum_rate_per_s']


def specification():
    return [asdict(p) for p in SPACE]


def initial_point():
    return np.array([p.encode(p.initial) for p in SPACE])


def informed_points():
    """Declared initial hypotheses, NOT invented prior optimization results."""
    hypotheses = [dict(exhaust_area_scale=.25, mixing_flow_fraction=.8,
                       r3_rate_600_per_s=8., r3_activation_j_mol=80000.),
                  dict(exhaust_area_scale=.5, mixing_flow_fraction=.55,
                       r3_rate_600_per_s=3., r3_activation_j_mol=75000.)]
    points = []
    for changes in hypotheses:
        point = np.array([p.encode(changes.get(p.name, p.initial)) for p in SPACE])
        if not admissible(point):
            raise ValueError('Informed initial hypothesis violates cold stability prior')
        points.append(point)
    return points


def decode(point):
    if len(point) != len(SPACE):
        raise ValueError('Wrong parameter dimension')
    return {p.name: p.decode(float(x)) for p, x in zip(SPACE, point)}


def split_hood(scene, layers=1):
    """Resolve an assumed 50-mm hot near field without inventing oxygen/mass.

    Thirty stream tubes follow the existing exits. Their TOTAL volume is
    box footprint * 50mm, removed from the old finite 1m3 receiver. A second
    axial layer is a resolution test, not a fitted geometry change. The
    network uses identical total diffusion resistance and cold, unchoked
    orifice resistance for any layer count (A'=n*A, Cd'=Cd/sqrt(n)). This does
    not promise equivalence for reacting, compressible, non-isothermal flow:
    sensitivity is explicitly measured by the controller.
    """
    if layers not in (0, 1, 2, 4):
        raise ValueError('Unsupported near-field resolution')
    if not layers:
        return
    gas = scene['gas_network']
    if any(c['id'].startswith('plume-') for c in gas['compartments']):
        raise ValueError('Near field already installed')
    hood = next(c for c in gas['compartments'] if c['id'] == 'hood')
    volume = BOX[0]*BOX[1]*.05
    density = hood['initial_mass_kg']/hood['volume_m3']
    hood['volume_m3'] -= volume
    hood['initial_mass_kg'] -= density*volume
    exits = [e for e in gas['channels'] if e['id'].startswith('exhaust-')]
    if len(exits) != 30:
        raise ValueError('Expected thirty explicit existing exhaust paths')
    gas['channels'] = [e for e in gas['channels'] if not e['id'].startswith('exhaust-')]
    for exit_edge in exits:
        j = int(exit_edge['id'].split('-')[-1])
        route = [exit_edge['from']]
        for layer in range(layers):
            ident = f'plume-{layer}-{j:02d}'
            node = deepcopy(hood)
            node.update(id=ident, volume_m3=volume/(30*layers), initial_mass_kg=density*volume/(30*layers))
            gas['compartments'].append(node)
            route.append(ident)
        route.append('hood')
        for k, (a, b) in enumerate(zip(route, route[1:])):
            edge = deepcopy(exit_edge)
            edge.update(id=f'exhaust-path-{j}-{k}', **{'from': a, 'to': b},
                        area_m2=exit_edge['area_m2']*(layers+1),
                        discharge_coefficient=exit_edge['discharge_coefficient']/math.sqrt(layers+1))
            gas['channels'].append(edge)
    selection = scene.get('process_history_selection')
    if selection is not None:
        selection['gas_prefixes'] = sorted(set(selection['gas_prefixes']+['plume-']))
    scene['metadata']['near_field'] = dict(layers=layers, total_volume_m3=volume,
        height_m=.05, inventory_removed_from='hood', measured=False,
        note='Declared spatial hypothesis; no direct thermal link bypasses the ABS or a casing')


def candidate(base, point, *, layers=1, fine=False, accuracy_level=None):
    """Return a fresh scene; never mutate a saved baseline or another trial."""
    s = deepcopy(base)
    p = decode(point)
    split_hood(s, layers)
    for cell in s['cells']:
        if cell['role'] == 'battery':
            cell['conversion']['reaction_rate_per_s'] = p['r2_rate_450_per_s']
        if cell['id'] == 'heater':
            ratio = math.sqrt(p['heater_power_w']/base['metadata']['heater_power_w'])
            for key in ('current_profile_a', 'voltage_profile_v'):
                cell['electrical'][key] = [[t, v*ratio] for t, v in cell['electrical'][key]]
    for shell in s['shells']:
        shell['contents_contact_h'] = p['casing_contact_h']
    gas = s['gas_network']
    gas['mixing_flow_fraction'] = p['mixing_flow_fraction']
    for node in gas['compartments']:
        node['reaction_rate_per_s'] = p['r3_rate_600_per_s']
        node['reaction_activation_j_per_mol'] = p['r3_activation_j_mol']
    for edge in gas['channels']:
        if edge['id'].startswith('free-'):
            edge['area_m2'] *= p['passage_open_fraction']
        elif edge['id'].startswith('exhaust-'):
            edge['area_m2'] *= p['exhaust_area_scale']
    for contact in gas['thermal_contacts']:
        if contact['port_id'].startswith('B'):
            law = contact['connectivity']
            ratio = p['gas_contact_h']/20.
            for key in ('background_conductance_w_per_k', 'active_conductance_w_per_k',
                        'destroyed_background_conductance_w_per_k', 'destroyed_active_conductance_w_per_k'):
                law[key] *= ratio
    level = int(fine) if accuracy_level is None else accuracy_level
    if level not in (0, 1, 2, 3):
        raise ValueError('Numerical accuracy level outside declared safe range')
    if level:
        s['simulation']['max_dt_s'] = base['simulation']['max_dt_s']/2**level
        gas['coupled_transport_interval_s'] = max(s['simulation']['minimum_dt_s'],
            base['gas_network']['coupled_transport_interval_s']/2**level)
        bounds = iteration_bounds()
        # Keep the established accuracy ladder unchanged; larger budgets are
        # selected explicitly for unusually ill-conditioned networks.
        gas['averaged_pressure_iterations'] = min(2048, bounds['averaged_pressure_iterations'][1], 512*2**level)
        gas['averaged_payload_iterations'] = min(bounds['averaged_payload_iterations'][1], 16*2**level)
    # Deliberately keep minimum_dt_s, inventories, heat capacities, all known
    # dimensions, all rupture laws and the absence of scheduled failures.
    numerical_name = 'standard' if level == 0 else 'fine' if level == 1 else f'fine-{level}'
    s['metadata'].update(search_parameters=p, search_numerics=numerical_name,
        comparison_scope='Exploratory envelope fit; NOT mapped-sensor calibration or independent validation')
    s['metadata']['heater_power_w'] = p['heater_power_w']
    s['name'] = 'FSRI 30: bounded exploratory cascade calibration'
    validate_controls(s)
    return s


def conserved_fingerprint(scene):
    """A small auditable contract for fields the optimizer may NOT adjust."""
    return dict(simulation_minimum_dt_s=scene['simulation']['minimum_dt_s'],
        batteries=[dict(id=c['id'], geometry=c['geometry'], pose=c['pose'], thermal=c['thermal'],
            conversion={k: v for k, v in c['conversion'].items() if k != 'reaction_rate_per_s'})
            for c in scene['cells'] if c['role'] == 'battery'],
        shells=[{k: v for k, v in x.items() if k != 'contents_contact_h'} for x in scene['shells']],
        ruptures=[e for e in scene['gas_network']['channels'] if e['kind'] == 'rupture'],
        scheduled_failures=scene['primary_failures'])
