"""Fixed-volume transport experiments from a single committed physical state."""
import numpy as np

import fsri30_transport_model as transport

SPACE = tuple(p for p in transport.SPACE if p.name != 'upper_sleeve_fraction')


def initial_point():
    return np.array([p.encode(p.initial) for p in SPACE])


def decode(x):
    if len(x) != len(SPACE):
        raise ValueError('Wrong fixed-volume suffix dimension')
    # log/exp can produce upper+1 ulp at x=1; retain exact declared endpoints
    # before encoding into the full transport profile again.
    return {p.name: min(p.upper,max(p.lower,p.decode(float(v)))) for p, v in zip(SPACE, x)}


def point(**changes):
    return np.array([p.encode(changes.get(p.name, p.initial)) for p in SPACE])


def candidate(base, x):
    s = transport.candidate(base, transport.point(**decode(x), upper_sleeve_fraction=0.))
    if s['gas_network']['compartments'] != base['gas_network']['compartments'] or s['ports'] != base['ports']:
        raise ValueError('A vented conditioned suffix may NOT remap volumes or ports')
    s['metadata'].update(conditioned_transport_diagnostic=True, full_warmup_validation=False)
    return s


def experiments(anchor=None):
    baseline=decode(initial_point() if anchor is None else anchor)
    def change(**values):return point(**(baseline | values))
    return [
        ('shared-facing-cavities', change(shared_gap_fraction=1.)),
        ('longer-local-residence', change(exhaust_area_scale=.05)),
        ('hot-directed-contact', change(hot_contact_span=30.)),
        ('faster-local-redistribution', change(passage_area_scale=4., mixing_flow_fraction=.9)),
        ('combined-transport', change(shared_gap_fraction=.75, exhaust_area_scale=.05,
            hot_contact_span=30., mixing_flow_fraction=.55)),
    ]


def energy_delivery(run):
    """Accepted CUDA film heat, not h*dT reconstructed from sparse samples."""
    objects = {}
    for row in run.get('gas_film_trace', {}).values():
        solid = row['solid']
        if not solid.startswith('B') or ':case:' not in solid:
            continue
        ident = solid.split(':')[0]
        value = objects.setdefault(ident, dict(net_j=0., positive_j=0., negative_j=0.))
        for key, field in (('net_j', 'heat_to_solid_j'), ('positive_j', 'positive_step_heat_j'),
                           ('negative_j', 'negative_step_heat_j')):
            value[key] += row[field][-1]
    totals = run.get('process_totals', {})
    def delta(field):
        a = totals.get(field)
        return None if a is None else float(a[-1]-a[0])
    return dict(shells=objects, global_gas_reaction_heat_j=delta('gas_released_energy_j'),
        global_gas_sensible_change_j=delta('gas_sensible_energy_j'),
        global_gas_chemical_change_j=delta('gas_chemical_j'),
        all_gas_volumes_included=bool(totals), interval_s=[run['time_s'][0], run['time_s'][-1]])
