"""Validated module/exterior cut; no material, inventory or thermal mutations."""
from copy import deepcopy
import math


def module_exits(scene, *, allow_closed=False):
    gas=scene['gas_network'];nodes={n['id']:n for n in gas['compartments']}
    outside={name for name in nodes if name.startswith('air-')}
    if not outside:raise ValueError('Expected resolved exterior air volumes')
    if any(n.get('ambient_boundary') for name,n in nodes.items() if name not in outside):
        raise ValueError('Internal ambient boundary would bypass the module exits')
    exits=[]
    for edge in gas['channels']:
        a,b=edge['from'],edge['to']
        if a not in nodes or b not in nodes:raise ValueError('Unresolved gas endpoint or direct ambient bypass')
        if (a in outside)==(b in outside):continue
        area=edge['area_m2']
        if not (edge['id'].startswith('vent-') and edge['kind']=='passage'
                and a.startswith('body-') and b in outside and math.isfinite(area)
                and (area>0. or (allow_closed and area==0.))):
            raise ValueError('Unqualified module/exterior crossing: '+edge['id'])
        exits.append(edge['id'])
    if len(exits)!=2 or len(set(exits))!=2:raise ValueError('Expected exactly two distinct retained module exits')
    return sorted(exits)


def restrict_exits(scene,scale):
    if not math.isfinite(scale) or not 0.<scale<=1.:
        raise ValueError('Use a positive restriction fraction, not a sealed system')
    cut=set(module_exits(scene));result=deepcopy(scene)
    for edge in result['gas_network']['channels']:
        if edge['id'] in cut:edge['area_m2']*=scale
    return result


def seal_module(scene):
    """Exactly zero mass AND counterexchange across the two retained exits."""
    cut=set(module_exits(scene));result=deepcopy(scene)
    for edge in result['gas_network']['channels']:
        if edge['id'] in cut:edge['area_m2']=0.
    result.setdefault('metadata',{})['sealed_module_diagnostic']=dict(
        channels=sorted(cut),mass_and_mixing_blocked=True,heat_rejection_unchanged=True,
        battery_ruptures_unchanged=True,internal_gas_paths_unchanged=True,
        source='Explicit user-requested counterfactual, not the measured open ABS fixture')
    module_exits(result,allow_closed=True)
    return result


def sealed_cut_audit(scene,run):
    cut=module_exits(scene,allow_closed=True);leaks=[]
    for edge in scene['gas_network']['channels']:
        if edge['id'] in cut and edge['area_m2']!=0.:raise ValueError('Closed cut has an open channel')
    fields=('transferred_mass_kg','transferred_chemical_j','transferred_oxidizer_kg',
            'transferred_vapor_kg','transferred_latent_j','mixed_mass_kg','delivered_gas_energy_j')
    for ident in cut:
        values=run['channels'][ident]
        for field in fields:
            if len(values[field])!=len(run['time_s']):raise ValueError('Incomplete sealed-channel history')
            deviation=max(abs(x-values[field][0]) for x in values[field])
            if not math.isfinite(deviation) or deviation!=0.:leaks.append(dict(channel=ident,field=field,change=deviation))
        if any(x!=0. for x in values['effective_area_m2']):leaks.append(dict(channel=ident,field='effective_area_m2'))
    return dict(accepted=not leaks,channels=cut,leaks=leaks,last_observed_s=run['time_s'][-1],
        scope='Module/exterior cut only; exterior air may still exchange with its ambient reservoir')
