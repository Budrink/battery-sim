"""Coarsen only exterior air/ABS, retaining every 3x3x3 battery gas volume.

Different exterior meshes require fresh trajectories and explicit paired
comparison. Coarse scores cannot be promoted as 20 mm validation.
"""
from copy import deepcopy
from fsri30_multilevel import remesh


def exterior_screen(base, spacing_mm):
    scene=remesh(base,spacing_mm)
    if spacing_mm==20:return scene
    batteries={c['id']:c for c in base['cells'] if c.get('role')=='battery'}
    for c in scene['cells']:
        if c['id'] in batteries:c['thermal']['divisions']=deepcopy(batteries[c['id']]['thermal']['divisions'])
    gas=scene['gas_network']; old=base['gas_network']
    gas['compartments']=[n for n in gas['compartments'] if not n['id'].startswith('inside-')]+[
        deepcopy(n) for n in old['compartments'] if n['id'].startswith('inside-')]
    gas['channels']=[e for e in gas['channels'] if not e['id'].startswith(('pore-','rupture-'))]+[
        deepcopy(e) for e in old['channels'] if e['id'].startswith(('pore-','rupture-'))]
    for name in batteries:
        if len([n for n in gas['compartments'] if n['id'].startswith('inside-'+name+'-')])!=27:
            raise ValueError('Internal battery pores must remain 3x3x3')
    scene['metadata']['search_resolution']['battery_divisions']=[3,3,3]
    scene['metadata']['search_resolution']['screening_only']=True
    return scene
