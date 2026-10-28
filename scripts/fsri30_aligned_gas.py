"""Attach aligned cylindrical cut-cell gas geometry to a fresh six-zone scene.

This is an engineering reconstruction, not experimental CAD. The outer air
mesh, material parameters and chemistry stay unchanged. The central lid hole
is the user's 1 inch hypothesis; four bolted openings are treated as closed.
"""
from collections import defaultdict
from copy import deepcopy
import math

from fsri30_cylinder_cutcell import (aligned_cylinder_gas_grid, aligned_gas_faces,
                                      cylinder_box_intersection, cylinder_side_area)
from generate_fsri30 import connectivity


def _disk_area(low, high, center, radius):
    c=(center[0],center[1],0.)
    return cylinder_box_intersection((low[0],low[1],-.5),(high[0],high[1],.5),c,radius,1.)['displaced_volume_m3']


def _overlap(a, b, c, d):
    return max(0.,min(b,d)-max(a,c))


def align_inner_gas(source, max_step=.0325, opening_diameter=.0254):
    s=deepcopy(source); g=s['gas_network']; meta=s['metadata']
    if abs(max_step-.0325)>1e-12:
        raise ValueError('This reconstruction uses 32.5 mm maximum spacing')
    batteries={c['id']:c for c in s['cells'] if c.get('role')=='battery'}
    if len(batteries)!=30 or any(c['thermal']['divisions']!=[3,1,2] for c in batteries.values()):
        raise ValueError('Expected 30 fresh six-zone batteries')
    t=meta['geometry_budget']['abs_thickness_m']
    outer=meta['geometry_reconstruction']['enclosure_dimensions_m']
    inner=[v-2*t for v in outer]
    cyl=[dict(id=name,center=meta['approximate_positions_m'][name],radius=.009,height=.065)
         for name in sorted(batteries)]
    axes,voxels=aligned_cylinder_gas_grid(inner,cyl,max_step)
    active=[v for v in voxels if v['free_volume_m3']>1e-16]
    grid={v['id']:v for v in active}
    ids={k:'aligned-inner-%d-%d-%d'%k for k in grid}
    former={n['id'] for n in g['compartments'] if n['id'].startswith(('body-','top-','base-'))}
    old_nodes=[n for n in g['compartments'] if n['id'] in former]
    if not old_nodes: raise ValueError('No original internal gas compartments')
    def chemistry(n):
        return {k:v for k,v in n.items() if k not in ('id','volume_m3','initial_mass_kg')}
    if any(chemistry(n)!=chemistry(old_nodes[0]) for n in old_nodes):
        raise ValueError('Nonuniform gas cannot be remeshed from initial state')
    rho=old_nodes[0]['initial_mass_kg']/old_nodes[0]['volume_m3']
    old_volume=sum(n['volume_m3'] for n in old_nodes)
    new_volume=sum(v['free_volume_m3'] for v in active)
    if not math.isclose(old_volume,new_volume,rel_tol=1e-10,abs_tol=1e-12):
        raise ValueError(f'Internal gas volume mismatch {old_volume} vs {new_volume}')
    nodes=[]
    for key,v in grid.items():
        n=deepcopy(old_nodes[0]);n.update(id=ids[key],volume_m3=v['free_volume_m3'],initial_mass_kg=rho*v['free_volume_m3']);nodes.append(n)
    g['compartments']=[n for n in g['compartments'] if n['id'] not in former]+nodes
    old_channels=g['channels'];channels=[]
    for edge in old_channels:
        if edge['id'].startswith('rupture-'):
            cell=edge['port_id'].split(':')[0]; c=next(x for x in cyl if x['id']==cell)
            xindex=edge['shell_thermal_volume']; x=c['center'][0]+(xindex-1)*2*c['radius']/3
            point=(x,c['center'][1],c['center'][2]+c['height']/2+1e-10)
            targets=[v for v in active if all(v['low_m'][i]-1e-12<=point[i]<v['high_m'][i]-1e-12 for i in range(3))]
            if len(targets)!=1: raise ValueError('Ambiguous rupture receiver: '+edge['id'])
            e=deepcopy(edge);e['to']=ids[targets[0]['id']];channels.append(e)
        elif edge['from'] not in former and edge['to'] not in former:
            channels.append(edge)
    for f in aligned_gas_faces(active,cyl):
        a,b=grid[f['from']],grid[f['to']];axis=f['axis']
        length=((a['high_m'][axis]-a['low_m'][axis])+(b['high_m'][axis]-b['low_m'][axis]))/2
        channels.append(dict(id='aligned-%s-%s'%('-'.join(map(str,f['from'])),'-'.join(map(str,f['to']))),
            kind='passage',**{'from':ids[f['from']],'to':ids[f['to']]},area_m2=f['area_m2'],
            mixing_length_m=length,mixing_axis=axis,discharge_coefficient=.7,opening_pressure_pa=0.))
    # The lid is z-positive. Clip its circular opening against BOTH the
    # internal and external gas rectangles; the four bolted holes stay closed.
    exterior=meta['gas_volume_geometry']; rad=opening_diameter/2; area_sum=0.
    for v in active:
        if not math.isclose(v['high_m'][2],inner[2]/2,abs_tol=1e-11): continue
        for air,box in exterior.items():
            if not math.isclose(box['low_m'][2],outer[2]/2,abs_tol=1e-11): continue
            lo=[max(v['low_m'][k],box['low_m'][k]) for k in (0,1)]
            hi=[min(v['high_m'][k],box['high_m'][k]) for k in (0,1)]
            if any(a>=b for a,b in zip(lo,hi)):continue
            area=_disk_area(lo,hi,(0.,0.),rad)
            if area<=1e-18:continue
            channels.append(dict(id='aligned-vent-%d-%d-%d-%s'%(*v['id'],air),kind='passage',
                **{'from':ids[v['id']],'to':air},area_m2=area,mixing_length_m=t,
                mixing_axis=2,discharge_coefficient=.7,opening_pressure_pa=0.))
            area_sum+=area
    if not math.isclose(area_sum,math.pi*rad*rad,rel_tol=1e-9,abs_tol=1e-13):
        raise ValueError('Central lid opening area does not close')
    g['channels']=channels
    # Preserve external films and rebuild internal exposure from actual surfaces.
    old_films=g['thermal_contacts'];films=[f for f in old_films if f['compartment'] not in former]
    surfaces=defaultdict(list)
    for f in old_films:
        if f['compartment'] in former and f['port_id'].split(':')[0] in batteries:
            surfaces[f['port_id']].append(f)
    def film_template(port):
        group=surfaces[port];f=deepcopy(group[0]);law=f['connectivity']
        for key in law:
            if key.endswith('conductance_w_per_k'):law[key]=sum(x['connectivity'][key] for x in group)
        return f
    for c in cyl:
        cell=batteries[c['id']]; dims=cell['geometry']['dimensions_m'];zcenter=c['center'][2]
        for face in ('-x','+x','-y','+y','-z','+z'):
            port=c['id']+':case:'+face+':outer';template=film_template(port)
            axis='xyz'.index(face[1]); us=(axis+1)%3;vs=(axis+2)%3
            # The shell panels have the same tangential subdivisions as contents.
            pd=[dims[k]+(2*.00018 if k>axis else 0.) if k!=axis else .00018 for k in range(3)]
            portarea=pd[us]*pd[vs]
            for v in active:
                half=0 if (v['low_m'][2]+v['high_m'][2])/2<zcenter else 1
                if axis<2:
                    # y faces are divided into three strips; x faces have one.
                    count=3 if axis==1 else 1
                    for x in range(count):
                        lo=list(v['low_m']);hi=list(v['high_m'])
                        if count==3:
                            span=c['radius']/math.sqrt(2)
                            lo[0]=max(lo[0],c['center'][0]-span+2*span*x/3)
                            hi[0]=min(hi[0],c['center'][0]-span+2*span*(x+1)/3)
                            if lo[0]>=hi[0]:continue
                        area=cylinder_side_area(lo,hi,c['center'],c['radius'],c['height'],face)
                        if area<=1e-18:continue
                        f=deepcopy(template);f['compartment']=ids[v['id']];f['thermal_volume']=x+count*half
                        for k in f['connectivity']:
                            if k.endswith('conductance_w_per_k'):f['connectivity'][k]*=area/portarea*(2*count)
                        films.append(f)
                else:
                    boundary=zcenter+(c['height']/2 if face=='+z' else -c['height']/2)
                    adjoining=v['low_m'][2] if face=='+z' else v['high_m'][2]
                    if not math.isclose(adjoining,boundary,abs_tol=1e-12):continue
                    for x in range(3):
                        lo=list(v['low_m']);hi=list(v['high_m'])
                        lo[0]=max(lo[0],c['center'][0]-c['radius']+2*c['radius']*x/3)
                        hi[0]=min(hi[0],c['center'][0]-c['radius']+2*c['radius']*(x+1)/3)
                        if lo[0]>=hi[0]:continue
                        area=_disk_area(lo,hi,c['center'],c['radius'])
                        if area<=1e-18:continue
                        f=deepcopy(template);f['compartment']=ids[v['id']];f['thermal_volume']=x
                        for k in f['connectivity']:
                            if k.endswith('conductance_w_per_k'):f['connectivity'][k]*=area/portarea*3
                        films.append(f)
    # Partition wall tiles against gas cells. Contact geometry excludes the
    # centre hole, but the existing solid ABS mesh itself remains an approximation.
    ports={p['id']:p for p in s['ports']}
    wallgroups={}
    for f in old_films:
        if f['compartment'] in former and f['port_id'].startswith('gasmesh:ABS:'):
            wallgroups[(f['port_id'],f.get('thermal_volume',-1))]=f
    for (name,idx),template in wallgroups.items():
        port=ports[name];wall=name.split(':')[2];axis='xyz'.index(wall[1]);sign=1 if wall[0]=='+' else -1
        u,v=(axis+1)%3,(axis+2)%3;center=port['centroid_m'];du,dv=port['rectangle_m']
        for box in active:
            boundary=box['high_m'][axis] if sign>0 else box['low_m'][axis]
            if not math.isclose(boundary,sign*inner[axis]/2,abs_tol=1e-11):continue
            lu=max(center[u]-du/2,box['low_m'][u]);hu=min(center[u]+du/2,box['high_m'][u])
            lv=max(center[v]-dv/2,box['low_m'][v]);hv=min(center[v]+dv/2,box['high_m'][v])
            if lu>=hu or lv>=hv:continue
            area=(hu-lu)*(hv-lv)
            if wall=='+z':area-=_disk_area((lu,lv),(hu,hv),(0.,0.),rad)
            if area<=1e-18:continue
            f=deepcopy(template);f['compartment']=ids[box['id']];f['connectivity']=connectivity(area,8.)
            films.append(f)
    g['thermal_contacts']=films
    s.setdefault('gas_history_groups',{})['module']=['aligned-inner-']
    selection=s.setdefault('process_history_selection',{})
    selection['gas_prefixes']=['aligned-inner-','inside-B10-']
    selection['channel_prefixes']=['rupture-','aligned-vent-']
    meta['aligned_inner_gas']=dict(max_step_m=max_step,axial_boundaries_m=axes[2],
        cells=len(active),open_passages=len(aligned_gas_faces(active,cyl)),
        aperture_diameter_m=opening_diameter,aperture_area_m2=area_sum,
        rupture_mapping='one receiver selected at each casing patch centre',
        limitation='ABS solid remains present behind the central hole; 6-zone core is a strip surrogate',
        calibrated=False,validation_passed=False)
    return s
