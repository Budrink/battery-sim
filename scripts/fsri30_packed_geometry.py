"""Fresh close-packed cylindrical-layout approximation for the FSRI fixture.

The thermal cores and shells remain equal-volume prisms. Gas control volumes
use physical cylinder cross sections and bounded nearest-centre polygons. A
single body compartment per cylinder lumps disconnected triangular pockets;
this deliberately economical graph is not resolved CFD or measured CAD.
"""
from copy import deepcopy
import math

import numpy as np

from free_gas_mesh import face_tiles, add_face_port, rectangle_intersection, rectangle_overlap
from fsri30_reconstruction import partition, polygon_area
from fsri30_resolved_gas import intersection_area
from generate_fsri30 import BOX, DIAMETER, HEIGHT, connectivity


def close_packed(base):
    s=deepcopy(base);meta=s['metadata'];gas=s['gas_network']
    if 'can_contact_hypothesis' in meta: raise ValueError('Rebuild geometry before adding contacts')
    if meta['air_mesh']['spacing_m'] not in (.02,.04,.08): raise ValueError('Unsupported external gas mesh')
    old=meta['approximate_positions_m']; positions=[]
    for col,(count,offset) in enumerate(zip((7,8,8,7),(.5,0.,.5,1.))):
        for row in range(count):positions.append([col*DIAMETER*math.sqrt(3)/2,(row+offset)*DIAMETER,0.])
    points=np.array(positions);points[:,:2]-=(points[:,:2].min(0)+points[:,:2].max(0))/2
    new={f'B{i:02d}':p.tolist() for i,p in enumerate(points)}
    thickness=meta['geometry_budget']['abs_thickness_m'];inner=np.array(BOX)-2*thickness
    polygons,neighbors,walls=partition(points,inner)
    if any(abs(np.linalg.norm(points[a]-points[b])-DIAMETER)>1e-10 for a,b in neighbors):
        raise ValueError('A nearest-cylinder edge is not tangential')
    # Move each solver prism and its existing film heater by the same displacement.
    for cell in s['cells']:
        name=cell['id']
        if name in new:cell['pose']['position_m']=new[name]
        elif name=='heater':
            h=meta['heated_component'];delta=np.array(new[h])-np.array(old[h])
            cell['pose']['position_m']=(np.array(cell['pose']['position_m'])+delta).tolist()
    for port in s['ports']:
        if port['cell_id']=='heater':
            h=meta['heated_component'];delta=np.array(new[h])-np.array(old[h])
            port['centroid_m']=(np.array(port['centroid_m'])+delta).tolist()
    meta['approximate_positions_m']=new
    meta['neighbor_pairs']=[list(x) for x in sorted(neighbors)]
    meta['neighbour_pairs']=[list(x) for x in sorted(neighbors)]
    geometry=meta['geometry_reconstruction']; geometry['polygon_xy_m']=[p.tolist() for p in polygons]
    geometry['shared_faces']=[dict(a=a,b=b,length_m=length) for (a,b),length in sorted(neighbors.items())]
    geometry['wall_faces']=walls
    cylinder=math.pi*DIAMETER**2/4;cap_height=(inner[2]-HEIGHT)/2
    areas=[polygon_area(p) for p in polygons]
    if min(areas)<=cylinder or cap_height<=0:raise ValueError('Nonpositive free gas geometry')
    nodes={n['id']:n for n in gas['compartments']}
    volume_before=sum(n['volume_m3'] for n in nodes.values())
    mass_before=sum(n['initial_mass_kg'] for n in nodes.values())
    for j,area in enumerate(areas):
        for band,vol in (('body',(area-cylinder)*HEIGHT),('top',area*cap_height),('base',area*cap_height)):
            n=nodes[f'{band}-{j:02d}'];rho=n['initial_mass_kg']/n['volume_m3']
            n['volume_m3']=float(vol);n['initial_mass_kg']=float(rho*vol)
    if not math.isclose(sum(n['volume_m3'] for n in nodes.values()),volume_before,rel_tol=1e-12):raise ValueError('Gas volume changed')
    if not math.isclose(sum(n['initial_mass_kg'] for n in nodes.values()),mass_before,rel_tol=1e-12):raise ValueError('Gas inventory changed')
    # Top and base communicate with the full free cross-section. In-plane
    # faces are bounded by tangent-cylinder Voronoi polygons. Because the body
    # regions are lumped, the face graph does not resolve individual pockets.
    gas['channels']=[e for e in gas['channels'] if not e['id'].startswith(('free-','cap-','vent-'))]
    def passage(name,a,b,area,length,axis,factor=None):
        q=dict(id=name,kind='passage',**{'from':a,'to':b},area_m2=float(area),mixing_length_m=float(length),
            mixing_axis=axis,discharge_coefficient=.7,opening_pressure_pa=0.)
        if factor is not None:q['hydraulic_resistance_factor']=factor
        gas['channels'].append(q)
    for j,area in enumerate(areas):
        for band in ('top','base'):
            passage(f'cap-{band}-{j}',f'body-{j:02d}',f'{band}-{j:02d}',area-cylinder,(HEIGHT+cap_height)/2,2)
    for (a,b),length in sorted(neighbors.items()):
        distance=float(np.linalg.norm(points[b,:2]-points[a,:2]));axis=int(np.argmax(abs(points[b,:2]-points[a,:2])))
        for band,height in (('body',HEIGHT),('top',cap_height),('base',cap_height)):
            passage(f'free-{band}-{a}-{b}',f'{band}-{a:02d}',f'{band}-{b:02d}',length*height,distance,axis)
    # Intersect each known end vent with the new wall partition and existing
    # resolved exterior cells. Its area remains the prior, not a measurement.
    side=math.sqrt(geometry['unmeasured_total_aperture_area_m2']/2)
    hole=[-side/2,side/2,-side/2,side/2];half=np.array(BOX)/2
    layouts=meta['gas_volume_geometry'];apertures=[]
    for face in ('-y','+y'):
        sign=1 if face[0]=='+' else -1
        for wall in (w for w in walls if w['face']==face):
            j=wall['component'];xs=np.array(wall['endpoints'])[:,0]
            region=rectangle_intersection(hole,[-side/2,side/2,float(xs.min()),float(xs.max())]);total=0.
            for name,v in layouts.items():
                low=np.array(v['low_m']);high=np.array(v['high_m'])
                if abs((low[1] if sign>0 else high[1])-sign*half[1])>1e-11:continue
                rect=[low[2],high[2],low[0],high[0]];area=rectangle_overlap(region,rect)
                if area<=1e-16:continue
                length=float(np.linalg.norm((low+high)/2-points[j]))
                passage(f'vent-{face}-{j}-{name}',f'body-{j:02d}',name,area,length,1,1.)
                total+=area
            if total>1e-16:apertures.append(dict(face=face,component=j,area_m2=total))
        if not math.isclose(sum(x['area_m2'] for x in apertures if x['face']==face),side**2,rel_tol=1e-9):
            raise ValueError('Packed vent failed area closure')
    geometry['central_front_back_apertures']=apertures
    # Reassign inner wall films according to the new body/top/base partitions.
    is_inside=lambda p:p.startswith('gasmesh:ABS:') and ':inside:' in p
    gas['thermal_contacts']=[f for f in gas['thermal_contacts'] if not is_inside(f['port_id'])]
    s['ports']=[p for p in s['ports'] if not is_inside(p['id'])]
    wall_lookup={(w['face'],w['component']):w for w in walls}
    totals={}
    for cell in (c for c in s['cells'] if c['id'].startswith('ABS:')):
        face=cell['id'].split(':')[1];axis='xyz'.index(face[1]);sign=1 if face[0]=='+' else -1
        inside=('-' if sign>0 else '+')+face[1];total=0.
        for index,rect,point in face_tiles(cell,inside):
            port=None
            for j in range(30):
                if axis==2:
                    bands=[('top' if sign>0 else 'base',intersection_area(polygons[j],rect))]
                else:
                    wall=wall_lookup.get((face,j))
                    if wall is None:continue
                    other=1-axis;ends=np.array(wall['endpoints'])[:,other];bands=[]
                    for band,zlo,zhi in (('body',-HEIGHT/2,HEIGHT/2),('top',HEIGHT/2,inner[2]/2),('base',-inner[2]/2,-HEIGHT/2)):
                        region=[float(ends.min()),float(ends.max()),zlo,zhi] if axis==0 else [zlo,zhi,float(ends.min()),float(ends.max())]
                        overlap=rectangle_intersection(rect,region);area=rectangle_overlap(rect,region)
                        if axis==1 and band=='body':area-=rectangle_overlap(overlap,hole)
                        bands.append((band,area))
                for band,area in bands:
                    if area<=1e-15:continue
                    if port is None:port=add_face_port(s,cell,inside,index,rect,point,'inside')
                    gas['thermal_contacts'].append(dict(compartment=f'{band}-{j:02d}',port_id=port,thermal_volume=index,
                        connectivity=connectivity(area,8.),gas_surface_connectivity=deepcopy(gas['surface_connectivity'])))
                    total+=area
        expected=inner[(axis+1)%3]*inner[(axis+2)%3]-(side**2 if axis==1 else 0.)
        if not math.isclose(total,expected,rel_tol=2e-10):raise ValueError('Packed wall contact area does not close: '+face)
        totals[face]=total
    meta['packed_cylinder_geometry']=dict(diameter_m=DIAMETER,height_m=HEIGHT,positions_measured=False,
        contact_count=len(neighbors),gas_partition='nearest-centre polygon minus actual cylinder',
        limitation='body compartment lumps disjoint tangent-cylinder pockets; no resolved momentum or holder',
        volume_before_m3=volume_before,volume_after_m3=sum(n['volume_m3'] for n in nodes.values()),
        mass_before_kg=mass_before,mass_after_kg=sum(n['initial_mass_kg'] for n in nodes.values()),
        wall_exposure_m2=totals,apertures=apertures,calibrated=False,validation_passed=False)
    return s
