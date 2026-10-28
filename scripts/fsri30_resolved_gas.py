"""FSRI fixture adapter: resolved external gas, local ABS films, heater protocol.

Imported material/kinetic parameters are immutable. The stirring coefficient
is an explicit reusable constitutive PRIOR, not a learned geometry correction.
"""
from copy import deepcopy
import math
import numpy as np

from free_gas_mesh import (exterior_grid,rectangle_overlap,rectangle_intersection,
                           face_tiles,add_face_port)
from fsri30_reconstruction import source_geometry,transfer_r2,clip,polygon_area
from generate_fsri30 import BOX,HEIGHT,connectivity


def intersection_area(poly,rect):
    p=np.asarray(poly)
    if (p[:,0].max()<=rect[0] or p[:,0].min()>=rect[1] or
        p[:,1].max()<=rect[2] or p[:,1].min()>=rect[3]):return 0.
    try:
        for normal,offset in (([-1.,0.],-rect[0]),([1.,0.],rect[1]),([0.,-1.],-rect[2]),([0.,1.],rect[3])):
            p=clip(p,np.array(normal),offset)
        return polygon_area(p)
    except ValueError:return 0.


def build_resolved(base,spacing=.04,*,stirring=.05,halo=.04,controlled_heater=True,
                   stirring_length=.02,pressure_patterns=False,vent_spacing=None,wall_spacing=None):
    if 'geometry_reconstruction' in base['metadata']:
        scene=deepcopy(base)
    else:scene=source_geometry(base)
    if 'r2_transfer' not in scene['metadata']:scene=transfer_r2(scene)
    gas=scene['gas_network'];meta=scene['metadata'];geom=meta['geometry_reconstruction']
    if 'air_mesh' in meta:raise ValueError('Air already resolved; build each level from the same source')
    outside=[n for n in gas['compartments'] if n['id']=='hood' or n['id'].startswith('plume-')]
    if not outside:raise ValueError('Missing old external gas inventory')
    reference=deepcopy(outside[0]);volume=sum(n['volume_m3'] for n in outside)
    density=sum(n['initial_mass_kg'] for n in outside)/volume
    oxygen=sum(n['initial_mass_kg']*n['initial_oxidizer_mass_fraction'] for n in outside)/(density*volume)
    if any(n['initial_temperature_k']!=reference['initial_temperature_k'] for n in outside):
        raise ValueError('Only a fresh uniform initial atmosphere can be spatially remeshed')
    voxels,faces=exterior_grid(BOX,spacing,halo,volume,vent_spacing=vent_spacing)
    wall_spacing=spacing if wall_spacing is None else wall_spacing
    if not 0.<wall_spacing<=.1:raise ValueError('Invalid ABS mesh spacing')
    old_ids={n['id'] for n in outside}
    gas['compartments']=[n for n in gas['compartments'] if n['id'] not in old_ids]
    layouts={}
    for voxel in voxels.values():
        node=deepcopy(reference);node.update(id=voxel.name,volume_m3=voxel.volume,
            initial_mass_kg=voxel.volume*density,initial_oxidizer_mass_fraction=oxygen)
        gas['compartments'].append(node)
        layouts[voxel.name]=dict(low_m=voxel.low.tolist(),high_m=voxel.high.tolist())
    gas['channels']=[e for e in gas['channels'] if e['from'] not in old_ids and e['to'] not in old_ids]
    def passage(ident,a,b,area,length,axis,factor=1.):
        return dict(id=ident,kind='passage',**{'from':a,'to':b},area_m2=float(area),
            mixing_length_m=float(length),mixing_axis=axis,discharge_coefficient=.7,
            hydraulic_resistance_factor=float(factor),opening_pressure_pa=0.)
    # Distributed free-space quadratic resistance is additive by LENGTH.
    # h-refinement keeps physical A and the total path resistance distinct.
    resistance_reference=.05
    for a,b,area,length,axis in faces:
        gas['channels'].append(passage(f'air-path-{a}-{axis}',a,b,area,length,axis,length/resistance_reference))
    half=np.array(BOX)/2;side=math.sqrt(geom['unmeasured_total_aperture_area_m2']/2)
    hole=[-side/2,side/2,-side/2,side/2]
    def at_surface(voxel,axis,sign):
        return abs((voxel.low[axis] if sign>0 else voxel.high[axis])-sign*half[axis])<1e-11
    positions=meta['approximate_positions_m']
    for aperture in geom['central_front_back_apertures']:
        face=aperture['face'];sign=1 if face[0]=='+' else -1;j=aperture['component']
        wall=next(w for w in geom['wall_faces'] if w['face']==face and w['component']==j)
        xs=np.array(wall['endpoints'])[:,0]
        # Face normal y -> ordered rectangle coordinates z,x.
        region=rectangle_intersection(hole,[-side/2,side/2,float(min(xs)),float(max(xs))])
        total=0.
        for voxel in voxels.values():
            if not at_surface(voxel,1,sign):continue
            rect=[voxel.low[2],voxel.high[2],voxel.low[0],voxel.high[0]]
            area=rectangle_overlap(region,rect)
            if area<=1e-16:continue
            distance=np.linalg.norm(voxel.center-np.array(positions[f'B{j:02d}']))
            gas['channels'].append(passage(f'vent-{face}-{j}-{voxel.name}',f'body-{j:02d}',voxel.name,area,distance,1))
            total+=area
        if not math.isclose(total,aperture['area_m2'],rel_tol=1e-9):raise ValueError('Air grid lost aperture area')
    gas['thermal_contacts']=[c for c in gas['thermal_contacts'] if not c['port_id'].startswith('ABS:')]
    scene['boundaries']=[b for b in scene['boundaries'] if not b['id'].startswith('room-')]
    material_capacity_before=sum(c['thermal']['heat_capacity_j_per_k'] for c in scene['cells'])
    thickness=meta['geometry_budget']['abs_thickness_m'];inner=np.array(BOX)-2*thickness
    wall_lookup={(w['face'],w['component']):w for w in geom['wall_faces']}
    expected_exposure={};actual_exposure={}
    for cell in scene['cells']:
        if not cell['id'].startswith('ABS:'):continue
        face=cell['id'].split(':')[1];axis='xyz'.index(face[1]);sign=1 if face[0]=='+' else -1
        dims=cell['geometry']['dimensions_m']
        cell['thermal']['divisions']=[1 if k==axis else max(1,math.ceil(dims[k]/wall_spacing)) for k in range(3)]
        u,v=(axis+1)%3,(axis+2)%3
        def film(comp,port,index,area,h):
            if area<=1e-15:return
            gas['thermal_contacts'].append(dict(compartment=comp,port_id=port,thermal_volume=index,
                connectivity=connectivity(area,h),gas_surface_connectivity=deepcopy(gas['surface_connectivity'])))
        # Outside films replace the old infinite-ambient wall boundary. Include
        # the same half-panel resistance in the reference conductance.
        h_outer=1/(1/8.+thickness/(2*.18))
        exterior_area=0.
        boundary_voxels=[n for n in voxels.values() if at_surface(n,axis,sign)]
        for index,rect,point in face_tiles(cell,face):
            port=None
            for voxel in boundary_voxels:
                patch=[voxel.low[u],voxel.high[u],voxel.low[v],voxel.high[v]]
                overlap=rectangle_intersection(rect,patch);area=rectangle_overlap(rect,patch)
                if axis==1:area-=rectangle_overlap(overlap,hole)
                if area<=1e-15:continue
                if port is None:port=add_face_port(scene,cell,face,index,rect,point,'outside')
                film(voxel.name,port,index,area,h_outer);exterior_area+=area
        # Inside surfaces: only the local polygon/boundary strip touches its
        # ABS voxel. A full-panel port must not smear a local gas film over it.
        inside=('-' if sign>0 else '+')+face[1];interior_area=0.
        for index,rect,point in face_tiles(cell,inside):
            port=None
            for j in range(30):
                if axis==2:
                    area=intersection_area(geom['polygon_xy_m'][j],rect)
                    bands=[('top' if sign>0 else 'base',area)]
                else:
                    wall=wall_lookup.get((face,j))
                    if wall is None:continue
                    other=1-axis;ends=np.array(wall['endpoints'])[:,other]
                    bands=[]
                    for band,zlo,zhi in (('body',-HEIGHT/2,HEIGHT/2),('top',HEIGHT/2,inner[2]/2),('base',-inner[2]/2,-HEIGHT/2)):
                        region=[float(min(ends)),float(max(ends)),zlo,zhi] if axis==0 else [zlo,zhi,float(min(ends)),float(max(ends))]
                        overlap=rectangle_intersection(rect,region);area=rectangle_overlap(rect,region)
                        if axis==1 and band=='body':area-=rectangle_overlap(overlap,hole)
                        bands.append((band,area))
                for band,area in bands:
                    if area<=1e-15:continue
                    if port is None:port=add_face_port(scene,cell,inside,index,rect,point,'inside')
                    film(f'{band}-{j:02d}',port,index,area,8.);interior_area+=area
        expected_inside=inner[(axis+1)%3]*inner[(axis+2)%3]-(side*side if axis==1 else 0.)
        expected_outside=dims[(axis+1)%3]*dims[(axis+2)%3]-(side*side if axis==1 else 0.)
        expected_exposure[face]=dict(inside_m2=float(expected_inside),outside_m2=float(expected_outside))
        actual_exposure[face]=dict(inside_m2=interior_area,outside_m2=exterior_area)
        if not math.isclose(interior_area,expected_inside,rel_tol=2e-10) or not math.isclose(exterior_area,expected_outside,rel_tol=2e-10):
            raise ValueError('Local ABS film areas do not close: '+face)
    if material_capacity_before!=sum(c['thermal']['heat_capacity_j_per_k'] for c in scene['cells']):raise ValueError('Meshing changed solid capacity')
    gas['thermal_stirring_coefficient']=float(stirring);gas['mixing_length_cap_m']=.05
    gas['thermal_stirring_length_m']=float(stirring_length)
    gas['pressure_pattern_predictor']=bool(pressure_patterns)
    gas['averaged_pressure_iterations']=2048
    gas['averaged_payload_iterations']=64
    if controlled_heater:
        heater=next(c for c in scene['cells'] if c['id']=='heater')
        heater['electrical'].pop('current_profile_a',None);heater['electrical'].pop('voltage_profile_v',None)
        heater['electrical']['current_a']=0.
        scene['heating_protocols']=[dict(id='fsri-film-ramp',heater_id='heater',sensor_id='B10:case:-x',
            initial_temperature_k=298.15,ramp_k_per_s=.1,feedforward_w=4.5,proportional_w_per_k=1.,maximum_power_w=30.,
            cutoff_temperature_k=773.15,cutoff_rate_k_per_s=10.)]
    meta.update(air_mesh=dict(spacing_m=spacing,halo_m=halo,external_volume_m3=volume,
        vent_spacing_m=vent_spacing,vent_refinement_half_width_m=.025 if vent_spacing is not None else None,
        vent_refinement_alignment='nested-coarse-cells-v1' if vent_spacing is not None else None,
        abs_spacing_m=wall_spacing,
        external_nodes=len(voxels),external_channels=len(faces),resistance_reference_length_m=resistance_reference,
        distributed_resistance_measured=False,boundary='finite conservative atmosphere; no infinite oxygen or pressure clamp',
        expected_abs_exposure=expected_exposure,actual_abs_exposure=actual_exposure),
        gas_volume_geometry=layouts,visualization_temperature_threshold_above_ambient_k=10.,
        thermal_stirring_prior=dict(coefficient=stirring,mixing_length_cap_m=.05,physical_length_m=stirring_length,
            law='fixed-physical-length-resolved-gradient-v2',measured=False,
            scope='free gas only; unresolved thermal circulation; NOT a fixture route or calibrated material coefficient'),
        heater_protocol=dict(controlled=controlled_heater,ramp_measured=True,controller_gains_measured=False,
            sensor_location_prior='reference-capacity mean of heated can -x panel; exact TC position unknown'),
        geometry_status='3D spatially resolved free gas and local ABS contact/thermal mesh',
        calibrated=False,validation_passed=False)
    scene['name']='FSRI resolved gas + UPV R2 / h='+str(spacing)
    # JSON histories stay small; dense all-volume fields are exported separately
    # as binary by the observer, never used by the CUDA physics.
    scene['process_history_selection']['gas_prefixes']=['body-','top-','base-']
    scene['process_history_selection']['channel_prefixes']=['rupture-','vent-']
    scene['gas_history_groups']={'outside':['air-'],'module':['body-','top-','base-']}
    return scene
