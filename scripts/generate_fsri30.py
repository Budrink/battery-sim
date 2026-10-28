#!/usr/bin/env python3
"""Unfitted, explicit 30-component reconstruction. No UPV physics imports."""
import argparse
import math
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from fsri30_data import DATA, import_archive, load, save

FACES = ('-x', '+x', '-y', '+y', '-z', '+z')
DIAMETER, HEIGHT, MASS = .018, .065, .045
BOX = [.092, .172, .092]  # Table 3, not dimensions of the 270-component module.
WIDTH = DIAMETER*math.sqrt(math.pi)/2
CASE_T = .00018
HEATED = 'B10'  # Fig.17: red component, second staggered column, fourth row.


def connectivity(area, h, destruction=1.):
    g = area*h
    return dict(kind='directed_geometry', geometry_reference_logit=0.,
        background_conductance_w_per_k=.01*g, active_conductance_w_per_k=1.98*g,
        destroyed_background_conductance_w_per_k=g*destruction,
        destroyed_active_conductance_w_per_k=0., endpoint_damage_weight=0.)


def positions():
    """Digitized pattern, not CAD coordinates: 7+8+8+7, staggered columns."""
    result=[]
    for col, (count, offset) in enumerate(zip((7,8,8,7),(.5,0.,.5,1.))):
        for row in range(count):
            result.append([col*.0180, (row+offset)*.0193, 0.])
    a=np.array(result);a[:,:2]-=(a[:,:2].min(axis=0)+a[:,:2].max(axis=0))/2
    return a.tolist()


def build(duration=3600., heater_power=6.):
    if not 1. <= duration <= 5945. or not 0. <= heater_power <= 30.:
        raise ValueError('Diagnostic duration/power outside explicit range')
    ambient=298.15; pos=positions(); outer=[WIDTH,WIDTH,HEIGHT]
    inner=[x-2*CASE_T for x in outer];vcore=math.prod(inner)
    mcase=(math.prod(outer)-vcore)*7800.
    abs_mass=1.7-30*MASS
    lo,hi=0.,.02
    for _ in range(60):
        t=(lo+hi)/2
        if (math.prod(BOX)-math.prod(d-2*t for d in BOX))*1050. < abs_mass:lo=t
        else:hi=t
    box_t=(lo+hi)/2
    insize=[x-2*box_t for x in BOX]
    body_v=insize[0]*insize[1]*HEIGHT-30*math.prod(outer)
    cap_v=insize[0]*insize[1]*(insize[2]-HEIGHT)/2
    if min(body_v,cap_v) <= 0:raise ValueError('Solids do not fit inside reconstructed box')
    scene=dict(name='UL FSRI 30 x 18650 — UNFITTED development reconstruction',
        source='https://doi.org/10.5281/zenodo.7244850',
        metadata=dict(calibration_status='UNFITTED; not validation; all unknowns declared',
            geometry_status='Fig.17 staggered layout + Table3 dimensions; equal-volume box surrogate for each cylinder',
            observed=dict(components=30,soc=1.,cell_mass_kg=MASS,cell_dimensions_m=[DIAMETER,HEIGHT],
                          case_with_cells_mass_kg=1.7,case_dimensions_m=BOX,heater_ramp_k_per_s=.1),
            assumptions=['No scheduled ignition, measured-temperature forcing, time shifts or UPV fitted coefficients',
                'Four staggered columns follow drawing; millimetre coordinates are approximate, not CAD',
                'CUDA cylinder mesher unavailable: equivalent square preserves cylinder volume/height; heat capacity preserves total cell mass',
                'Steel casing thickness/material/contact, bulk k/cp and all chemical rates are unmeasured engineering priors',
                'Fresh 100% SOC maps to healthy charged Bc=1, initial degradation zero; SOC-to-inventory relation is a prior',
                'Effective R2 energy budget includes intrinsic plus pre-existing charge reserve; not nominal electrical energy alone',
                'Nine sensor-to-component locations unconfirmed: compare envelopes, not invented per-channel RMSE',
                'ABS is a passive thermal barrier; ABS pyrolysis/combustion and thermal radiation are NOT modeled',
                'Annular free spaces and upper/lower cavities are lumped per component; passage graph follows planar neighbour geometry',
                'Open exhaust hood approximated by finite 1 m3 oxygen-bearing reservoir, no room CFD or infinite oxygen source',
                'Six-watt heater is an open-loop estimate from 45J/K * 0.1K/s plus losses, NOT measured heater power',
                'Heater switched off after assumed 1800 s; actual small-test heater switch-off is unavailable',
                'Report Table2 gas volume213L refers to one thirty-component mock-up, NOT one 18650',
                'CSV integrated calorimeter HRR is about128MJ: test boundary/scale is unresolved; do not use it to calibrate this 30-cell chemistry',
                'Full report Fig17 widths3.1in and Table3 box92mm differ; Table3 external envelope retained']),
        simulation=dict(duration_s=duration,max_dt_s=.2,minimum_dt_s=.01,output_interval_s=5.,stability_factor=.4),
        cells=[],ports=[],interfaces=[],boundaries=[],shells=[],primary_failures=[],measurement_probes=[],
        failure_sample_times_s=sorted(set([0.,duration]+np.arange(5.,duration,5.).tolist())),
        thermal_path_trace=False,gas_film_trace=False,surface_probe_trace=False,
        process_history_selection=dict(gas_prefixes=['top-','body-','hood'],channel_prefixes=['rupture-']))
    nodes={};ports={}
    def solid(ident,dims,p,capacity,conductivity,role='passive',divisions=None):
        c=dict(id=ident,role=role,material_id='unidentified-18650' if role=='battery' else ident.split(':')[0],
            geometry=dict(kind='prismatic',dimensions_m=dims),pose=dict(position_m=p),
            thermal=dict(heat_capacity_j_per_k=capacity,initial_temperature_k=ambient,
                divisions=divisions or [1,1,1],conductivity_w_per_m_k=conductivity,
                bulk_connectivity=dict(kind='directed_geometry',background_factor=.01,active_factor=1.98,
                    destroyed_background_factor=1.,destroyed_active_factor=0.)),
            runaway=dict(energy_j=0.,duration_s=1.))
        scene['cells'].append(c);return c
    def port(ident,face,p,dims):
        axis='xyz'.index(face[1]);sign=1 if face[0]=='+' else -1
        key=ident+':'+face
        center=list(p);center[axis]+=sign*dims[axis]/2
        normal=[0.,0.,0.];normal[axis]=sign
        q=dict(id=key,cell_id=ident,kind='contact',surface=face,
            area_m2=dims[(axis+1)%3]*dims[(axis+2)%3],
            rectangle_m=[dims[(axis+1)%3],dims[(axis+2)%3]],centroid_m=center,normal=normal)
        scene['ports'].append(q);ports[key]=q;return key
    gas=dict(transport='split_local_equilibrium',mass_backend='cuda_network',network_integrator='averaged',
        averaged_pressure_iterations=512,averaged_payload_iterations=16,
        mixing_integrator='implicit_surfaces',coupled_gas_films=True,coupled_transport_interval_s=.05,
        free_gas_response_only=True,reference_temperature_k=298.15,cv_j_per_kg_k=1000.,specific_r_j_per_kg_k=287.,
        max_fraction_per_step=.02,max_temperature_step_k=2.,mixing_diffusivity_m2_s=2e-5,mixing_flow_fraction=.25,
        vapor_fuel_heat_j_kg=2e6,surface_connectivity=dict(reference_temperature_k=ambient,
            temperature_log_response=[[ambient,0.],[600.,math.log(3.)],[1200.,math.log(10.)]]),
        mixing_connectivity=connectivity(1.,1.),compartments=[],channels=[],thermal_contacts=[])
    scene['gas_network']=gas
    def node(ident,v,p,oxygen=True,binding=None):
        q=dict(id=ident,volume_m3=v,initial_mass_kg=101325.*v/(287.*ambient),
            initial_temperature_k=ambient,initial_oxidizer_mass_fraction=.232 if oxygen else 0.,
            reaction_rate_per_s=1.,reaction_reference_k=600.,reaction_activation_j_per_mol=80000.,
            oxidizer_heat_j_per_kg=13.1e6)
        if binding is not None:q['local_equilibrium']=binding
        gas['compartments'].append(q);nodes[ident]=p;return q
    def passage(ident,a,b,area,length,axis=0,internal=False):
        gas['channels'].append(dict(id=ident,kind='internal' if internal else 'passage',
            **{'from':a,'to':b},area_m2=area,mixing_length_m=length,mixing_axis=axis,
            discharge_coefficient=.7,opening_pressure_pa=0.))
    def film(comp,port_id,area,h=12.):
        gas['thermal_contacts'].append(dict(compartment=comp,port_id=port_id,
            connectivity=connectivity(area,h),gas_surface_connectivity=gas['surface_connectivity']))
    for j,p in enumerate(pos):
        ident=f'B{j:02d}'
        c=solid(ident,inner,p,MASS*1000.-mcase*500.,[1.,1.,20.],'battery',[3,3,3])
        c['conversion']=dict(chemistry='r1-r2-r3',mass_kg=.025,reactant_cp=1000.,activated_cp=1000.,residue_cp=800.,
            net_heat_j_per_kg=2.4e6,initial_precursor_fraction=0.,initial_charged_fraction=1.,
            degradation_rate_per_s=1e-6,degradation_reference_k=400.,degradation_activation_j_per_mol=80000.,
            reaction_rate_per_s=2e-5,reaction_reference_k=450.,reaction_activation_j_per_mol=160000.,
            precursor_yield=.42,gas_yield=.33,gas_yield_max=.33,gas_chemical_fraction=.65,
            gas_compartment=f'inside-{ident}-0',residue_conductivity=[1.,1.,1.],
            max_fraction_per_step=.05,max_temperature_step_k=2.,
            volatile_phase=dict(initial_mass_kg=.003,liquid_cp=1800.,density_kg_m3=1200.,latent_j_per_kg=5e5,
                boiling_temperature_k=400.,saturation_reference_pa=101325.,vapor_specific_r=287.,relaxation_per_s=10.,chemical_j_per_kg=2e6))
        scene['shells'].append(dict(contents_id=ident,material_id='steel-can-PROXY',thickness_m=CASE_T,
            density_kg_m3=7800.,specific_heat_j_kg_k=500.,conductivity_w_per_m_k=[15.,15.,15.],
            initial_temperature_k=ambient,contents_contact_h=500.,seam_contact_h=1000.,
            damage_onset_k=450.,damage_rate_per_s=.001,damage_scale_k=40.,destroyed_conductance_factor=.1))
        for band,v,z in [('body',body_v/30,0.),('top',cap_v/30,(HEIGHT+insize[2])/4),('base',cap_v/30,-(HEIGHT+insize[2])/4)]:
            node(f'{band}-{j:02d}',v,[*p[:2],z])
        passage(f'cap-up-{j}',f'body-{j:02d}',f'top-{j:02d}',WIDTH*.002,.01,2)
        passage(f'cap-down-{j}',f'body-{j:02d}',f'base-{j:02d}',WIDTH*.002,.01,2)
        for z in range(3):
            for y in range(3):
                for x in range(3):
                    i=x+3*y+9*z;name=f'inside-{ident}-{i}'
                    q=node(name,.5e-6/27,p,False,dict(cell_id=ident,thermal_volume=i,
                        reactant_density_kg_m3=2300.,residue_density_kg_m3=2000.))
                    if i==0:q['contents_id']=ident
                    for axis,coord in enumerate((x,y,z)):
                        if coord<2:passage(f'pore-{ident}-{i}-{axis}',name,f'inside-{ident}-{i+3**axis}',
                            .01*(inner[(axis+1)%3]/3)*(inner[(axis+2)%3]/3),inner[axis]/3,axis,True)
                    if z==2:
                        patch=x+3*y
                        gas['channels'].append(dict(id=f'rupture-{ident}-{patch}',kind='rupture',
                            **{'from':name,'to':f'top-{j:02d}'},port_id=f'{ident}:case:+z:outer',
                            shell_thermal_volume=patch,
                            opening_pressure_pa=6e5,strength_reference_k=ambient,strength_temperature_scale_k=1000.,
                            discharge_coefficient=.7,mixing_length_m=CASE_T,mixing_axis=2,
                            damage_opening=dict(mode='deterministic_aperture',area_rate_per_s=5.,pressure_exponent=1.,
                                max_damage_step=.005,fracture_energy_j_per_m2=500.,
                                **({'valve':dict(radius_m=.001,center_m=[0.,0.],pressure_factor=.2)} if patch==4 else {}))))
        # Outer ports added by shell builder. Matching panel dimensions are
        # known analytically; no bare-content/exterior shortcuts are created.
        for face in FACES:
            axis='xyz'.index(face[1]);panel=[CASE_T if k==axis else inner[k]+(2*CASE_T if k>axis else 0.) for k in range(3)]
            area=panel[(axis+1)%3]*panel[(axis+2)%3]
            comp=f'top-{j:02d}' if face=='+z' else f'base-{j:02d}' if face=='-z' else f'body-{j:02d}'
            film(comp,f'{ident}:case:{face}:outer',area,20.)
    neighbours=set()
    for tri in Delaunay(np.array(pos)[:,:2]).simplices:
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[0],tri[2])):
            a,b=sorted((int(a),int(b)));dist=math.dist(pos[a],pos[b])
            if dist<.029:neighbours.add((a,b))
    for a,b in sorted(neighbours):
        dist=math.dist(pos[a],pos[b]);gap=max(.0003,dist-DIAMETER)
        for band in ('body','top','base'):
            passage(f'free-{band}-{a}-{b}',f'{band}-{a:02d}',f'{band}-{b:02d}',
                gap*(HEIGHT if band=='body' else (insize[2]-HEIGHT)/2),dist)
    # Six nonoverlapping ABS panels: x owns corners, then y, then z.
    for axis in range(3):
        for sign in (-1,1):
            dims=[box_t if k==axis else BOX[k]-(2*box_t if k<axis else 0.) for k in range(3)]
            p=[0.,0.,0.];p[axis]=sign*(BOX[axis]-box_t)/2
            face=('+' if sign>0 else '-')+'xyz'[axis];ident='ABS:'+face
            c=solid(ident,dims,p,math.prod(dims)*1050.*1400.,[.18]*3)
            inside=port(ident,('-' if sign>0 else '+')+'xyz'[axis],p,dims)
            outside=port(ident,face,p,dims);area=ports[inside]['area_m2']
            band='top' if axis==2 and sign>0 else 'base' if axis==2 else 'body'
            for j in range(30):film(f'{band}-{j:02d}',inside,area/30,8.)
            scene['boundaries'].append(dict(id='room-'+face,port_id=outside,temperature_k=ambient,
                conductance_w_per_k=area*8.,connectivity=connectivity(area,8.)))
    node('hood',1.,[0.,0.,.5])
    for j in range(30):
        # Published mock-up vent routing unresolved at this scale. These are
        # explicit perforation paths through the enclosure, not solid-crossing
        # edges accidentally introduced by a spatial nearest-neighbour search.
        passage(f'exhaust-{j}',f'top-{j:02d}','hood',2e-5,.05,2)
    # A separate finite resistive heater: I2R is explicit external work, no
    # chemical charging of batteries and no imposed battery temperature.
    index=int(HEATED[1:]);pp=list(pos[index]);pp[0]-=WIDTH/2+.0001
    hd=[.0002,WIDTH,HEIGHT];heater=solid('heater',hd,pp,1.,[15.]*3)
    current=math.sqrt(heater_power)
    stop=min(duration,1800.);drive=[[0.,current]]
    if stop<duration:
        switch=stop+min(.001,(duration-stop)/2)
        drive.extend([[stop,current],[switch,0.],[duration,0.]])
    else:drive.append([duration,current])
    heater['electrical']=dict(current_profile_a=drive,voltage_profile_v=[[t,i]for t,i in drive],
        resistance_ohm=1.,min_resistance_ohm=1.,max_resistance_ohm=1.,reference_temperature_k=ambient,
        max_temperature_step_k=2.,terminal_axis='z')
    hp=port('heater','+x',pp,hd)
    scene['interfaces'].append(dict(id='heater--'+HEATED,port_a=hp,port_b=HEATED+':case:-x:outer',
        law=connectivity(WIDTH*HEIGHT,500.)))
    scene['metadata'].update(heated_component=HEATED,heater_power_w=heater_power,
        assumed_heater_cutoff_s=1800.,approximate_positions_m={f'B{j:02d}':p for j,p in enumerate(pos)},
        neighbour_pairs=[list(e)for e in sorted(neighbours)],
        geometry_budget=dict(cylinder_volume_m3=math.pi*(DIAMETER/2)**2*HEIGHT,
            surrogate_volume_m3=math.prod(outer),cell_total_capacity_j_k=MASS*1000.,
            core_capacity_j_k=MASS*1000.-mcase*500.,steel_mass_kg=mcase,
            abs_mass_kg=abs_mass,abs_thickness_m=box_t,gas_free_volume_m3=body_v+2*cap_v),
        comparison_scope='Development diagnostic; sensor locations and chemistry unidentified; no fitted accuracy claim')
    return scene


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--archive',type=Path);ap.add_argument('--report',type=Path)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--duration',type=float,default=3600.)
    args=ap.parse_args()
    if args.archive:import_archive(args.archive,args.report)
    data,manifest=load();scene=build(args.duration);scene['metadata']['data_manifest']=manifest
    save(args.output,scene);print('Generated',args.output,'batteries=30 gas=',len(scene['gas_network']['compartments']))


if __name__=='__main__':main()
