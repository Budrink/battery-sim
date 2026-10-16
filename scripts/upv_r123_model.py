"""Fresh physical stack + continuous R123 + Joule heating. No legacy profiles."""
import copy
import math
import numpy as np
import upv_r123_data as dataio

REFERENCE = 298.15
DEFAULT = dict(resistance_ohm=.008, resistance_temperature=-.001, resistance_damage=1.,
               r1_rate=1e-6, r1_activation=40000., overcharge_gain=2., r2_rate=300., r2_activation=120000.,
               energy_j=300000., gas_fraction=.7, gas_yield=.4, gas_yield_temperature=.0005, gas_rate=20.,
               precursor_yield=1., precursor_temperature=0., precursor_voltage=0.,
               iss_h=500., tim_h=2000., case_h=500., environment_h=8.,
               surface_iss=0., surface_tim=0., bulk_slope=0.,
               shell_strength_pa=1e6, opening_rate=.2, sink_g=10.)
# Bounds are declared before inspecting held-out observations. They are
# engineering search ranges, not measured properties or uncertainty intervals.
# gas_yield_temperature is linear in T around the R2 reference 450 K; at that
# T, bT=0.08 /K is E_eff = bT R T^2 = 135 kJ/mol. Negative slopes (more gas
# while cold than in runaway) are excluded; 0 is a temperature-independent yield.
BOUNDS = dict(resistance_ohm=(.001,.0245), resistance_temperature=(-.02,.02), resistance_damage=(0.,4.),
              r1_rate=(1e-8,1e-4), r1_activation=(20000.,60000.), overcharge_gain=(.05,20.),
              r2_rate=(.1,1e5), r2_activation=(80000.,180000.),
              energy_j=(150000.,600000.), gas_fraction=(.2,.9), gas_yield=(.1,.7),
              gas_yield_temperature=(0.,.08), gas_rate=(1.,200.),
              precursor_yield=(.01,1.), precursor_temperature=(0.,.08), precursor_voltage=(0.,2.),
              iss_h=(10.,10000.), tim_h=(10.,10000.), case_h=(20.,2000.), environment_h=(1.,30.),
              surface_iss=(-6.,6.), surface_tim=(-6.,6.), bulk_slope=(-6.,6.),
              shell_strength_pa=(1e5,1e7), opening_rate=(.01,5.), sink_g=(2.,50.))


def response(slope=0., contrast=0.):
    return dict(reference_temperature_k=REFERENCE, temperature_scale_k=100.,
                linear=[slope]*3, quadratic=[0.]*3, damage=[0.]*3,
                temperature_contrast=[contrast]*3, damage_contrast=[0.]*3)


def contact(area, h, destroyed=.02):
    return dict(kind='directed_geometry', geometry_reference_logit=0.,
                background_conductance_w_per_k=area*.01*h,
                active_conductance_w_per_k=area*1.98*h,
                endpoint_damage_weight=1., destroyed_geometry_reference_logit=0.,
                destroyed_background_conductance_w_per_k=area*h*destroyed,
                destroyed_active_conductance_w_per_k=0.)


def build(name, parameters=None, *, warmup=False, duration=None):
    p=dict(DEFAULT, **(parameters or {})); d, record=dataio.load(name); mark=dataio.event(d)
    duration=duration or (mark['previous_s']-20. if warmup else mark['time_s']+1800.)
    times=dataio.observation_times(d,mark,duration,warmup)
    ambient=float(d['T109_k'][0]); pressure=float(2.5*d['U114_v'][0]*1e5)
    drive, volts, electrical_meta=dataio.current_drive(name,d,mark)
    scene=dict(name='UPV R123 + I²R: '+name, source=dataio.SOURCE,
               metadata=dict(experiment=name, calibration_status='Fresh uncalibrated R123/Joule model',
                 canonical_sha256=record['canonical_sha256'], electrical=electrical_meta,
                 fixed_properties='Cell C=0.410*1007 J/K, k=(0.9,28.3,28.3); ISS k=.145, cp=770; TIM k=2, cp=1175; geometry/thickness fixed',
                 topology='Unconfirmed swapped-witnesses hypothesis retained explicitly: C3 left, C1 right; Table 5 material-side mapping',
                 assumptions=['No prescribed C2 temperature or prescribed ignition',
                   'One uninterrupted chemistry/pressure/thermal trajectory from ambient',
                   'Electrical resistance is unidentified; terminal U/I is NOT internal R',
                   'Casing laminate, gap .2 mm, wall mass and rupture properties are engineering proxies',
                   'Environment thermocouple is an observer, not an extra heat source',
                   'Inert R1/R2 condensed residue keeps intact conductivity; it is not an ash insulator']),
               simulation=dict(duration_s=duration,max_dt_s=.2,output_interval_s=5.,stability_factor=.4),
               cells=[],ports=[],interfaces=[],boundaries=[],shells=[],primary_failures=[],
               measurement_probes=[],failure_sample_times_s=times)
    layers=[('steel-L',.010,'steel',45.,7850.,500.),('ISS1',.002,'ISS',.145,200.,770.),
            ('C3',.011,'battery',.9,2291.,1007.),('TIM1',.002,'TIM',2.,3300.,1175.),
            ('Al1',.001,'aluminium',235.,2700.,900.),('ISS2',.002,'ISS',.145,200.,770.),
            ('C2',.011,'battery',.9,2291.,1007.),('TIM2',.002,'TIM',2.,3300.,1175.),
            ('Al2',.001,'aluminium',235.,2700.,900.),('ISS3',.002,'ISS',.145,200.,770.),
            ('C1',.011,'battery',.9,2291.,1007.),('TIM3',.002,'TIM',2.,3300.,1175.),
            ('Al3',.001,'aluminium',235.,2700.,900.),('ISS4',.002,'ISS',.145,200.,770.),
            ('steel-R',.010,'steel',45.,7850.,500.)]
    objects={};outer={};x=0.;width,height=.087,.187;t=.00015
    for ident,thickness,material,k,rho,cp in layers:
        x+=thickness/2;dims=[thickness,width,height];battery=material=='battery'
        initial=ambient
        if battery:
            channels=('103','104','105','106') if ident=='C2' else (('101','102') if ident=='C1' else ('107','108'))
            initial=float(np.mean([d['T'+c+'_k'][0] for c in channels]))
        outer[ident]=dims.copy()
        cap=.410*1007. if battery else math.prod(dims)*rho*cp
        if battery:
            inner=[q-2*t for q in dims];cap-=(math.prod(dims)-math.prod(inner))*1600.*1250.;dims=inner
        slope=p['surface_iss'] if material=='ISS' else p['surface_tim'] if material=='TIM' else 0.
        thermal=dict(heat_capacity_j_per_k=cap, initial_temperature_k=initial,
                     divisions=[3,3,3] if battery else [1,2,4],
                     conductivity_w_per_m_k=[k,28.3,28.3] if battery else [k]*3,
                     surface_connectivity=response(slope))
        if battery:
            thermal.update(bulk_response=response(p['bulk_slope']),bulk_connectivity=dict(
                kind='directed_geometry',background_factor=.01,active_factor=1.98,
                destroyed_background_factor=1.,destroyed_active_factor=0.))
        c=dict(id=ident,role='battery' if battery else 'passive',material_id=material,
               geometry=dict(kind='pouch' if battery else 'prismatic',dimensions_m=dims),
               pose=dict(position_m=[x,0.,0.]),thermal=thermal,runaway=dict(energy_j=0.,duration_s=1.))
        if battery:
            c['conversion']=dict(chemistry='r1-r2-r3',mass_kg=.2,reactant_cp=1007.,activated_cp=1007.,residue_cp=800.,
                net_heat_j_per_kg=p['energy_j']/.2,degradation_rate_per_s=p['r1_rate'],
                degradation_reference_k=400.,degradation_activation_j_per_mol=p['r1_activation'],
                reaction_rate_per_s=p['r2_rate'],reaction_reference_k=450.,reaction_activation_j_per_mol=p['r2_activation'],
                initial_precursor_fraction=0.,overcharge_conversion_per_charge=p['overcharge_gain'],gas_yield=p['gas_yield'],gas_yield_max=.8,
                gas_yield_reference_k=450.,gas_yield_temperature_per_k=p['gas_yield_temperature'],gas_yield_density_exponent=.2,
                precursor_yield=p['precursor_yield'],precursor_yield_temperature_per_k=p['precursor_temperature'],
                precursor_yield_voltage_per_v=p['precursor_voltage'],precursor_temperature_ref_k=REFERENCE,
                precursor_voltage_ref_v=3.65,
                gas_chemical_fraction=p['gas_fraction'],gas_compartment='inside-'+ident+'-0',
                # S and R2 condensed C are chemically inert, not ash: keep intact k.
                residue_conductivity=[1.,1.,1.],max_fraction_per_step=.02,max_temperature_step_k=1.)
            scene['shells'].append(dict(contents_id=ident,material_id='unidentified-pouch-laminate-PROXY',thickness_m=t,
                density_kg_m3=1600.,specific_heat_j_kg_k=1250.,conductivity_w_per_m_k=[.3,50.,50.],
                conductivity_frame='normal_tangential',initial_temperature_k=initial,contents_contact_h=p['case_h'],
                seam_contact_h=500.,damage_onset_k=425.,damage_rate_per_s=.1,damage_scale_k=25.,destroyed_conductance_factor=.02))
        if ident=='C2':
            c['electrical']=dict(current_profile_a=drive,voltage_profile_v=volts,resistance_ohm=p['resistance_ohm'],
                temperature_log_slope_per_k=p['resistance_temperature'],damage_log_slope=p['resistance_damage'],
                reference_temperature_k=REFERENCE,min_resistance_ohm=.0001,max_resistance_ohm=.15,max_temperature_step_k=1.,
                overcharge_fraction=1.,nominal_capacity_coulomb=35.*3600.)
        objects[ident]=c;scene['cells'].append(c);x+=thickness/2

    ports={}
    def port(ident,face):
        key=ident+':'+face
        if key in ports:return key
        c=objects[ident];axis='xyz'.index(face[1]);sign=1 if face[0]=='+' else -1
        dims=outer.get(ident,c['geometry']['dimensions_m']);pos=c['pose']['position_m'].copy();pos[axis]+=sign*dims[axis]/2
        target=ident+':case:'+face if c['role']=='battery' else ident
        panel=[t if a==axis else dims[a]-(2*t if a<axis else 0.) for a in range(3)] if c['role']=='battery' else dims
        normal=[0.,0.,0.];normal[axis]=sign
        v=dict(id=key,cell_id=target,kind='contact',surface=face,area_m2=panel[(axis+1)%3]*panel[(axis+2)%3],
               rectangle_m=[panel[(axis+1)%3],panel[(axis+2)%3]],centroid_m=pos,normal=normal)
        ports[key]=v;scene['ports'].append(v);return key
    def edge(a,af,b,bf,h):
        pa,pb=port(a,af),port(b,bf);area=min(ports[pa]['area_m2'],ports[pb]['area_m2'])
        ident=a+'--'+b
        scene['interfaces'].append(dict(id=ident,port_a=pa,port_b=pb,law=contact(area,h)))
        return ident
    for a,b in zip(layers,layers[1:]):
        material=a[2] if a[2] in ('ISS','TIM') else b[2]
        edge(a[0],'+x',b[0],'-x',p['iss_h'] if material=='ISS' else p['tim_h'] if material=='TIM' else 2000.)
    mapping=(('101','C1','ISS'),('102','C1','TIM'),('107','C3','TIM'),('108','C3','ISS'),
             ('C2_TIM','C2','TIM'),('C2_ISS','C2','ISS'))
    for channel,cell,mat in mapping:
        matches=[]
        for e in scene['interfaces']:
            a,b=ports[e['port_a']],ports[e['port_b']]
            for own,other in ((a,b),(b,a)):
                if own['cell_id'].startswith(cell+':case:') and objects[other['cell_id']]['material_id']==mat:
                    matches.append(dict(id=channel,cell_id=own['cell_id'],interface_id=e['id']))
        if len(matches)!=1:raise ValueError('Unambiguous physical sensor contact required: '+channel)
        scene['measurement_probes']+=matches

    # Every exterior face is available to gas contacts; solid layer faces do
    # not acquire an artificial shortcut through an intact pad.
    for ident in objects:
        for face in ('-x','+x','-y','+y','-z','+z'):port(ident,face)
    gas=dict(transport='split_local_equilibrium',mass_backend='cuda_network',network_pressure_fraction=1.,
             reference_temperature_k=REFERENCE,cv_j_per_kg_k=1000.,specific_r_j_per_kg_k=287.,
             max_fraction_per_step=.02,max_temperature_step_k=1.,compartments=[],channels=[],thermal_contacts=[])
    scene['gas_network']=gas
    def node(ident,volume,temperature,binding=None,oxygen=True):
        v=dict(id=ident,volume_m3=volume,initial_mass_kg=pressure*volume/(287.*temperature),initial_temperature_k=temperature)
        if oxygen:v.update(initial_oxidizer_mass_fraction=.232,reaction_rate_per_s=p['gas_rate'],oxidizer_heat_j_per_kg=13.1e6)
        if binding:v['local_equilibrium']=binding
        gas['compartments'].append(v);return v
    def path(ident,a,b,area,anchor=None):
        q=dict(id=ident,kind='vent' if anchor else 'internal',**{'from':a,'to':b},area_m2=area,
               discharge_coefficient=.7,opening_pressure_pa=0.)
        if anchor:q['port_id']=anchor
        gas['channels'].append(q)
    for ident in ('C1','C2','C3'):
        c=objects[ident];dims=c['geometry']['dimensions_m'];sizes=[v/3 for v in dims]
        for i in range(27):
            n=node(f'inside-{ident}-{i}',5e-6/27,c['thermal']['initial_temperature_k'],
                   dict(cell_id=ident,thermal_volume=i,reactant_density_kg_m3=2000.,residue_density_kg_m3=2000.),False)
            if i==0:n['contents_id']=ident
        for z in range(3):
            for y in range(3):
                for x in range(3):
                    ijk=[x,y,z];i=x+3*(y+3*z)
                    for axis in range(3):
                        if ijk[axis]==2:continue
                        j=i+3**axis
                        path(f'pore-{ident}-{i}-{j}',f'inside-{ident}-{i}',f'inside-{ident}-{j}',sizes[(axis+1)%3]*sizes[(axis+2)%3]*.03)
    # Four connected peripheral cavities, with no path through the plates.
    for face in ('-y','+y','-z','+z'):
        n=node('near-'+face,.0002,ambient)
        path('near-to-chamber-'+face,n['id'],'chamber',ports['C2:'+face]['area_m2'],'C2:'+face)
    gap=.0002;pad_gap={}
    for ident,c in objects.items():
        if c['material_id'] not in ('ISS','TIM') or ident=='ISS4':continue
        dims=c['geometry']['dimensions_m'];area=dims[1]*dims[2]
        inlet='gap-'+ident+'-inlet';node(inlet,area*gap*.1,ambient);pad_gap[ident]=inlet
        for i in range(8):
            n='gap-'+ident+'-'+str(i)
            node(n,area*gap*.9/8,c['thermal']['initial_temperature_k'],dict(cell_id=ident,thermal_volume=i))
            face='+x' if c['material_id']=='ISS' else '-x'
            path('feed-'+n,inlet,n,area/8,ident+':'+face)
            y,z=i%2,i//2
            for sign,bound in (('-',0),('+',1)):
                if y==bound:path(n+'-'+sign+'y',n,'near-'+sign+'y',gap*dims[2]/4,ident+':'+sign+'y')
            for sign,bound in (('-',0),('+',3)):
                if z==bound:path(n+'-'+sign+'z',n,'near-'+sign+'z',gap*dims[1]/2,ident+':'+sign+'z')
            if y<1:path(n+'-y',n,'gap-'+ident+'-'+str(i+1),gap*dims[2]/4)
            if z<3:path(n+'-z',n,'gap-'+ident+'-'+str(i+2),gap*dims[1]/2)
    for ident in ('C1','C2','C3'):
        c=objects[ident];temp=c['thermal']['initial_temperature_k'];layer_index=next(i for i,v in enumerate(layers) if v[0]==ident)
        for axis in range(3):
            for sign in ('-','+'):
                face=sign+'xyz'[axis];target='near-'+face if axis else pad_gap[layers[layer_index+(-1 if sign=='-' else 1)][0]]
                for v in range(3):
                    for u in range(3):
                        ijk=[0,0,0];ijk[axis]=0 if sign=='-' else 2;ijk[(axis+1)%3]=u;ijk[(axis+2)%3]=v
                        i=ijk[0]+3*(ijk[1]+3*ijk[2])
                        # Panel local ordering excludes its singleton normal axis.
                        pane=ijk.copy();pane[axis]=0;nd=[3,3,3];nd[axis]=1
                        local=pane[0]+nd[0]*(pane[1]+nd[1]*pane[2])
                        gas['channels'].append(dict(id=f'rupture-{ident}-{face}-{local}',kind='rupture',
                            **{'from':f'inside-{ident}-{i}','to':target},port_id=f'{ident}:case:{face}:outer',
                            shell_thermal_volume=local,opening_pressure_pa=p['shell_strength_pa'],
                            strength_reference_k=REFERENCE,strength_temperature_scale_k=300.,discharge_coefficient=.7,
                            damage_opening=dict(mode='deterministic_aperture',area_rate_per_s=p['opening_rate'],pressure_exponent=1.,
                                max_damage_step=.02,seam_width_m=.001,seam_strength_factor=.2,fracture_energy_j_per_m2=2000.)))
        for face in ('-y','+y','-z','+z'):
            gas['thermal_contacts'].append(dict(compartment='near-'+face,port_id=ident+':'+face,h_w_per_m2_k=p['environment_h']))
    # Explicit thermal reservoir wall; its 10 kg mass is an unmeasured proxy.
    wall=dict(id='chamber-wall',role='passive',material_id='steel-PROXY',
        geometry=dict(kind='prismatic',dimensions_m=[.002,1.,1.]),pose=dict(position_m=[-1.,0.,0.]),
        thermal=dict(heat_capacity_j_per_k=5000.,initial_temperature_k=ambient,divisions=[1,1,1],conductivity_w_per_m_k=[45.]*3),
        runaway=dict(energy_j=0.,duration_s=1.))
    scene['cells'].append(wall);objects['chamber-wall']=wall
    for face in ('-x','+x'):port('chamber-wall',face)
    gas['thermal_contacts'].append(dict(compartment='chamber',port_id='chamber-wall:+x',h_w_per_m2_k=p['environment_h']))
    scene['boundaries'].append(dict(id='room',port_id='chamber-wall:-x',temperature_k=ambient,conductance_w_per_k=5.))
    external_volume=sum(n['volume_m3'] for n in gas['compartments'] if not n['id'].startswith('inside-'))
    node('chamber',.120-external_volume,ambient)
    if name=='active-cooling':attach_cooling(scene,objects,ports,port,p,ambient)
    scene['metadata'].update(parameters=p,gas_nodes=len(gas['compartments']),event_marker_diagnostic=mark,
                             chamber_volume_m3=.120,source_temperature_is_observation=True)
    return scene,dict(data=d,event=mark,observed=dataio.observed(d,times))


def attach_cooling(scene,objects,ports,port,p,ambient):
    # Aluminium wall + constant water sink, NOT a prescribed measured ch110.
    left=objects['steel-L'];right=objects['steel-R'];lo=left['pose']['position_m'][0]-.005
    hi=right['pose']['position_m'][0]+.005;dims=[hi-lo,.087,.013];pos=[(lo+hi)/2,0.,-.187/2-.013/2]
    c=dict(id='cooling-plate',role='passive',material_id='aluminium',geometry=dict(kind='prismatic',dimensions_m=dims),
           pose=dict(position_m=pos),thermal=dict(heat_capacity_j_per_k=math.prod(dims)*2700*900,
           initial_temperature_k=ambient,divisions=[7,3,1],conductivity_w_per_m_k=[235.]*3,
           surface_connectivity=response()),runaway=dict(energy_j=0.,duration_s=1.))
    scene['cells'].append(c);objects[c['id']]=c
    for ident in ('Al1','Al2','Al3'):
        a=ports[port(ident,'-z')];key='cooling-plate:'+ident
        scene['ports'].append(dict(id=key,cell_id='cooling-plate',kind='contact',surface='+z',area_m2=a['area_m2'],
            rectangle_m=a['rectangle_m'],centroid_m=a['centroid_m'],normal=[0.,0.,1.]))
        scene['interfaces'].append(dict(id=ident+'--cooling',port_a=a['id'],port_b=key,law=contact(a['area_m2'],10000.,1.)))
    sink=port('cooling-plate','-z');scene['boundaries'].append(dict(id='water',port_id=sink,temperature_k=298.15,conductance_w_per_k=p['sink_g']))
    # Bottom cavity drains around the edges; no route through the metal plate.
    gas=scene['gas_network'];gas['channels']=[q for q in gas['channels'] if q['id']!='near-to-chamber--z']
    for face in ('-y','+y'):
        gas['channels'].append(dict(id='bottom-around-'+face,kind='vent',**{'from':'near--z','to':'near-'+face},
            port_id='C2:'+face,area_m2=.0002,discharge_coefficient=.7,opening_pressure_pa=0.))
