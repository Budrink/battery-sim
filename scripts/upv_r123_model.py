"""Fresh physical stack + continuous R123 + Joule heating. No legacy profiles."""
import copy
import math
import numpy as np
import upv_r123_data as dataio
import upv_gas_geometry as gas_geometry_model
from upv_connectivity import contact

REFERENCE = 298.15
DEFAULT = dict(resistance_ohm=.008, resistance_temperature=-.001, resistance_charge=0., product_conductivity_ratio=.2,
               r1_rate=1e-6, r1_activation=40000., charge_formation_gain=2., r2_rate=300., r2_activation=120000.,
               energy_j=300000., gas_fraction=.7, gas_yield=.4, gas_yield_temperature=.0005, gas_rate=20.,
               gas_reference_k=600., gas_activation=80000.,
               precursor_yield=1., precursor_temperature=0.,
               iss_h=500., tim_h=2000., battery_iss_factor=1., battery_tim_factor=1.,
               case_h=500., environment_h=8., chamber_gas_h=8.,
               surface_iss=0., surface_tim=0., bulk_slope=0.,case_warm_response=0.,case_hot_response=0.,
               shell_strength_pa=1e6, opening_rate=.2, sink_g=10.)
# Bounds are declared before inspecting held-out observations. They are
# engineering search ranges, not measured properties or uncertainty intervals.
# gas_yield_temperature is linear in T around the R2 reference 450 K; at that
# T, bT=0.08 /K is E_eff = bT R T^2 = 135 kJ/mol. Negative slopes (more gas
# while cold than in runaway) are excluded; 0 is a temperature-independent yield.
BOUNDS = dict(resistance_ohm=(.001,.0195), resistance_temperature=(-.02,.02), resistance_charge=(0.,20.), product_conductivity_ratio=(.01,1.),
              r1_rate=(1e-8,1e-4), r1_activation=(20000.,60000.), charge_formation_gain=(.05,20.),
              r2_rate=(.1,1e5), r2_activation=(80000.,180000.),
              energy_j=(150000.,600000.), gas_fraction=(.2,.9), gas_yield=(.1,.7),
              gas_yield_temperature=(0.,.08), gas_rate=(1.,200.), gas_activation=(40000.,160000.),
              precursor_yield=(.01,1.), precursor_temperature=(0.,.08),
              iss_h=(10.,10000.), tim_h=(10.,10000.), battery_iss_factor=(.1,10.), battery_tim_factor=(.1,10.),
              case_h=(20.,2000.), environment_h=(1.,30.), chamber_gas_h=(1.,500.),
              surface_iss=(-6.,6.), surface_tim=(-6.,6.), bulk_slope=(-6.,6.),
              case_warm_response=(-6.,6.),case_hot_response=(-6.,6.),
              shell_strength_pa=(1e5,1e7), opening_rate=(.01,5.), sink_g=(2.,50.))


def response(slope=0., contrast=0.):
    return dict(reference_temperature_k=REFERENCE, temperature_scale_k=100.,
                linear=[slope]*3, quadratic=[0.]*3, damage=[0.]*3,
                temperature_contrast=[contrast]*3, damage_contrast=[0.]*3)


def ambient_boundary(ident, port, temperature, g):
    # Same directed S-link as solid contacts. conductance_w_per_k remains the
    # parser scale; the law, not a Constant link, carries the heat.
    return dict(id=ident, port_id=port, temperature_k=temperature, conductance_w_per_k=g,
                connectivity=contact(1., g), remote_surface_connectivity=response())


def casing_response(p):
    # Minimal non-polynomial shape: reference fixed to zero; two fitted state
    # values. This changes contact geometry, NEVER measured material C or k.
    return dict(reference_temperature_k=REFERENCE,temperature_scale_k=100.,
                temperature_log_response=[[REFERENCE,0.],[330.,p['case_warm_response']],
                                          [390.,p['case_hot_response']]])


def pad_contact_h(material_a, material_b, p):
    """Independent casing/pad and metal/pad contacts; never change bulk k.

    Unit factors preserve every prior physical scenario exactly. A factor
    applies to the material pair on ALL batteries, not one thermocouple.
    """
    pair={material_a,material_b}
    if 'ISS' in pair:
        base,factor=p['iss_h'],p['battery_iss_factor'] if 'battery' in pair else 1.
    elif 'TIM' in pair:
        base,factor=p['tim_h'],p['battery_tim_factor'] if 'battery' in pair else 1.
    else:
        return 2000.
    if not math.isfinite(factor) or not .1<=factor<=10.:
        raise ValueError('Casing/pad contact factor must be in [0.1, 10]')
    h=base*factor
    if not math.isfinite(h) or not 10.<=h<=10000.:
        raise ValueError('Effective pad-contact h must be in [10, 10000] W/(m2 K)')
    return h


def build(name, parameters=None, *, warmup=False, duration=None, dense_after=None, dense_span=10., gas_extensions=None, gas_geometry=None, diagnostic_removed_pads=None):
    if parameters and 'overcharge_pay' in parameters:
        raise ValueError('Retired overcharge_pay: charge formation is fully paid; thermal R1 is separate')
    if parameters and 'resistance_damage' in parameters:
        raise ValueError('Old R(T,D) calibration is invalid for Kirchhoff/product-mixture electrical conduction')
    unknown=set(parameters or {})-set(DEFAULT)
    if unknown:raise ValueError('Unsupported calibration fields; migrate the full profile explicitly: '+', '.join(sorted(unknown)))
    p=dict(DEFAULT, **(parameters or {})); d, record=dataio.load(name); mark=dataio.event(d)
    geometry=gas_geometry_model.configuration(gas_geometry)
    duration=duration or (mark['previous_s']-20. if warmup else mark['time_s']+1800.)
    times=dataio.observation_times(d,mark,duration,warmup,dense_after=dense_after,dense_span=dense_span)
    ambient=float(d['T109_k'][0]); pressure=float(2.5*d['U114_v'][0]*1e5)
    drive, volts, electrical_meta=dataio.current_drive(name,d,mark)
    scene=dict(name='UPV R123 + I²R: '+name, source=dataio.SOURCE,
               metadata=dict(experiment=name, calibration_status='Fresh uncalibrated R123/Joule model',
                 canonical_sha256=record['canonical_sha256'], electrical=electrical_meta,
                 fixed_properties='Cell C=0.410*1007 J/K, k=(0.9,28.3,28.3); ISS k=.145, cp=770; TIM k=2, cp=1175; layer thickness fixed; pad footprint follows explicit geometry controls',
                 topology='Unconfirmed swapped-witnesses hypothesis retained explicitly: C3 left, C1 right; Table 5 material-side mapping',
                 assumptions=['No prescribed C2 temperature or prescribed ignition',
                   'One uninterrupted chemistry/pressure/thermal trajectory from ambient',
                   'Electrical resistance is unidentified; terminal U/I is NOT internal R',
                   'Casing laminate, external void/face slits, wall mass and rupture properties are engineering proxies',
                   'Every finite thermal contact, including gas surfaces, uses the shared directed S-law',
                   'Casing/pad and metal/pad references are independent shared material-pair contacts; unit factors retain prior references',
                   'Environment thermocouple is an observer, not an extra heat source',
                   'R1/R2 condensed residue retains intact THERMAL conductivity; one reduced coefficient controls ELECTRICAL conductivity',
                   'Charging polarization follows accepted unconsumed Bc/(A+Bc), not passed-charge time; U·I is not heat',
                   'Charging formation follows I/Q, independent of the thermal R1 Arrhenius; all charging B is electrically paid',
                   'Healthy charged Bc and degraded Bd are separate; only R1 accumulates defects; remaining healthy mass sets charge capacity',
                   'Both B fractions can undergo R2; q_charge=q_intrinsic is an unmeasured energy-scale prior, paid charge work has its own ledger',
                   'I²R plus chemical storage cannot exceed U*I; the reversible-voltage floor bounds heat, not chemical storage',
                   'Initial cell is assumed nearly new (A only, no B/S); cycle history and SOH were not published',
                   'Electrical opposite full faces along z are an effective geometry proxy, not measured electrode-tab positions']),
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
        if material in ('ISS','TIM'):
            dims[1]-=2*geometry['pad_edge_clearance_m'];dims[2]-=2*geometry['pad_edge_clearance_m']
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
            thermal['surface_connectivity']=casing_response(p)
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
                initial_precursor_fraction=0.,charge_conversion_per_nominal_charge=p['charge_formation_gain'],gas_yield=p['gas_yield'],gas_yield_max=.8,
                gas_yield_reference_k=450.,gas_yield_temperature_per_k=p['gas_yield_temperature'],gas_yield_density_exponent=.2,
                precursor_yield=p['precursor_yield'],precursor_yield_temperature_per_k=p['precursor_temperature'],
                precursor_temperature_ref_k=REFERENCE,
                gas_chemical_fraction=p['gas_fraction'],gas_compartment='inside-'+ident+'-0',
                # S and R2 condensed C are chemically inert, not ash: keep intact k.
                residue_conductivity=[1.,1.,1.],max_fraction_per_step=.02,max_temperature_step_k=1.)
            scene['shells'].append(dict(contents_id=ident,material_id='unidentified-pouch-laminate-PROXY',thickness_m=t,
                density_kg_m3=1600.,specific_heat_j_kg_k=1250.,conductivity_w_per_m_k=[.3,50.,50.],
                conductivity_frame='normal_tangential',initial_temperature_k=initial,contents_contact_h=p['case_h'],
                surface_connectivity=casing_response(p),
                seam_contact_h=500.,damage_onset_k=425.,damage_rate_per_s=.1,damage_scale_k=25.,destroyed_conductance_factor=.02))
        if ident=='C2':
            c['electrical']=dict(current_profile_a=drive,voltage_profile_v=volts,resistance_ohm=p['resistance_ohm'],
                temperature_log_slope_per_k=p['resistance_temperature'],charge_log_slope=p['resistance_charge'],
                charge_state_input='stored_precursor',
                product_conductivity_ratio=p['product_conductivity_ratio'],
                terminal_axis='z', # unmeasured effective opposite-face current geometry, not reconstructed electrode tabs
                # No unmeasured fixed OCV reserve: charge storage is paid explicitly.
                # Native still enforces Joule heat + charge work <= measured UI.
                reversible_voltage_floor_v=0.,
                # The final measured U/I response admits ~1.3 ohm immediately
                # before source OFF. This guard is not fitted cell material R.
                reference_temperature_k=REFERENCE,min_resistance_ohm=.0001,max_resistance_ohm=2.,max_temperature_step_k=1.,
                charge_acceptance_fraction=1.,nominal_capacity_coulomb=35.*3600.)
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
        edge(a[0],'+x',b[0],'-x',pad_contact_h(a[2],b[2],p))
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
    # Explicit thermal reservoir wall; its 10 kg mass is an unmeasured proxy.
    wall=dict(id='chamber-wall',role='passive',material_id='steel-PROXY',
        geometry=dict(kind='prismatic',dimensions_m=[.002,1.,1.]),pose=dict(position_m=[-1.,0.,0.]),
        thermal=dict(heat_capacity_j_per_k=5000.,initial_temperature_k=ambient,divisions=[1,1,1],conductivity_w_per_m_k=[45.]*3),
        runaway=dict(energy_j=0.,duration_s=1.))
    scene['cells'].append(wall);objects['chamber-wall']=wall
    for face in ('-x','+x'):port('chamber-wall',face)
    # Internal hot gas/wall contact is NOT the quiescent external-air film.
    # Independently identifiable in destruction; old solid contacts stay fixed.
    scene['boundaries'].append(ambient_boundary('room','chamber-wall:-x',ambient,5.))
    if name=='active-cooling':attach_cooling(scene,objects,ports,port,p,ambient)
    if diagnostic_removed_pads is not None:
        from upv_pad_ablation import configure
        configure(scene,diagnostic_removed_pads)
    gas=gas_geometry_model.build(scene,p,ambient,pressure,gas_geometry)
    scene['metadata'].update(parameters=p,gas_nodes=len(gas['compartments']),event_marker_diagnostic=mark,
                             chamber_volume_m3=.120,source_temperature_is_observation=True)
    if gas_extensions is not None:apply_gas_extensions(scene,gas_extensions)
    return scene,dict(data=d,event=mark,observed=dataio.observed(d,times),experiment=name)


def apply_gas_extensions(scene,configuration):
    """Explicit diagnostic physics, NOT a migrated or accepted calibration."""
    allowed={'mixing_diffusivity_m2_s','mixing_flow_fraction','volatile_phase'}
    if configuration.keys()-allowed:raise ValueError('Unknown gas extension')
    gas=scene['gas_network']
    for key in ('mixing_diffusivity_m2_s','mixing_flow_fraction'):
        value=float(configuration.get(key,0.))
        if not math.isfinite(value) or value<0.:raise ValueError('Invalid mixing coefficient')
        gas[key]=value
    for channel in gas['channels']:
        # Geometric path lengths: pore center distance; half a face width to
        # perimeter gaps; 0.1 m reservoir access. No mesh-count fit coefficient.
        if channel['id'].startswith('pore-'):
            obj=next(c for c in scene['cells']if c['id']==channel['from'].split('-')[1])
            a=int(channel['from'].rsplit('-',1)[1]);b=int(channel['to'].rsplit('-',1)[1])
            axis={1:0,3:1,9:2}[abs(b-a)]
            length=obj['geometry']['dimensions_m'][axis]/3
        elif channel['id'].startswith('near-to-chamber-'):length=.1
        elif channel['kind']=='rupture':length=.001
        elif channel['id'].startswith('feed-gap-'):length=.0002
        else:length=channel.get('mixing_length_m',.02)
        channel['mixing_length_m']=length
    phase=configuration.get('volatile_phase')
    if phase is not None:
        gas['vapor_fuel_heat_j_kg']=phase.get('chemical_j_per_kg',0.)
        for c in scene['cells']:
            if 'conversion'in c:c['conversion']['volatile_phase']=copy.deepcopy(phase)
    scene['metadata'].update(gas_extensions=copy.deepcopy(configuration),zone_diagnostics=True,
        gas_extensions_status='DIAGNOSTIC priors, not calibrated; no solid material/contact changes')


def attach_cooling(scene,objects,ports,port,p,ambient):
    # Aluminium wall + directed water S-link, NOT a prescribed measured ch110.
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
        scene['interfaces'].append(dict(id=ident+'--cooling',port_a=a['id'],port_b=key,law=contact(a['area_m2'],2000.)))
    sink=port('cooling-plate','-z');scene['boundaries'].append(ambient_boundary('water',sink,298.15,p['sink_g']))
    # The free-volume builder excludes this physical plate and discovers its
    # accessible edge routes; no special bottom-to-side gas bypass is added.
