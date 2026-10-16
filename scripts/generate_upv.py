#!/usr/bin/env python3
"""Measured-boundary UPV heat-transfer cases, NOT autonomous runaway models.

C2 is an external measured boundary, not a solved volume. Only C1/C3 and the
passive thermal paths are evolved. Full-area geometry is a documented prior.
"""
from __future__ import annotations
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TARGETS=('101','102','107','108')
INPUTS=('103','104','105','106','109')
SOURCE='https://zenodo.org/records/22273467'
TOPOLOGIES=('swapped-witnesses','published')
CHANNEL_FACES=(('101','C1','ISS'),('102','C1','TIM'),('107','C3','TIM'),('108','C3','ISS'))


def bind_measurement_probes(scene):
    """Bind Table 5 channels by battery AND neighboring material, never array order."""
    ports={p['id']:p for p in scene['ports']}
    cells={c['id']:c for c in scene['cells']}
    probes=[]
    for channel,cell,material in CHANNEL_FACES:
        matches=[]
        for edge in scene['interfaces']:
            a,b=ports[edge['port_a']],ports[edge['port_b']]
            for own,other in ((a,b),(b,a)):
                if own['cell_id']==cell and cells[other['cell_id']]['material_id']==material:
                    matches.append(dict(id=channel,cell_id=cell,interface_id=edge['id'],
                                        port_id=own['id'],neighbor_port_id=other['id'],
                                        neighbor_material=material))
        if len(matches)!=1:raise ValueError(f'Channel {channel}: expected one {cell}/{material} contact, got {len(matches)}')
        probes.append(matches[0])
    scene['measurement_probes']=probes


def load_case(directory, experiment):
    manifest=json.loads((directory/'manifest.json').read_text())
    record=manifest['experiments'][experiment]
    raw=(directory/record['canonical_csv']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record['canonical_sha256']:
        raise ValueError('Canonical CSV checksum mismatch')
    rows=list(csv.DictReader(raw.decode().splitlines()))
    data={k:[float(row[k]) if row[k] else math.nan for row in rows]
          for k in rows[0] if k=='time_s' or k.startswith('T')}
    return data, record, manifest


def pre_event_duration(data, *, use_contact_driver=True):
    # Protocol fixed before fitting: first invalid driver or C2 surface >=150 C.
    # No witness/validation target influences window selection.
    drivers=INPUTS+('110',) if use_contact_driver and 'T110_k' in data else INPUTS
    for i,time in enumerate(data['time_s']):
        invalid=any(not math.isfinite(data[f'T{c}_k'][i]) for c in drivers)
        hot=any(data[f'T{c}_k'][i]>=423.15 for c in INPUTS[:4])
        if invalid or hot:
            end=math.floor((time-2)/2)*2
            if end<30:raise ValueError('Insufficient valid pre-event forcing')
            return end
    raise ValueError('No forcing-defined end to pre-event window')


def interpolate(time, values, at):
    i=max(0,min(len(time)-2,bisect.bisect_right(time,at)-1))
    return values[i]+(values[i+1]-values[i])*(at-time[i])/(time[i+1]-time[i])


def generate(directory, experiment, topology='swapped-witnesses', connectivity='constant', battery_divisions=(3,3,3), cooling_model='thermal-wall'):
    if experiment not in ('no-cooling','active-cooling'):raise ValueError('A module experiment is required')
    if topology not in TOPOLOGIES:raise ValueError('Unknown witness topology')
    if connectivity not in ('constant','state_geometry','directed_geometry'):raise ValueError('Unknown connectivity model')
    if cooling_model not in ('thermal-wall', 'legacy-boundary'):raise ValueError('Unknown cooling model')
    if len(battery_divisions)!=3 or any(type(n) is not int or not 1<=n<=256 for n in battery_divisions):
        raise ValueError('Battery divisions must be three integers in [1,256]')
    data,record,manifest=load_case(directory,experiment)
    duration=pre_event_duration(data, use_contact_driver=cooling_model=='legacy-boundary')
    def curve(channels):
        values=[sum(data[f'T{c}_k'][i] for c in channels)/len(channels) for i in range(len(data['time_s']))]
        result=[[t,interpolate(data['time_s'],values,t)] for t in range(duration+1)]
        if any(not math.isfinite(v) or v<=0 for _,v in result):raise ValueError('Invalid forcing; no implicit gap filling')
        return result
    ambient=curve(['109']);initial=ambient[0][1]
    s=dict(name=f'UPV {experiment}: conditional pre-event heat transfer',source=SOURCE,
           metadata=dict(dataset_id='upv-sssb-v1',experiment_id=experiment,split=record['split'],
                         calibration_status='uncalibrated measured-boundary baseline',
                         validation_scope='pre-runaway conditional heat transfer only; not cascade/aging validation',
                         geometry_status='layer thickness/order incl. outer ISS4 from Fig.2/Table3; full-face areas and fixture extent assumed',
                         topology=topology,
                         battery_grid=list(battery_divisions),
                         region_definition='Core: no exterior face; shell: one numerical layer on all six faces, not a casing material',
                         topology_status=('UNCONFIRMED acquisition-numbering hypothesis: C3 left, C1 right; contradicts Fig.2 witness labels; Table5 material sides retained'
                                          if topology=='swapped-witnesses' else 'Published Fig.2 numbering: C1 left, C3 right'),
                         raw_archive_sha256=manifest['archive_sha256'],canonical_sha256=record['canonical_sha256'],
                         drivers=['C2 surface temperatures','ch109 environment']+(['ch110 cooling contact'] if experiment=='active-cooling' else []),
                         target_channels=list(TARGETS),source_cell_not_solved='C2 represented by measured surface boundary conditions',
                         window_rule='t=0 to first invalid driver or C2 surface >=150 C, minus 2 s; floor to 2 s',
                         assumptions=['surface thermocouples compared to reconstructed face means; exact coordinates unavailable',
                                      'C2 upper/lower measurements averaged on each face',
                                      'ISS, TIM, aluminium and steel modeled with full cell face area',
                                      'passive cp priors: ISS/TIM 1000, Al 900, steel 500 J/kg/K; not measured in source',
                                      'pressure plates assumed same face dimensions as cell; resin holders omitted',
                                      'baseline h=100 W/m2/K; calibration may distinguish material-pair contacts; environmental h is effective',
                                      'cooled case: ch110 imposed on all three spreader tips; spatial uniformity unverified',
                                      'l0=1e-4 m is an order-parameter convention, NOT identified by this fit']),
           simulation=dict(duration_s=duration,max_dt_s=.5,output_interval_s=2,stability_factor=.4),
           cells=[],ports=[],interfaces=[],boundaries=[],primary_failures=[],measurement_probes=[])
    width,height=.087,.187
    objects={}
    left,right=('C3','C1') if topology=='swapped-witnesses' else ('C1','C3')
    # The omitted source C2 occupies its actual 11 mm gap.
    layers=[('steel-L',.010,'steel',45,7850,500),('ISS1',.002,'ISS',.145,200,1000),
            (left,.011,'battery',.9,2291,1007),('TIM1',.002,'TIM',2,3300,1000),('Al1',.001,'aluminium',235,2700,900),
            ('ISS2',.002,'ISS',.145,200,1000),('C2',.011,'source',0,0,0),
            ('TIM2',.002,'TIM',2,3300,1000),('Al2',.001,'aluminium',235,2700,900),
            ('ISS3',.002,'ISS',.145,200,1000),(right,.011,'battery',.9,2291,1007),
            ('TIM3',.002,'TIM',2,3300,1000),('Al3',.001,'aluminium',235,2700,900),
            ('ISS4',.002,'ISS',.145,200,1000),('steel-R',.010,'steel',45,7850,500)]
    x=0
    for identifier,thickness,material,k,rho,cp in layers:
        x+=thickness/2
        if identifier!='C2':
            temperature=initial
            if identifier in ('C1','C3'):
                channels=('101','102') if identifier=='C1' else ('107','108')
                temperature=sum(data[f'T{c}_k'][0] for c in channels)/2
            capacity=.410*1007 if material=='battery' else thickness*width*height*rho*cp
            obj=dict(id=identifier,role='battery' if material=='battery' else 'passive',material_id=material,
                     geometry=dict(kind='pouch' if material=='battery' else 'prismatic',dimensions_m=[thickness,width,height]),
                     pose=dict(position_m=[x,0,0]),
                     thermal=dict(heat_capacity_j_per_k=capacity,initial_temperature_k=temperature,
                                  divisions=list(battery_divisions) if material=='battery' else [1,2,4],
                                  conductivity_w_per_m_k=[k,28.3,28.3] if material=='battery' else [k,k,k]),
                     runaway=dict(energy_j=0,duration_s=1))
            objects[identifier]=obj;s['cells'].append(obj)
        x+=thickness/2
    def port(identifier,face):
        key=identifier+':'+face
        if any(p['id']==key for p in s['ports']):return key
        obj=objects[identifier];axis='xyz'.index(face[1]);sign=1 if face[0]=='+' else -1
        dims=obj['geometry']['dimensions_m'];pos=obj['pose']['position_m'].copy();pos[axis]+=sign*dims[axis]/2
        normal=[0,0,0];normal[axis]=sign
        s['ports'].append(dict(id=key,cell_id=identifier,kind='contact',surface=face,
                               area_m2=dims[(axis+1)%3]*dims[(axis+2)%3],centroid_m=pos,normal=normal))
        return key
    def boundary(identifier,face,label,profile,h):
        p=port(identifier,face);area=next(v['area_m2'] for v in s['ports'] if v['id']==p)
        s['boundaries'].append(dict(id=label,port_id=p,temperature_k=profile[0][1],
                                    temperature_profile=profile,conductance_w_per_k=h*area))
    for a,b in zip(layers,layers[1:]):
        if 'C2' in (a[0],b[0]):continue
        s['interfaces'].append(dict(id=a[0]+'--'+b[0],port_a=port(a[0],'+x'),port_b=port(b[0],'-x'),
                                    law=dict(kind='constant',l0_m=1e-4,background_conductance_w_per_k=100*width*height)))
    boundary('ISS2','+x','source-C2-ISS',curve(['105','106']),100)
    boundary('TIM2','-x','source-C2-TIM',curve(['103','104']),100)
    cooled=experiment=='active-cooling'
    for identifier in objects:
        for face in ('-y','+y','-z','+z'):
            if cooled and identifier.startswith('Al') and face=='-z':
                # h here is a fixed high-contact prior, not fitted on validation.
                # Wall mode uses these only as geometry handles for attach();
                # channel 110 must not affect its forcing or window selection.
                profile=curve(['110']) if cooling_model=='legacy-boundary' else [[0.,initial],[duration,initial]]
                boundary(identifier,face,'cooling-'+identifier,profile,10000)
            else:boundary(identifier,face,'ambient-'+identifier+face,ambient,5)
    boundary('steel-L','-x','ambient-end-L',ambient,5)
    boundary('steel-R','+x','ambient-end-R',ambient,5)
    bind_measurement_probes(s)
    if connectivity in ('state_geometry','directed_geometry'):
        # Priors, not measurements of the actual porous ISS/composite silicone.
        # All bulk material properties remain fixed throughout this calibration.
        s['metadata']['connectivity_model']=connectivity
        s['metadata']['fixed_pad_cp']=True
        s['metadata']['pad_specific_heat_j_kg_k']={'ISS':770.,'TIM':1175.}
        s['metadata']['assumptions']=[v for v in s['metadata']['assumptions'] if not v.startswith('passive cp priors')]
        s['metadata']['assumptions'] += [
            'Fixed working cp: ISS 770, TIM 1175, Al 900, steel 500 J/kg/K; pad composition-specific cp unverified',
            ('Directed log connectivity: sender polynomial plus signed temperature/damage contrasts; no endpoint averaging'
             if connectivity=='directed_geometry' else 'Legacy reciprocal pair projection of local log metrics'),
            'Contact h bounded 1..10000 W/m2/K; these are engineering bounds, not material measurements',
            'Destruction branch and damage kinetics are NOT identifiable from pre-event data']
        response=dict(reference_temperature_k=298.15,temperature_scale_k=100.,
                      linear=[0.,0.,0.],quadratic=[0.,0.,0.],damage=[0.,0.,0.],
                      temperature_contrast=[0.,0.,0.],damage_contrast=[0.,0.,0.])
        for obj in s['cells']:
            cp=s['metadata']['pad_specific_heat_j_kg_k'].get(obj['material_id'])
            if cp:obj['thermal']['heat_capacity_j_per_k']*=cp/1000.
            obj['thermal']['surface_connectivity']=dict(response)
        def law(area):
            return dict(kind=connectivity,l0_m=1e-4,activation_width_m=1e-5,
                        geometry_reference_logit=math.log((100.-1.)/(10000.-100.)),
                        background_conductance_w_per_k=area,active_conductance_w_per_k=9999*area,
                        endpoint_damage_weight=1.,destruction_damage=1.,
                        destroyed_l0_m=2e-4,destroyed_l_m=2e-4,
                        destroyed_background_conductance_w_per_k=area,
                        destroyed_active_conductance_w_per_k=0.)
        ports={p['id']:p for p in s['ports']}
        for edge in s['interfaces']:edge['law']=law(ports[edge['port_a']]['area_m2'])
        for b in s['boundaries']:
            if b['id'].startswith('source-'):
                b['connectivity']=law(ports[b['port_id']]['area_m2'])
                b['remote_surface_connectivity']=dict(response)
    if cooled and cooling_model == 'thermal-wall':
        from upv_cooling_wall import attach
        s = attach(s, coolant_temperature_k=data.get('T111_k', [298.15])[0])
    return s


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/experiments/upv')
    p.add_argument('--experiment',choices=('no-cooling','active-cooling'),required=True)
    p.add_argument('--topology',choices=TOPOLOGIES,default='swapped-witnesses')
    p.add_argument('--connectivity',choices=('constant','state_geometry','directed_geometry'),default='constant')
    p.add_argument('--battery-divisions',type=int,nargs=3,default=(3,3,3),metavar=('NX','NY','NZ'))
    p.add_argument('--cooling-model', choices=('thermal-wall', 'legacy-boundary'), default='thermal-wall',
                   help='Explicit finite-capacity wall; legacy is only for historical reproduction')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();s=generate(a.data,a.experiment,a.topology,a.connectivity,a.battery_divisions,a.cooling_model)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(s,separators=(',',':'))+'\n')
    print(a.output, 'duration_s=',s['simulation']['duration_s'])


if __name__=='__main__':main()
