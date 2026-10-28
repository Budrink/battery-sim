"""Accepted CUDA heat records and independent gas-inventory energy closure."""
import numpy as np


def solid_group(name):
    if name=='chamber-wall':return 'chamber-wall'
    if ':case:'in name:return name.split(':')[0]+' casing'
    if name.startswith('ISS'):return 'ISS pads'
    if name.startswith('TIM'):return 'TIM pads'
    return 'plates'


def cumulative(run,at):
    """Net contact energy, not stored energy or a sparse flux quadrature.

    Positive/negative columns are split by accepted physical step. They are
    not claimed as gross per-reactor-substage exchange.
    """
    times=np.asarray(run['time_s']);objects={};groups={}
    for contact in run['gas_film_trace'].values():
        name=contact['solid'];row=objects.setdefault(name,dict(net_j=0.,positive_step_j=0.,negative_step_j=0.))
        group=groups.setdefault(solid_group(name),dict(net_j=0.,positive_step_j=0.,negative_step_j=0.))
        for out,key in [('net_j','heat_to_solid_j'),('positive_step_j','positive_step_heat_j'),('negative_step_j','negative_step_heat_j')]:
            value=float(np.interp(at,times,contact[key]))
            row[out]+=value;group[out]+=value
    return dict(time_s=at,objects=objects,groups=groups)


def gas_region(scene,run,names,at):
    """Independent local closure: transport + R3 - gas storage = surface heat."""
    times=run['time_s'];network=scene['gas_network'];cv=network['cv_j_per_kg_k'];ref=network['reference_temperature_k']
    net=burn=storage=actual=initial_o2=oxygen=consumed=oxygen_in=0.
    def value(row,key):return float(np.interp(at,times,row[key]))
    for name in names:
        g=run['gas'][name]
        # Interpolate the extensive energy, NOT mass and temperature separately.
        # Their product is nonlinear and creates a false audit residual between
        # sparse observations even when every accepted CUDA step conserves E.
        sensible=cv*np.asarray(g['mass_kg'])*(np.asarray(g['temperature_k'])-ref)
        storage+=float(np.interp(at,times,sensible))-sensible[0]
        storage+=value(g,'latent_j')-g['latent_j'][0]
        burn+=value(g,'released_energy_j')-g['released_energy_j'][0]
        initial_o2+=g['oxidizer_kg'][0];oxygen+=value(g,'oxidizer_kg')
        consumed+=value(g,'consumed_oxidizer_kg')-g['consumed_oxidizer_kg'][0]
    for edge in network['channels']:
        sign=int(edge['to']in names)-int(edge['from']in names)
        if not sign:continue
        row=run['channels'][edge['id']]
        net+=sign*(value(row,'delivered_gas_energy_j')-row['delivered_gas_energy_j'][0]+
                   value(row,'transferred_latent_j')-row['transferred_latent_j'][0])
        oxygen_in+=sign*(value(row,'transferred_oxidizer_kg')-row['transferred_oxidizer_kg'][0])
    for contact in run['gas_film_trace'].values():
        if contact['gas']in names:actual+=value(contact,'heat_to_solid_j')-contact['heat_to_solid_j'][0]
    return dict(nodes=len(names),net_transport_j=net,r3_j=burn,gas_energy_change_j=storage,
        actual_heat_to_surfaces_j=actual,energy_closure_j=net+burn-storage-actual,
        remaining_oxygen_kg=oxygen,consumed_oxygen_kg=consumed,net_inward_oxygen_kg=oxygen_in,
        oxygen_closure_kg=initial_o2+oxygen_in-consumed-oxygen)


def inspect(scene,run):
    if 'gas_film_trace'not in run:raise ValueError('Actual accepted-step heat trace required')
    end=float(run['time_s'][-1]);times=sorted({0.,750.,860.,885.,900.,930.,950.,end})
    times=[at for at in times if run['time_s'][0]<=at<=end]
    exterior={name for name in run['gas']if not name.startswith('inside-')}
    wall_nodes={c['compartment']for c in scene['gas_network']['thermal_contacts']
                if c['port_id'].startswith('chamber-wall:gas-partition:')}
    volumes={n['id']:n['volume_m3']for n in scene['gas_network']['compartments']}
    external_volume=sum(volumes[n]for n in exterior)
    c3iss={name for name in exterior if name.startswith('gap-ISS1-')}
    c3iss.update(c['gas']for c in scene['metadata'].get('pad_openings',{}).get('exposed_contacts',[])
                 if c['port'].startswith('opening-contact:ISS1:'))
    snapshots=[]
    for at in times:
        row=cumulative(run,at)
        row['external_gas']=gas_region(scene,run,exterior,at)
        row['C3_ISS_slit_and_edge_galleries']=gas_region(scene,run,c3iss,at)
        wall_burn=sum(float(np.interp(at,run['time_s'],run['gas'][n]['released_energy_j']))-
                      run['gas'][n]['released_energy_j'][0]for n in wall_nodes)
        total_burn=row['external_gas']['r3_j']
        row['reaction_location']=dict(wall_touching_nodes=len(wall_nodes),external_nodes=len(exterior),
            wall_touching_reference_volume_fraction=sum(volumes[n]for n in wall_nodes)/external_volume,
            wall_touching_r3_j=wall_burn,wall_touching_r3_fraction=wall_burn/total_burn if total_burn>0.else None)
        for region in ('external_gas','C3_ISS_slit_and_edge_galleries'):
            if abs(row[region]['energy_closure_j'])>1e-4 or abs(row[region]['oxygen_closure_kg'])>1e-10:
                raise ValueError('Independent gas-region balance does not close: '+region)
        snapshots.append(row)
    return dict(snapshots=snapshots,end_s=end,wall_s=run['wall_s'],
        speed=(run['advanced_to_s']-run['started_at_s'])/run['wall_s'],
        minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'],
        residuals={k:v for k,v in run.items()if k.startswith('max_')and 'residual'in k})


if __name__=='__main__':
    import argparse
    import gzip
    import json
    from pathlib import Path
    from calibrate_upv_r123 import save
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenario',type=Path,required=True);ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    scene=json.loads(args.scenario.read_text())
    opener=gzip.open if args.run.suffix=='.gz'else open
    with opener(args.run,'rt')as stream:run=json.load(stream)
    save(args.output,inspect(scene,run))
