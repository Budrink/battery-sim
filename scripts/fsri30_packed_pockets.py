"""Split lateral gas at tangencies into physical triangular void pockets.

Each equilateral inter-cylinder pocket and the connected exterior body space
is a gas compartment; the spaces above and below the cans join the pockets.
The graph resolves connectivity, not velocity, flame shape or the unknown
actual mechanical spacers. Temperature and species are uniform per pocket.
"""
from copy import deepcopy
import math
import numpy as np
from scipy.spatial import Delaunay

from generate_fsri30 import DIAMETER, HEIGHT


def split_tangent_pockets(base, angular_samples=1440):
    s=deepcopy(base);meta=s['metadata'];gas=s['gas_network']
    if 'packed_cylinder_geometry' not in meta or 'can_contact_hypothesis' in meta:
        raise ValueError('Require packed layout before creating metal contacts')
    if angular_samples<360 or angular_samples%8:raise ValueError('Angular quadrature must be divisible by 8')
    points=np.array([meta['approximate_positions_m'][f'B{i:02d}'][:2] for i in range(30)])
    triangulation=Delaunay(points);triangles=[];simplex_pocket={}
    for index,vertices in enumerate(triangulation.simplices):
        if all(abs(np.linalg.norm(points[vertices[i]]-points[vertices[(i+1)%3]])-DIAMETER)<1e-10 for i in range(3)):
            simplex_pocket[index]=len(triangles);triangles.append(sorted(map(int,vertices)))
    if len(triangles)!=len(meta['neighbour_pairs'])-len(points)+1:
        raise ValueError('The planar packed graph is not completely triangulated')
    radius=DIAMETER/2
    pore_area=(math.sqrt(3)-math.pi/2)*radius**2
    if pore_area<=0:raise ValueError('Invalid triangular pocket area')
    nodes={n['id']:n for n in gas['compartments']};old_bodies=[nodes[f'body-{i:02d}'] for i in range(30)]
    for n in old_bodies:
        if any(n[k]!=old_bodies[0][k] for k in ('initial_temperature_k','initial_oxidizer_mass_fraction','reaction_rate_per_s','reaction_reference_k','reaction_activation_j_per_mol')):
            raise ValueError('Body gas materials differ')
    original_volume=sum(n['volume_m3'] for n in old_bodies)
    rho=old_bodies[0]['initial_mass_kg']/old_bodies[0]['volume_m3']
    incidence={i:[] for i in range(30)}
    for j,triangle in enumerate(triangles):
        for i in triangle:incidence[i].append(j)
    exterior_shares=[]
    for i,n in enumerate(old_bodies):
        area=n['volume_m3']/HEIGHT-len(incidence[i])*pore_area/3
        if area < -1e-12:raise ValueError('Pocket partition exceeds cell void')
        exterior_shares.append(max(0.,area))
    exterior_volume=sum(exterior_shares)*HEIGHT
    if not math.isclose(exterior_volume+len(triangles)*pore_area*HEIGHT,original_volume,rel_tol=2e-12):
        raise ValueError('Pocket gas volume does not close')
    gas['compartments']=[n for n in gas['compartments'] if not (n['id'].startswith('body-') and n['id'][5:].isdigit())]
    def node(name,volume):
        n=deepcopy(old_bodies[0]);n.update(id=name,volume_m3=volume,initial_mass_kg=rho*volume)
        gas['compartments'].append(n)
    node('body-exterior',exterior_volume)
    for j in range(len(triangles)):node(f'body-pocket-{j:02d}',pore_area*HEIGHT)
    # Remove fictitious same-height connections through tangency. Gas can go
    # via the common top/base chambers; external vent holes reach exterior gas.
    gas['channels']=[e for e in gas['channels'] if not e['id'].startswith(('free-body-','cap-'))]
    for edge in gas['channels']:
        if edge['id'].startswith('vent-'):
            edge['from']='body-exterior'
    cap_height=((np.asarray(meta['geometry_reconstruction']['enclosure_dimensions_m'])-2*meta['geometry_budget']['abs_thickness_m'])[2]-HEIGHT)/2
    def passage(name,a,b,area):
        gas['channels'].append(dict(id=name,kind='passage',**{'from':a,'to':b},area_m2=area,
            mixing_length_m=(HEIGHT+cap_height)/2,mixing_axis=2,discharge_coefficient=.7,opening_pressure_pa=0.))
    for i in range(30):
        for band in ('top','base'):
            if exterior_shares[i]>1e-15:
                passage(f'cap-{band}-outer-{i}', 'body-exterior',f'{band}-{i:02d}',exterior_shares[i])
            for j in incidence[i]:
                passage(f'cap-{band}-pocket-{j}-{i}',f'body-pocket-{j:02d}',f'{band}-{i:02d}',pore_area/3)
    # Assign each cylinder face's lateral gas exposure by angular quadrature.
    # Its four planar surrogate faces each represent a 90-degree cylinder arc.
    # Points infinitesimally outside the can select the adjacent Delaunay pore.
    angle=(np.arange(angular_samples)+.5)*(2*math.pi/angular_samples)
    directions=np.stack((np.cos(angle),np.sin(angle)),axis=1)
    faces=np.where(abs(directions[:,0])>=abs(directions[:,1]),
        np.where(directions[:,0]>=0,'+x','-x'),np.where(directions[:,1]>=0,'+y','-y'))
    fractions={}
    for i in range(30):
        simplex=triangulation.find_simplex(points[i]+directions*(radius+1e-8),tol=1e-12)
        for face in ('-x','+x','-y','+y'):
            indices=np.flatnonzero(faces==face);counts={}
            for k in indices:
                region=f'body-pocket-{simplex_pocket[int(simplex[k])]:02d}' if int(simplex[k]) in simplex_pocket else 'body-exterior'
                counts[region]=counts.get(region,0)+1
            fractions[(i,face)]={name:count/len(indices) for name,count in counts.items()}
    new_films=[]
    for film in gas['thermal_contacts']:
        body=film['compartment']
        if not (body.startswith('body-') and body[5:].isdigit()):
            new_films.append(film);continue
        if film['port_id'].startswith('gasmesh:ABS:'):
            f=deepcopy(film);f['compartment']='body-exterior';new_films.append(f);continue
        i=int(body[5:]);face=film['port_id'].split(':case:')[1].split(':')[0]
        if (i,face) not in fractions:raise ValueError('Unexpected body gas film '+film['port_id'])
        for region,fraction in fractions[(i,face)].items():
            f=deepcopy(film);f['compartment']=region
            for key in f['connectivity']:
                if key.endswith('conductance_w_per_k'):f['connectivity'][key]*=fraction
            new_films.append(f)
    gas['thermal_contacts']=new_films
    final_volume=sum(n['volume_m3'] for n in gas['compartments'] if n['id'].startswith('body-'))
    if not math.isclose(final_volume,original_volume,rel_tol=2e-12):raise ValueError('Split body volume changed')
    meta['packed_cylinder_geometry']['limitation']='isothermal triangular pockets; no resolved momentum or unknown holder'
    meta['packed_cylinder_geometry']['gas_partition']='40 triangular tangent-cylinder pockets plus exterior body, linked through top/base'
    meta['packed_pocket_geometry']=dict(pocket_count=len(triangles),pockets=triangles,pocket_area_m2=pore_area,
        exterior_volume_m3=exterior_volume,original_body_volume_m3=original_volume,
        total_body_volume_m3=final_volume,angular_quadrature_samples=angular_samples,
        direct_body_pocket_passages=0,film_fractions={f'B{i:02d}:{face}':v for (i,face),v in fractions.items()},
        calibrated=False,validation_passed=False)
    return s
