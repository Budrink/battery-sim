"""Spatial UPV gas geometry; chemistry and the solid stack are not fitted here.

The publication does not resolve holder openings or interfacial clearances.
Consequently the outer envelope and face slits below are explicit geometric
priors, NOT dimensions read from the schematic. No passage crosses a solid.
"""
import copy
import math

from gas_void_geometry import VoidGrid, cell_box, shell_panel, volume_centers
from chamber_geometry import graded_chamber
from upv_connectivity import explicit_gas_contacts
from upv_solid_geometry import physical_obstacles

DEFAULT = dict(outer_extent_m=.02, face_gap_m=.0002, pad_edge_clearance_m=0.)
FACES = ('-x', '+x', '-y', '+y', '-z', '+z')


def configuration(values=None):
    result = dict(DEFAULT)
    if set(values or {}) - result.keys():
        raise ValueError('Unknown gas geometry setting')
    result.update(values or {})
    for key, value in result.items():
        if not math.isfinite(value) or (value < 0 if key=='pad_edge_clearance_m' else value <= 0):
            raise ValueError('Invalid gas geometry setting: ' + key)
    if result['outer_extent_m'] > .1 or result['face_gap_m'] > .002:
        raise ValueError('Gas geometry outside declared diagnostic range')
    if result['pad_edge_clearance_m']>.005:raise ValueError('Pad edge clearance exceeds 5 mm diagnostic limit')
    return result


def build(scene, parameters, ambient, pressure, values=None):
    """Construct gas nodes, face passages, local rupture targets and films.

    Returns JSON only. Thermal/mass evolution remains in the CUDA solver.
    Interior pores retain LTE; exterior gases NEVER inherit a pad's capacity.
    """
    cfg = configuration(values)
    scene['metadata']['gas_geometry_controls']=cfg
    objects = {c['id']: c for c in scene['cells']}
    ports = {p['id']: p for p in scene['ports']}
    shells = {s['contents_id']: s for s in scene['shells']}
    ablation=scene['metadata'].get('pad_ablation')
    removed=set(ablation['removed'])if ablation else set()
    if removed and cfg['pad_edge_clearance_m']>0.:
        raise ValueError('Use either pad removal or retained-pad edge openings, not both diagnostics')
    gas = dict(transport='split_local_equilibrium', mass_backend='cuda_network',
        network_pressure_fraction=1., reference_temperature_k=298.15,
        cv_j_per_kg_k=1000., specific_r_j_per_kg_k=287.,
        max_fraction_per_step=.02, max_temperature_step_k=1.,
        compartments=[], channels=[], thermal_contacts=[])
    scene['gas_network'] = gas
    positions, boxes, routes = {}, {}, []
    opening_inlets=[]

    def node(ident, volume, temperature, position, binding=None, oxygen=True):
        if volume <= 0:
            raise ValueError('Nonpositive gas volume: ' + ident)
        n = dict(id=ident, volume_m3=volume,
            initial_mass_kg=pressure*volume/(287.*temperature),
            initial_temperature_k=temperature,
            initial_oxidizer_mass_fraction=.232 if oxygen else 0.,
            reaction_rate_per_s=parameters['gas_rate'],
            oxidizer_heat_j_per_kg=13.1e6,
            reaction_reference_k=parameters['gas_reference_k'],
            reaction_activation_j_per_mol=parameters['gas_activation'])
        if binding:
            n['local_equilibrium'] = binding
        positions[ident] = list(position)
        gas['compartments'].append(n)
        return n

    def passage(ident, a, b, area, length, axis=None):
        if area <= 0 or length <= 0:
            raise ValueError('Degenerate gas passage: ' + ident)
        kind = 'internal' if ident.startswith('pore-') else 'passage'
        if axis is None:
            axis=max(range(3),key=lambda k:abs(positions[a][k]-positions[b][k]))
        gas['channels'].append(dict(id=ident, kind=kind, **{'from': a, 'to': b},
            area_m2=area, discharge_coefficient=.7, opening_pressure_pa=0.,
            mixing_length_m=length,mixing_axis=axis))

    def film(compartment, port_id, thermal_volume=None):
        # Retain the reference scale only. Normalize to the common S law below;
        # the solver never uses this scalar directly as a heat-transfer law.
        q = dict(compartment=compartment, port_id=port_id,
                 h_w_per_m2_k=parameters['environment_h'])
        if thermal_volume is not None:
            q['thermal_volume'] = thermal_volume
        gas['thermal_contacts'].append(q)

    for ident in ('C1', 'C2', 'C3'):
        c = objects[ident]
        sizes = [v/3 for v in c['geometry']['dimensions_m']]
        for i, point in volume_centers(c):
            n = node(f'inside-{ident}-{i}', 5e-6/27,
                c['thermal']['initial_temperature_k'], point,
                dict(cell_id=ident, thermal_volume=i,
                     reactant_density_kg_m3=2000., residue_density_kg_m3=2000.), False)
            if i == 0:
                n['contents_id'] = ident
            xyz = [i % 3, i//3 % 3, i//9]
            for axis in range(3):
                if xyz[axis] == 2:
                    continue
                j = i+3**axis
                passage(f'pore-{ident}-{i}-{j}', n['id'], f'inside-{ident}-{j}',
                    .03*sizes[(axis+1)%3]*sizes[(axis+2)%3], sizes[axis],axis)

    # Layer thicknesses remain fixed. The closed control uses a stack envelope;
    # recessed pads require the individual occupied boxes and their free bands.
    lo, hi = cell_box(objects['steel-L'])[0], cell_box(objects['steel-R'])[1]
    pads = {}
    for e in scene['interfaces']:
        pa, pb = ports[e['port_a']], ports[e['port_b']]
        for own, other in ((pa, pb), (pb, pa)):
            c = objects.get(own['cell_id'])
            if c and c['material_id'] in ('ISS', 'TIM') and ':case:' in other['cell_id']:
                pads[c['id']] = (c, own, other)
    slit_volume=sum(c['geometry']['dimensions_m'][1]*c['geometry']['dimensions_m'][2]*cfg['face_gap_m']
                    for c,_own,_other in pads.values())
    divisions = [[lo[0], hi[0]],
                 [lo[1]+(hi[1]-lo[1])*i/2 for i in range(3)],
                 [lo[2]+(hi[2]-lo[2])*i/4 for i in range(5)]]
    for ident in ('C1', 'C2', 'C3'):
        divisions[0] += [ports[ident+':'+f]['centroid_m'][0] for f in ('-x', '+x')]
    obstacles = [(lo, hi)]
    if ablation is not None or cfg['pad_edge_clearance_m']>0.:
        obstacles=physical_obstacles(objects,shells,removed)
        for c in objects.values():
            if c['material_id']in ('ISS','TIM'):
                a,b=cell_box(c);divisions[0]+=[a[0],b[0]]
    elif 'cooling-plate' in objects:
        obstacles.append(cell_box(objects['cooling-plate']))
    grid,wall_areas,chamber_shape = graded_chamber(lo,hi,obstacles,divisions,cfg['outer_extent_m'],.120-slit_volume)
    bounds=grid.bounds
    for c in grid.cells.values():
        node(c['id'], c['volume'], ambient, c['center'])
        boxes[c['id']] = dict(lo=c['lo'], hi=c['hi'])
    for i, (a, b, area, axis, sign) in enumerate(grid.faces()):
        if b is None:continue # a sealed wall, not a mass passage to a mixed hub
        size = a['hi'][axis]-a['lo'][axis]
        distance = abs(a['center'][axis]-b['center'][axis]) if b else size/2
        passage('void-face-'+str(i), a['id'], b['id'], area, distance,axis)

    def outside(point, face):
        axis = 'xyz'.index(face[1])
        p = list(point)
        # Open pads reach their OWN edge, not an artificial stack-wide edge.
        # Use a physical point just outside the pad when locating its inlet.
        p[axis] = (hi if face[0] == '+' else lo)[axis] + (1e-8 if face[0] == '+' else -1e-8)
        result = grid.locate(p)
        return result['id'] if result else None

    # Interface-local slits, no common inlet manifold. Each patch has a finite
    # contact to the pad AND to the physical outer casing, not to bare contents.
    for ident, (c, own, other) in pads.items():
        dims = c['thermal']['divisions']
        size = c['geometry']['dimensions_m']
        centers = dict(volume_centers(c))
        for i, point in centers.items():
            name = f'gap-{ident}-{i}'
            area = size[1]*size[2]/(dims[1]*dims[2])
            point = list(point)
            point[0] = own['centroid_m'][0]
            node(name, area*cfg['face_gap_m'], ambient, point)
            if ident not in removed:film(name, own['id'], i)
            # Cropped casing port: intersected against the unchanged 3x3 shell
            # grid by the native face-patch builder, not centroid-only heating.
            port_id = name+':case-contact'
            p = copy.deepcopy(other)
            p.update(id=port_id, area_m2=area,
                rectangle_m=[size[1]/dims[1], size[2]/dims[2]], centroid_m=point)
            scene['ports'].append(p)
            film(name, port_id)
            y, z = i % dims[1], i//dims[1]
            for axis, coord, count, cross in ((1, y, dims[1], size[2]/dims[2]),
                                            (2, z, dims[2], size[1]/dims[1])):
                for sign, boundary in (('-', 0), ('+', count-1)):
                    if coord != boundary:
                        continue
                    face = sign+'xyz'[axis]
                    if cfg['pad_edge_clearance_m']>0.:
                        inlet=list(point);a,b=cell_box(c)
                        inlet[axis]=(b if sign=='+'else a)[axis]+(1e-8 if sign=='+'else-1e-8)
                        # The slit is an unresolved face layer. Probe into the
                        # physical pad-thickness gallery, not a closed casing.
                        inlet[0]=c['pose']['position_m'][0]
                        entry=grid.locate(inlet);target=entry['id']if entry else None
                    else:target = outside(point, face)
                    # A cooling plate can obstruct a lower exit. No fallback
                    # shortcut to the chamber or an unrelated side is allowed.
                    if target:
                        channel=f'escape-{ident}-{i}-{face}'
                        if cfg['pad_edge_clearance_m']>0.:
                            mouth=list(point);mouth[axis]=(b if sign=='+'else a)[axis]
                            length=math.dist(point,mouth)+math.dist(mouth,positions[target])
                            opening_inlets.append(dict(channel=channel,path_m=[list(point),mouth,positions[target]],
                                area_m2=cfg['face_gap_m']*cross,length_m=length))
                        else:length=abs(point[axis]-positions[target][axis])
                        passage(channel, name, target,
                            cfg['face_gap_m']*cross, length,axis)
                if coord+1 < count:
                    j = i+(1 if axis == 1 else dims[1])
                    passage(f'gap-{ident}-{i}-{j}', name, f'gap-{ident}-{j}',
                        cfg['face_gap_m']*cross, size[axis]/count,axis)

    if ablation is not None:
        from upv_pad_ablation import couple_empty_slots
        couple_empty_slots(scene,grid,objects,ports,removed,pads,passage,film)

    if cfg['pad_edge_clearance_m']>0.:
        from upv_pad_openings import exposed_contacts
        exposed_contacts(scene,grid,objects,ports,shells,film)
        scene['metadata']['pad_openings']['slit_inlets']=opening_inlets

    # An opening belongs to ONE casing tile and ONE local receiver. Preserve
    # its exact mechanical law and fracture energy, do not duplicate apertures
    # when the gas and shell grids have different resolution.
    for ident in ('C1', 'C2', 'C3'):
        core, shell = objects[ident], shells[ident]
        for face in FACES:
            axis = 'xyz'.index(face[1])
            panel = shell_panel(core, shell, face)
            for local, point in volume_centers(panel):
                ijk = []
                core_lo, core_hi = cell_box(core)
                for k in range(3):
                    ijk.append(max(0, min(2, int(3*(point[k]-core_lo[k])/(core_hi[k]-core_lo[k])))))
                ijk[axis] = 0 if face[0] == '-' else 2
                source = ijk[0]+3*(ijk[1]+3*ijk[2])
                if axis == 0:
                    pad = next((v for v in pads.values() if v[2]['cell_id'] == ident+':case:'+face), None)
                    if pad is None:
                        raise ValueError('Missing battery face slit: '+ident+face)
                    c = pad[0]
                    a, b = cell_box(c)
                    ny, nz = c['thermal']['divisions'][1:]
                    y = max(0, min(ny-1, int(ny*(point[1]-a[1])/(b[1]-a[1]))))
                    z = max(0, min(nz-1, int(nz*(point[2]-a[2])/(b[2]-a[2]))))
                    target = f'gap-{c["id"]}-{y+ny*z}'
                else:
                    target = outside(point, face)
                    # A blocked shell remains a mechanical barrier backed by
                    # an occupied solid, not an outlet into a nonexistent void.
                    if target is None:
                        continue
                label = f'rupture-{ident}-{face}-{local}'
                gas['channels'].append(dict(id=label, kind='rupture',
                    **{'from': f'inside-{ident}-{source}', 'to': target},
                    port_id=ident+':case:'+face+':outer', shell_thermal_volume=local,
                    opening_pressure_pa=parameters['shell_strength_pa'],
                    strength_reference_k=298.15, strength_temperature_scale_k=300.,
                    discharge_coefficient=.7, mixing_length_m=.001,mixing_axis=axis,
                    damage_opening=dict(mode='deterministic_aperture',
                        area_rate_per_s=parameters['opening_rate'], pressure_exponent=1.,
                        max_damage_step=.02, seam_width_m=.001,
                        seam_strength_factor=.2, fracture_energy_j_per_m2=2000.)))
                routes.append(dict(id=label, point_m=point, receiver=target, face=face))
                if axis:
                    film(target, ident+':'+face, local)

    # One thermal wall object, partitioned reference scales. Actual heat uses S.
    # The sum of reference h*A is EXACTLY
    # the former whole-wall value, not an additional or stronger cooling path.
    wall_port=ports['chamber-wall:+x'];total_area=sum(wall_areas.values());cursor=0.
    for j,(name,area)in enumerate(sorted(wall_areas.items())):
        fraction=1.-cursor if j+1==len(wall_areas) else area/total_area
        port=copy.deepcopy(wall_port);port['id']='chamber-wall:gas-partition:'+str(j)
        width,height=wall_port['rectangle_m'];port['rectangle_m']=[width,height*fraction]
        port['area_m2']=wall_port['area_m2']*fraction
        port['centroid_m'][2]+=height*(cursor+fraction/2-.5);cursor+=fraction
        scene['ports'].append(port)
        gas['thermal_contacts'].append(dict(compartment=name,port_id=port['id'],h_w_per_m2_k=parameters['chamber_gas_h']))
    explicit_gas_contacts(scene)
    scene['metadata']['gas_pressure_observers']=dict(chamber=dict(mode='volume_mean',
        compartments=[n['id']for n in gas['compartments']if not n['id'].startswith('inside-')]))
    scene['metadata']['gas_geometry'] = dict(version='spatial-closed-chamber-v2', configuration=cfg,
        envelope_m=bounds, void_boxes=boxes, node_positions_m=positions, rupture_routes=routes,
        chamber_shape=chamber_shape,wall_contact_partition=wall_areas,
        external_volume_m3=.120, void_cells=len(grid.cells),
        gap_contact_h_w_m2_k=parameters['environment_h'],
        status='Diagnostic geometry, not calibrated; passage/holder clearances are not measured',
        limitations=[('Pad edge recess is an explicit unmeasured hypothesis'if cfg['pad_edge_clearance_m']>0.else
            'Solid full-face footprints retained; schematic not treated as a scale drawing'),
            'Interface slits are unresolved clearance priors, not permeability through intact pads',
            'Equal-clearance chamber shape is a volume-constrained proxy, not measured dimensions',
            'One shell-tile opening routes by centroid; thermal films use area intersections',
            'Directed gas metric has neutral unmeasured defaults; reference scales shared across cells',
            'No momentum, radiation or deformation-driven widening; no CFD'])
    return gas
