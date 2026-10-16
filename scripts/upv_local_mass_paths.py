"""Local UPV gas routing; apply AFTER selecting the independent casing grid.

Preserves thermal/chemical laws and initial inventories. The existing pad-face
slits remain explicit, unmeasured subgrid assumptions (not pad permeability).
The outer void is now spatially resolved, with no whole-side mixing reservoir.
"""
import copy
import math
from gas_void_geometry import VoidGrid, cell_box, volume_centers, shell_panel
import upv_mass_paths as old


def attach(scene, *, extent_m=.02, radial_layers=2, obstacles=()):
    if not math.isfinite(extent_m) or extent_m<=0 or not isinstance(radial_layers,int) or radial_layers<1:
        raise ValueError('Invalid local void resolution')
    s = copy.deepcopy(scene);frozen = old.thermal_signature(s)
    cells = {c['id']: c for c in s['cells']};shells = {c['contents_id']: c for c in s['shells']}
    ports = {p['id']: p for p in s['ports']};gas = s['gas_network']
    original = {n['id']: n for n in gas['compartments']}
    if 'near-+y' not in original:
        raise ValueError('Expected the frozen mass-path scenario, not an already localized network')
    external = [n for n in original.values() if not n['id'].startswith('inside-C2')]
    total_v = sum(n['volume_m3'] for n in external);total_m = sum(n['initial_mass_kg'] for n in external)
    cv, tref, r = gas['cv_j_per_kg_k'], gas['reference_temperature_k'], gas['specific_r_j_per_kg_k']
    total_u = sum(n['initial_mass_kg']*cv*(n['initial_temperature_k']-tref) for n in external)
    chamber = copy.deepcopy(original['chamber'])
    ambient = original['near-+y']['initial_temperature_k']
    p0 = original['near-+y']['initial_mass_kg']*r*ambient/original['near-+y']['volume_m3']
    nodes = [copy.deepcopy(n) for n in original.values() if n['id'].startswith('inside-C2') or
             (n['id'].startswith('gap-') and not n['id'].endswith('-inlet'))]
    # Remove inlet manifolds, restoring their volume to the adjacent slit cells.
    for pad in ('ISS2', 'TIM2'):
        for n in nodes:
            if n['id'].startswith('gap-'+pad+'-'):
                n['volume_m3'] /= .9
                n['initial_mass_kg'] = p0*n['volume_m3']/(r*n['initial_temperature_k'])
    paths = [copy.deepcopy(p) for p in gas['channels'] if p['id'].startswith('core-') or p['id'].startswith('gap-')]
    for p in paths:
        if p['id'].startswith('gap-'):p['kind']='passage'
    path_geometry = []

    def passage(a, b, area, label, **geometry):
        paths.append(dict(id=label, kind='passage', **{'from': a, 'to': b}, area_m2=area,
                          discharge_coefficient=old.FIXED['gap_cd'] if a.startswith('gap-') else old.FIXED['near_cd'], opening_pressure_pa=0.))
        path_geometry.append(dict(id=label, **geometry))

    # The frozen stack has flush full-face layers. Do not invent open channels
    # through their occupied volume. Unmeasured slits are represented separately.
    left, right = cell_box(cells['steel-L']), cell_box(cells['steel-R'])
    lo, hi = [left[0][0], left[0][1], left[0][2]], [right[1][0], right[1][1], right[1][2]]
    bounds = ([lo[0], lo[1]-extent_m, lo[2]-extent_m], [hi[0], hi[1]+extent_m, hi[2]+extent_m])
    solids = [(lo,hi), *obstacles]
    if 'cooling-plate' in cells:solids.append(cell_box(cells['cooling-plate']))
    divisions = [[], [], []]
    # Avoid thin artificial strips at the contents/casing boundary in x.
    divisions[0] = [v for c in cells.values() if not c['id'].startswith('C') and c['id'] not in ('chamber-wall','cooling-plate')
                    for v in (cell_box(c)[0][0],cell_box(c)[1][0])]
    for axis, count in ((1,2),(2,4)):
        divisions[axis] = [lo[axis]+(hi[axis]-lo[axis])*i/count for i in range(count+1)]
        for side in (lo[axis],hi[axis]):
            sign = -1 if side==lo[axis] else 1
            divisions[axis] += [side+sign*extent_m*i/radial_layers for i in range(1,radial_layers+1)]
    grid = VoidGrid(bounds,solids,divisions)
    for c in grid.cells.values():
        n = dict(id=c['id'],volume_m3=c['volume'],initial_mass_kg=p0*c['volume']/(r*ambient),initial_temperature_k=ambient)
        for k in ('initial_oxidizer_mass_fraction','reaction_rate_per_s','oxidizer_heat_j_per_kg','ignition_temperature_k'):
            if k in chamber:n[k]=chamber[k]
        nodes.append(n)
    for i,(a,b,area,axis,sign) in enumerate(grid.faces()):
        passage(a['id'],b['id'] if b else 'chamber',area,f'void-face-{i}',axis=axis,sign=sign,area_m2=area)

    blocked = []
    def outside(point,face):
        p=list(point);axis='xyz'.index(face[1]);p[axis]=(hi if face[0]=='+' else lo)[axis]+(1 if face[0]=='+' else -1)*1e-8
        return grid.locate(p)

    # Each slit reaches only its own perimeter, not a remote face or battery.
    gap = s['metadata']['mass_paths']['parameters']['gap_m']
    for pad in ('ISS1','TIM1','ISS2','TIM2','ISS3','TIM3'):
        c=cells[pad];dims=c['thermal']['divisions'];size=c['geometry']['dimensions_m']
        for i,point in volume_centers(c):
            y,z=i%dims[1],i//dims[1]
            for axis,coord,n,cross in ((1,y,dims[1],size[2]/dims[2]),(2,z,dims[2],size[1]/dims[1])):
                for sign,boundary in (('-',0),('+',n-1)):
                    if coord!=boundary:continue
                    face=sign+'xyz'[axis];q=outside(point,face);label=f'escape-{pad}-{i}-{face}'
                    if q:passage(f'gap-{pad}-{i}',q['id'],gap*cross,label,face=face,point_m=point)
                    else:blocked.append(label)

    # A separate pressure/receiver pair for every casing voxel. Hole position
    # is no longer averaged away before the mass-transfer transaction.
    source=cells['C2'];dims=source['thermal']['divisions'];source_lo,source_hi=cell_box(source)
    for previous in gas['channels']:
        if previous['kind']!='rupture':continue
        face=previous['port_id'].split(':')[2];axis='xyz'.index(face[1])
        panel=shell_panel(source,shells['C2'],face)
        for i,point in volume_centers(panel):
            ijk=[max(0,min(n-1,int((p-a)/(b-a)*n))) for p,a,b,n in zip(point,source_lo,source_hi,dims)]
            ijk[axis]=0 if face[0]=='-' else dims[axis]-1
            source_i=old.index(*ijk,dims);label=previous['id']+f'-patch-{i}'
            if axis==0:
                pad='ISS2' if face[0]=='-' else 'TIM2';c=cells[pad];a,b=cell_box(c);n=c['thermal']['divisions']
                y,z=[max(0,min(n[k]-1,int((point[k]-a[k])/(b[k]-a[k])*n[k]))) for k in (1,2)]
                target=f'gap-{pad}-{y+n[1]*z}'
            else:
                q=outside(point,face)
                if not q:blocked.append(label);continue
                target=q['id']
            ch=copy.deepcopy(previous);ch.update(id=label,shell_thermal_volume=i,**{'from':'inside-C2'+(f'-{source_i}' if source_i else ''),'to':target})
            paths.append(ch);path_geometry.append(dict(id=label,point_m=point,face=face))

    # Retain h and physical contact area. Only gas-side neighbours are localized.
    contacts=[]
    for contact in gas['thermal_contacts']:
        if not contact['compartment'].startswith('near-'):
            contacts.append(copy.deepcopy(contact));continue
        port=ports[contact['port_id']];face=port['surface'];owner=port['cell_id']
        if ':case:' in owner:
            core=owner.split(':case:')[0];c=shell_panel(cells[core],shells[core],face)
        else:c=cells[owner]
        axis='xyz'.index(face[1]);n=c['thermal']['divisions'];edge=0 if face[0]=='-' else n[axis]-1
        for i,point in volume_centers(c):
            coord=(i//math.prod(n[:axis]))%n[axis]
            if coord!=edge:continue
            q=outside(point,face)
            if q:contacts.append(dict(contact,compartment=q['id'],thermal_volume=i))
            else:blocked.append('film-'+contact['port_id']+f'-{i}')

    ext=[n for n in nodes if not n['id'].startswith('inside-C2')]
    chamber['volume_m3']=total_v-sum(n['volume_m3'] for n in ext)
    chamber['initial_mass_kg']=total_m-sum(n['initial_mass_kg'] for n in ext)
    chamber['initial_temperature_k']=tref+(total_u-sum(n['initial_mass_kg']*cv*(n['initial_temperature_k']-tref) for n in ext))/(chamber['initial_mass_kg']*cv)
    if min(chamber['volume_m3'],chamber['initial_mass_kg'],chamber['initial_temperature_k'])<=0:raise ValueError('Void geometry exceeds chamber inventory')
    nodes.append(chamber);gas.update(compartments=nodes,channels=paths,thermal_contacts=contacts)
    if old.thermal_signature(s)!=frozen:raise ValueError('Thermal calibration changed')
    s['metadata']['local_mass_paths']=dict(thermal_sha256=frozen,source=old.SOURCE,
        extent_m=extent_m,radial_layers=radial_layers,obstacles=solids,void_cells=list(grid.cells.values()),
        paths=path_geometry,blocked_paths=blocked,gas_nodes=len(nodes),pressure_paths=len(paths),
        calibration_accepted=False,
        assumptions=['Frozen stack uses full-face solid layers; schematic is not a dimensioned CAD drawing',
                     'Holder geometry unknown: no additional holder obstruction unless supplied explicitly',
                     'Pad-face slits retain the prior unmeasured gap height and local LTE approximation',
                     'Casing voxel routes to the containing core/gap volume by centroid; no subvoxel hole direction',
                     'Pressure-only orifice network, no jet momentum or buoyancy',
                     'Only outer boundary faces reach the mixed 120 L chamber; no external sink'])
    return s
