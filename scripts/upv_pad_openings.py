"""Expose real free perimeter bands; never connect gas through an intact pad."""
import copy
import math
from gas_void_geometry import cell_box


def exposed_contacts(scene,grid,objects,ports,shells,film):
    clearance=scene['metadata']['gas_geometry_controls']['pad_edge_clearance_m']
    slots=[]
    for c in objects.values():
        if c['material_id']not in ('ISS','TIM'):continue
        nominal=copy.deepcopy(c)
        for k in (1,2):nominal['geometry']['dimensions_m'][k]+=2*clearance
        slots.append((c['id'],cell_box(nominal)))
    surfaces=[]
    for name,c in objects.items():
        if name=='chamber-wall':continue
        physical=copy.deepcopy(c)
        if name in shells:
            physical['geometry']['dimensions_m']=[v+2*shells[name]['thickness_m']for v in c['geometry']['dimensions_m']]
        a,b=cell_box(physical)
        for axis in range(3):
            for sign,plane in (('-',a[axis]),('+',b[axis])):
                face=sign+'xyz'[axis];port=ports.get(name+':'+face)
                if port:surfaces.append((name,axis,plane,face,a,b,port))
    galleries=[];contacts=[]
    for cell in grid.cells.values():
        slot=next((name for name,(a,b)in slots if all(a[k]-1e-12<=cell['center'][k]<=b[k]+1e-12 for k in range(3))),None)
        if slot is None:continue
        galleries.append(cell['id'])
        for name,axis,plane,face,a,b,base in surfaces:
            # Solid outward face and gas inward face coincide. The overlap is
            # geometric, not a full-face proxy attached to a neighbouring node.
            boundary=cell['lo'][axis]if face[0]=='+'else cell['hi'][axis]
            if abs(boundary-plane)>1e-11:continue
            u,v=(axis+1)%3,(axis+2)%3
            overlap=[(max(a[k],cell['lo'][k]),min(b[k],cell['hi'][k]))for k in (u,v)]
            if any(hi-lo<=1e-12 for lo,hi in overlap):continue
            port=copy.deepcopy(base);port['id']=f'opening-contact:{slot}:{name}:{face}:{cell["id"]}'
            port['rectangle_m']=[hi-lo for lo,hi in overlap];port['area_m2']=math.prod(port['rectangle_m'])
            center=list(cell['center']);center[axis]=plane
            center[u],center[v]=[(lo+hi)/2 for lo,hi in overlap];port['centroid_m']=center
            scene['ports'].append(port);film(cell['id'],port['id'])
            contacts.append(dict(gas=cell['id'],port=port['id'],solid=port['cell_id'],area_m2=port['area_m2']))
    if not galleries or not contacts:raise ValueError('Open pad edges produced no physical gas access')
    scene['metadata']['pad_openings']=dict(mode='recessed pad perimeter, retained solid centre',
        clearance_m=clearance,gallery_nodes=galleries,exposed_contacts=contacts,
        measured=False,status='Explicit geometry hypothesis, not a fitted experiment')
