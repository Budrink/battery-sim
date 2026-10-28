"""Diagnostic removal: inactive solid energy escrows, physically empty slots."""
import copy
import math
from gas_void_geometry import cell_box
from upv_solid_geometry import physical_obstacles

PADS=('ISS1','TIM1','ISS2','TIM2','ISS3','TIM3','ISS4')
CONDUCTANCES=('background_conductance_w_per_k','active_conductance_w_per_k',
    'destroyed_background_conductance_w_per_k','destroyed_active_conductance_w_per_k')

def configure(scene,removed):
    removed=list(removed)
    if len(set(removed))!=len(removed) or set(removed)-set(PADS):raise ValueError('Unknown/duplicate ablated pad')
    ports={p['id']:p for p in scene['ports']};objects={c['id']:c for c in scene['cells']}
    if any(objects[n]['material_id']not in ('ISS','TIM')for n in removed):raise ValueError('Ablation must target pads')
    disconnected=[]
    for e in scene['interfaces']:
        if any(ports[e[k]]['cell_id']in removed for k in ('port_a','port_b')):
            if e['law']['kind']!='directed_geometry':raise ValueError('Unexpected pad contact')
            for k in CONDUCTANCES:e['law'][k]=0.
            disconnected.append(e['id'])
    scene['metadata']['pad_ablation']=dict(removed=removed,disabled_interfaces=disconnected,
        mode='post-warmup isolated energy escrows; gas occupies removed pad slots',
        external_free_volume_m3=.120,calibrated=False)

def couple_empty_slots(scene,grid,objects,ports,removed,pads,passage,film):
    for ident in sorted(removed):
        if ident not in pads:continue
        c,own,_case=pads[ident];lo,hi=cell_box(c);ny,nz=c['thermal']['divisions'][1:]
        for i in range(ny*nz):
            y,z=i%ny,i//ny
            point=[(lo[0]+hi[0])/2,lo[1]+(y+.5)*(hi[1]-lo[1])/ny,lo[2]+(z+.5)*(hi[2]-lo[2])/nz]
            receiver=grid.locate(point)
            if receiver is None:raise ValueError('Removed pad slot is still occupied')
            area=(hi[1]-lo[1])*(hi[2]-lo[2])/(ny*nz)
            passage(f'ablated-face-{ident}-{i}',f'gap-{ident}-{i}',receiver['id'],area,abs(point[0]-own['centroid_m'][0])+.0001)
    # Newly exposed back plates use the frozen gas-film coefficient, no new h.
    for ident in sorted(removed):
        slot=cell_box(objects[ident])
        for cell in grid.cells.values():
            if not all(a-1e-12<=p<=b+1e-12 for a,b,p in zip(*slot,cell['center'])):continue
            for name,solid in objects.items():
                if name in removed or name=='chamber-wall' or solid['role']=='battery':continue
                a,b=cell_box(solid)
                for sign,plane in (('-',cell['lo'][0]),('+',cell['hi'][0])):
                    if abs((b[0]if sign=='-'else a[0])-plane)>1e-11:continue
                    overlap=[(max(a[k],cell['lo'][k]),min(b[k],cell['hi'][k]))for k in (1,2)]
                    if any(hi<=lo for lo,hi in overlap):continue
                    face='+x'if sign=='-'else'-x';port=copy.deepcopy(ports[name+':'+face])
                    port['id']=f'ablation-contact:{ident}:{name}:{cell["id"]}'
                    port['rectangle_m']=[hi-lo for lo,hi in overlap];port['area_m2']=math.prod(port['rectangle_m'])
                    port['centroid_m']=[plane,*[(lo+hi)/2 for lo,hi in overlap]]
                    scene['ports'].append(port);film(cell['id'],port['id'])
