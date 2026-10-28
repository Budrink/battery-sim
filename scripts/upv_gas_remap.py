"""Conservative geometry overlap for a CLOSED pre-rupture checkpoint.

Old v1 'chamber' is the unresolved complement of its local void boxes. The
v2 grid resolves that complement explicitly. Never remap pores into free gas.
"""
import math


def overlap(a,b):
    return math.prod(max(0.,min(a['hi'][k],b['hi'][k])-max(a['lo'][k],b['lo'][k]))for k in range(3))


def unvented_remap(source,target):
    old=source['gas_network']['compartments'];new=target['gas_network']['compartments']
    oldbox=source['metadata']['gas_geometry']['void_boxes']
    newbox=target['metadata']['gas_geometry']['void_boxes']
    oldid={n['id']:i for i,n in enumerate(old)}
    hub=oldid.get('chamber') if 'chamber' not in oldbox else None
    result=[]
    for j,n in enumerate(new):
        ident=n['id'];volume=n['volume_m3']
        if ident not in newbox:
            if ident not in oldid or ident in oldbox:raise ValueError('Nonspatial node changed: '+ident)
            i=oldid[ident]
            if old[i]['volume_m3']!=volume:raise ValueError('Pore/slit volume changed: '+ident)
            result.append(dict(from_=i,to=j,fraction=1.));continue
        covered=0.
        for name,box in oldbox.items():
            v=overlap(box,newbox[ident])
            if v>0.:
                result.append(dict(from_=oldid[name],to=j,fraction=v/old[oldid[name]]['volume_m3']))
                covered+=v
        remaining=volume-covered
        if remaining>max(1e-16,volume*1e-10):
            if hub is None:raise ValueError('New spatial volume outside source chamber')
            result.append(dict(from_=hub,to=j,fraction=remaining/old[hub]['volume_m3']))
        elif remaining < -max(1e-16,volume*1e-10):raise ValueError('Overlapping source boxes')
    totals=[0.]*len(old);received=[0.]*len(new)
    for e in result:
        totals[e['from_']]+=e['fraction'];received[e['to']]+=e['fraction']*old[e['from_']]['volume_m3']
        e['from']=e.pop('from_')
    if any(abs(v-1.)>1e-10 for v in totals):raise ValueError('Source inventory not fully mapped')
    if any(abs(v-n['volume_m3'])>1e-10*n['volume_m3']+1e-16 for v,n in zip(received,new)):
        raise ValueError('Target free volume not fully covered')
    return result
