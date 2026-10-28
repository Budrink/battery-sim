"""Conservative, nearest-visible can radiation in the retained prismatic geometry.

This is a deliberately incomplete direct-view model, not a radiosity solution:
only opposing overlapping rectangles are paired. No flame radiation, reflected
rays, can/ABS exchange, or rays outside the common projection are invented.
The shared effective emissivity is UNMEASURED and is not a battery heat source.
"""
from copy import deepcopy
import math


def rectangle_view_factor(width, height, gap):
    """Exact identical coaxial finite rectangles; dimensions in any common unit.

    Reference form: NASA/CP-2005-213431, appendix A finite-rectangle integral;
    Hamilton/Morgan identical-rectangle specialization. Ai Fij = Aj Fji.
    """
    if any(not math.isfinite(x) or x <= 0 for x in (width, height, gap)):
        raise ValueError('Positive finite rectangle dimensions required')
    x,y=width/gap,height/gap
    value=2./(math.pi*x*y)*(
        .5*math.log1p(x*x*y*y/(1.+x*x+y*y))
        +x*math.sqrt(1.+y*y)*math.atan(x/math.sqrt(1.+y*y))
        +y*math.sqrt(1.+x*x)*math.atan(y/math.sqrt(1.+x*x))
        -x*math.atan(x)-y*math.atan(y))
    if not 0.<value<=1.+1e-12:raise ValueError('Invalid view factor')
    return min(1.,value)


def add_internal_radiation(base, effective_emissivity):
    if not math.isfinite(effective_emissivity) or not 0<=effective_emissivity<=1:
        raise ValueError('Effective emissivity must be in [0,1]')
    if any('radiation' in e for e in base.get('interfaces',[])):
        raise ValueError('Finite radiation already configured')
    scene=deepcopy(base)
    cells={c['id']:c for c in scene['cells'] if c.get('role')=='battery'}
    shells={s['contents_id']:s for s in scene['shells']}
    boxes={}
    for name,c in cells.items():
        if c['geometry']['kind']!='prismatic' or c['pose'].get('orientation_xyzw',[0,0,0,1])!=[0,0,0,1]:
            raise ValueError('Radiation reconstruction requires axis-aligned prismatic cans')
        dims=c['geometry']['dimensions_m'];pos=c['pose']['position_m'];t=shells[name]['thickness_m']
        boxes[name]=[(pos[k]-dims[k]/2.-t,pos[k]+dims[k]/2.+t) for k in range(3)]
    # Heater foils and fixture walls are opaque too. In particular the source's
    # -x foil must NOT be bypassed by a newly invented direct can-to-can ray.
    for c in scene['cells']:
        if c['id'] in cells:continue
        if c['geometry']['kind']!='prismatic' or c['pose'].get('orientation_xyzw',[0,0,0,1])!=[0,0,0,1]:
            raise ValueError('Opaque fixture requires explicit axis-aligned geometry')
        dims=c['geometry']['dimensions_m'];pos=c['pose']['position_m']
        boxes[c['id']]=[(pos[k]-dims[k]/2.,pos[k]+dims[k]/2.) for k in range(3)]
    # Iterate all geometric pairs, not a fitted list of desired cascade recipients.
    records=[];face_rectangles={}
    for ai,a in enumerate(sorted(cells)):
        for b in sorted(cells)[ai+1:]:
            for axis in (0,1,2):
                ca,cb=cells[a],cells[b];pa,pb=ca['pose']['position_m'],cb['pose']['position_m']
                sign=1 if pb[axis]>pa[axis] else -1
                xa=boxes[a][axis][int(sign>0)];xb=boxes[b][axis][int(sign<0)]
                gap=(xb-xa)*sign
                if gap<=0.:continue
                u,v=(axis+1)%3,(axis+2)%3
                # Match shell.cpp's corner ownership exactly, not full bounding boxes.
                def bounds(name,k):
                    c=cells[name];t=shells[name]['thickness_m']
                    width=c['geometry']['dimensions_m'][k]+(2*t if k>axis else 0.)
                    center=c['pose']['position_m'][k]
                    return (center-width/2.,center+width/2.)
                rect=[max(bounds(a,u)[0],bounds(b,u)[0]),min(bounds(a,u)[1],bounds(b,u)[1]),
                      max(bounds(a,v)[0],bounds(b,v)[0]),min(bounds(a,v)[1],bounds(b,v)[1])]
                width,height=rect[1]-rect[0],rect[3]-rect[2]
                if min(width,height)<=1e-12:continue
                beam=[None]*3;beam[axis]=sorted([xa,xb]);beam[u]=rect[:2];beam[v]=rect[2:]
                # Every included ray stays inside this convex prism. A blocker
                # excludes the entire pair, conservatively, even if only partial.
                blocked=any(all(min(box[k][1],beam[k][1])-max(box[k][0],beam[k][0])>1e-10 for k in range(3))
                            for name,box in boxes.items() if name not in (a,b))
                if blocked:continue
                ports=[]
                for name,direction,plane in ((a,sign,xa),(b,-sign,xb)):
                    face=('+' if direction>0 else '-')+'xyz'[axis]
                    key=(name,face)
                    for old in face_rectangles.get(key,[]):
                        if min(old[1],rect[1])-max(old[0],rect[0])>1e-10 and min(old[3],rect[3])-max(old[2],rect[2])>1e-10:
                            raise ValueError('Direct-view patches overlap on '+str(key))
                    face_rectangles.setdefault(key,[]).append(rect)
                    ident=f'can-radiation:{a}:{b}:{name}:{face}'
                    point=[0.]*3;point[axis]=plane;point[u]=(rect[0]+rect[1])/2.;point[v]=(rect[2]+rect[3])/2.
                    normal=[0.]*3;normal[axis]=direction
                    scene['ports'].append(dict(id=ident,cell_id=name+':case:'+face,kind='contact',surface=face,
                        area_m2=width*height,rectangle_m=[width,height],centroid_m=point,normal=normal))
                    ports.append(ident)
                factor=rectangle_view_factor(width,height,gap);area=width*height*factor
                scene['interfaces'].append(dict(id=f'can-radiation:{a}:{b}:{axis}',port_a=ports[0],port_b=ports[1],
                    interface_material='unmeasured-direct-can-radiation',radiation=dict(
                        exchange_area_m2=area,effective_emissivity=effective_emissivity,temperature_limit_k=4000.)))
                records.append(dict(a=a,b=b,axis=axis,gap_m=gap,rectangle_m=[width,height],
                                    view_factor=factor,exchange_area_m2=area))
    if not records:raise ValueError('No unobstructed opposing cans')
    scene['metadata']['internal_radiation']=dict(effective_emissivity=effective_emissivity,
        emissivity_measured=False,pairs=records,scope=__doc__,calibrated=False,validation_passed=False,
        radiation_law_reference='https://ntrs.nasa.gov/api/citations/20050206383/downloads/20050206383.pdf')
    return scene
