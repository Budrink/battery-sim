"""Geometry-only graded Cartesian air around an axis-aligned closed envelope.

No chemical rates, temperatures fitted to data, or synthetic solid shortcuts.
The physical envelope, halo and total available air volume stay fixed when h
changes. Connected face areas and distances are geometry, not fit coefficients.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class AirVoxel:
    key: tuple
    low: np.ndarray
    high: np.ndarray

    @property
    def name(self): return 'air-'+'-'.join(map(str,self.key))
    @property
    def center(self): return (self.low+self.high)/2
    @property
    def volume(self): return float(np.prod(self.high-self.low))


def exterior_grid(envelope,spacing,halo=.04,total_air_m3=1.,*,vent_spacing=None,vent_half_width=.025):
    envelope=np.asarray(envelope,dtype=float)
    if envelope.shape!=(3,) or np.any(envelope<=0.) or not 0<spacing<=.1 or not halo>0. or not total_air_m3>0.:
        raise ValueError('Invalid physical air-grid geometry')
    outer=(total_air_m3+float(np.prod(envelope)))**(1/3)/2
    if np.any(envelope/2+halo>=outer):raise ValueError('Halo does not fit ambient domain')
    if vent_spacing is not None and (not 0.<vent_spacing<=spacing or not 0.<vent_half_width<min(envelope[[0,2]])/2):
        raise ValueError('Invalid local aperture refinement')
    axes=[]
    for direction,size in enumerate(envelope):
        count=max(1,math.ceil(size/spacing));extra=max(1,math.ceil(halo/spacing));half=size/2
        axis=np.unique(np.concatenate(([-outer],np.linspace(-half-halo,-half,extra+1),
            np.linspace(-half,half,count+1),np.linspace(half,half+halo,extra+1),[outer])))
        if vent_spacing is not None:
            # Nested local refinement: retain EVERY coarse plane and split
            # selected coarse cells. Re-anchoring a lattice at +/- jet width
            # used to move even unchanged cells and relocate the jet relative
            # to voxel centers. That aliases nonlinear reaction/entrainment.
            # The Cartesian mesh stays conforming (no skew octree TPFA).
            parts=[]
            for low,high in zip(axis,axis[1:]):
                middle=(low+high)/2
                fine=(half+1e-12<abs(middle)<half+halo+1e-12) if direction==1 else abs(middle)<vent_half_width
                divisions=max(1,math.ceil((high-low)/vent_spacing-1e-10)) if fine else 1
                parts.append(np.linspace(low,high,divisions+1))
            axis=np.unique(np.concatenate(parts))
        axes.append(axis)
    voxels={}
    for iz in range(len(axes[2])-1):
        for iy in range(len(axes[1])-1):
            for ix in range(len(axes[0])-1):
                key=(ix,iy,iz);low=np.array([axes[a][key[a]] for a in range(3)])
                high=np.array([axes[a][key[a]+1] for a in range(3)])
                if np.all(abs((low+high)/2)<envelope/2-1e-12):continue
                voxels[key]=AirVoxel(key,low,high)
    faces=[]
    for key,v in voxels.items():
        for axis in range(3):
            other=tuple(q+(a==axis) for a,q in enumerate(key))
            if other not in voxels:continue
            w=voxels[other];u,z=(axis+1)%3,(axis+2)%3
            area=(v.high[u]-v.low[u])*(v.high[z]-v.low[z])
            faces.append((v.name,w.name,float(area),float(w.center[axis]-v.center[axis]),axis))
    if not math.isclose(sum(v.volume for v in voxels.values()),total_air_m3,rel_tol=2e-13):
        raise ValueError('Exterior voxel volumes do not close physical air inventory')
    return voxels,faces


def rectangle_overlap(a,b):
    return max(0.,min(a[1],b[1])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[2],b[2]))


def rectangle_intersection(a,b):
    return [max(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),min(a[3],b[3])]


def face_tiles(cell,face):
    axis='xyz'.index(face[1]);sign=1 if face[0]=='+' else -1
    dims=np.array(cell['geometry']['dimensions_m']);div=np.array(cell['thermal']['divisions'])
    center=np.array(cell['pose']['position_m']);u,v=(axis+1)%3,(axis+2)%3
    for j in range(int(div[v])):
        for i in range(int(div[u])):
            low_u=center[u]-dims[u]/2+i*dims[u]/div[u];low_v=center[v]-dims[v]/2+j*dims[v]/div[v]
            rect=[low_u,low_u+dims[u]/div[u],low_v,low_v+dims[v]/div[v]]
            indices=[0,0,0];indices[axis]=int(div[axis]-1) if sign>0 else 0;indices[u]=i;indices[v]=j
            index=indices[0]+int(div[0])*(indices[1]+int(div[1])*indices[2])
            point=center.copy();point[axis]+=sign*dims[axis]/2
            point[u]=(rect[0]+rect[1])/2;point[v]=(rect[2]+rect[3])/2
            yield index,rect,point


def add_face_port(scene,cell,face,index,rect,point,label):
    ident=f'gasmesh:{cell["id"]}:{label}:{index}'
    if not any(p['id']==ident for p in scene['ports']):
        axis='xyz'.index(face[1]);normal=[0.,0.,0.];normal[axis]=1. if face[0]=='+' else -1.
        width,height=rect[1]-rect[0],rect[3]-rect[2]
        scene['ports'].append(dict(id=ident,cell_id=cell['id'],kind='contact',surface=face,
            area_m2=width*height,rectangle_m=[width,height],centroid_m=point.tolist(),normal=normal))
    return ident
