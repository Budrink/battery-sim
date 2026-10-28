"""Volume-constrained graded chamber; no instantaneous mixed reservoir.

Only chamber volume is published. Equal outer clearances are an EXPLICIT shape
prior, not a claim about measured facility dimensions. Near-grid cuts remain
independent of wall position; changing near resolution cannot create an outlet.
"""
import math
from gas_void_geometry import VoidGrid


def graded_chamber(lo,hi,obstacles,divisions,near_extent,free_volume):
    occupied=sum(math.prod(b-a for a,b in zip(*box))for box in obstacles)
    if len(obstacles)==2:
        occupied-=math.prod(max(0.,min(obstacles[0][1][k],obstacles[1][1][k])-
                               max(obstacles[0][0][k],obstacles[1][0][k]))for k in range(3))
    def volume(e):return math.prod(b-a+2*e for a,b in zip(lo,hi))-occupied
    lower,upper=near_extent,1.
    if volume(lower)>=free_volume or volume(upper)<free_volume:
        raise ValueError('Near field does not fit the volume-constrained chamber')
    for _ in range(70):
        mid=(lower+upper)/2
        if volume(mid)<free_volume:lower=mid
        else:upper=mid
    extent=(lower+upper)/2
    bounds=([v-extent for v in lo],[v+extent for v in hi])
    cuts=[list(row)+[lo[k]-near_extent,hi[k]+near_extent]for k,row in enumerate(divisions)]
    grid=VoidGrid(bounds,obstacles,cuts)
    # No compartment named 'chamber': that would let old consumers silently
    # mistake one corner's pressure for the chamber pressure observer.
    wall_area={}
    for cell,other,area,_axis,_sign in grid.faces():
        if other is None:wall_area[cell['id']]=wall_area.get(cell['id'],0.)+area
    return grid,wall_area,dict(bounds_m=bounds,equal_clearance_m=extent,
        shape_status='Unmeasured equal-clearance box constrained by the published free volume',
        resolved_free_volume_m3=sum(v['volume']for v in grid.cells.values()))
