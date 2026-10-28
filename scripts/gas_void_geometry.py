"""Axis-aligned free-volume topology. No flow solver and no fitted physics.

Solid boxes are excluded exactly using their faces as grid cuts. A connection
exists only across a shared face; touching at an edge/corner is insufficient.
Outer faces are exposed to the caller; they need not be gas outlets.
"""
from bisect import bisect_right
from itertools import product
import math


def cuts(values):
    return sorted(set(round(float(v), 12) for v in values))


def contains(box, point):
    return all(lo <= p <= hi for lo, hi, p in zip(*box, point))


class VoidGrid:
    def __init__(self, bounds, obstacles, divisions):
        self.bounds = bounds
        if any(not math.isfinite(v) for row in bounds for v in row) or any(b <= a for a, b in zip(*bounds)):
            raise ValueError('Invalid free-space envelope')
        for lo, hi in obstacles:
            if any(not math.isfinite(v) for row in (lo, hi) for v in row) or any(b <= a for a, b in zip(lo, hi)):
                raise ValueError('Invalid solid obstacle')
        if any(not math.isfinite(v) or v < bounds[0][k] or v > bounds[1][k]
               for k in range(3) for v in divisions[k]):
            raise ValueError('Grid cuts outside envelope')
        # Round only interior construction cuts. Preserve exact endpoints and
        # reject invalid input BEFORE deduplication; never clip an outside cut.
        self.cuts=[]
        for k in range(3):
            a,b=bounds[0][k],bounds[1][k]
            raw=[*divisions[k],*[v[k]for box in obstacles for v in box]]
            interior=cuts(v for v in raw if a<v<b)
            self.cuts.append([a,*[v for v in interior if a<v<b],b])
        self.cells = {}
        for ijk in product(*(range(len(c)-1) for c in self.cuts)):
            lo = [self.cuts[k][i] for k, i in enumerate(ijk)]
            hi = [self.cuts[k][i+1] for k, i in enumerate(ijk)]
            center = [(a+b)/2 for a, b in zip(lo, hi)]
            if any(contains(box, center) for box in obstacles):
                continue
            self.cells[ijk] = dict(id='void-'+'-'.join(map(str, ijk)), lo=lo, hi=hi,
                                   center=center, volume=math.prod(b-a for a, b in zip(lo, hi)))

    def locate(self, point):
        ijk = tuple(bisect_right(c, p)-1 for c, p in zip(self.cuts, point))
        return self.cells.get(ijk)

    def faces(self):
        """(cell, neighbour-or-None, area, axis, sign), each passage once."""
        for ijk, c in self.cells.items():
            size = [b-a for a, b in zip(c['lo'], c['hi'])]
            for axis in range(3):
                area = size[(axis+1)%3]*size[(axis+2)%3]
                j = list(ijk);j[axis] += 1
                if tuple(j) in self.cells:
                    yield c, self.cells[tuple(j)], area, axis, 1
                for sign, edge in ((-1, 0), (1, len(self.cuts[axis])-2)):
                    if ijk[axis] == edge:
                        yield c, None, area, axis, sign


def cell_box(cell):
    pos, size = cell['pose']['position_m'], cell['geometry']['dimensions_m']
    return ([p-d/2 for p, d in zip(pos, size)], [p+d/2 for p, d in zip(pos, size)])


def volume_centers(cell):
    dims = cell['thermal']['divisions'];lo, hi = cell_box(cell)
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                ijk = [x, y, z]
                yield x+dims[0]*(y+dims[1]*z), [(a+(i+.5)*(b-a)/n) for a, b, i, n in zip(lo, hi, ijk, dims)]


def shell_panel(core, shell, face):
    """Match native shell.cpp corner ownership and x-fastest indexing."""
    import copy
    panel = copy.deepcopy(core);axis = 'xyz'.index(face[1]);sign = 1 if face[0]=='+' else -1
    t = shell['thickness_m'];size = core['geometry']['dimensions_m']
    panel['id'] = core['id']+':case:'+face
    panel['geometry']['dimensions_m'] = [t if k==axis else d+(2*t if k>axis else 0) for k, d in enumerate(size)]
    panel['pose']['position_m'][axis] += sign*(size[axis]+t)/2
    panel['thermal']['divisions'] = list(shell.get('surface_divisions', core['thermal']['divisions']))
    panel['thermal']['divisions'][axis] = 1
    return panel
