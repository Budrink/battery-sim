"""Geometric intersections for a vertical cylindrical cell and gas boxes.

This module only computes geometry; it does not change a simulation scenario.
The caller must partition gas boxes without overlap and attach heat/mass links
to the corresponding physical surface and axial slice.
"""
import math

from scipy.integrate import quad


def _arc_measure(x0, x1, y0, y1, radius, face=None):
    """Angle of the circle circumference contained in an XY rectangle."""
    if face not in (None, '-x', '+x', '-y', '+y'):
        raise ValueError('Invalid cylindrical side')
    cuts = [i * math.pi / 4 for i in range(9)]
    for x in (x0, x1):
        if -radius < x < radius:
            a = math.acos(x / radius)
            cuts.extend((a, 2.0 * math.pi - a))
    for y in (y0, y1):
        if -radius < y < radius:
            a = math.asin(y / radius)
            cuts.extend((a % (2.0 * math.pi), (math.pi - a) % (2.0 * math.pi)))
    cuts = sorted(set(cuts))
    total = 0.0
    for a, b in zip(cuts, cuts[1:]):
        mid = (a + b) / 2.0
        x, y = radius * math.cos(mid), radius * math.sin(mid)
        side = ('+x' if x >= 0 else '-x') if abs(x) >= abs(y) else ('+y' if y >= 0 else '-y')
        if x0 <= x <= x1 and y0 <= y <= y1 and (face is None or side == face):
            total += b - a
    return total


def cylinder_side_area(low, high, center, radius, height, face):
    """Exact arc area of a given cylinder side inside an axis-aligned box."""
    z0, z1 = center[2]-height/2, center[2]+height/2
    dz = max(0., min(high[2],z1)-max(low[2],z0))
    return radius*dz*_arc_measure(low[0]-center[0],high[0]-center[0],
                                 low[1]-center[1],high[1]-center[1],radius,face)


def cylinder_box_intersection(low, high, center, radius, height):
    """Return displaced volume and lateral/cap contact areas in SI units.

    Cylinder axis is z; center is its 3-D midpoint. Cap areas are returned
    separately so a caller cannot silently count a shared cap twice.
    """
    if len(low) != 3 or len(high) != 3 or len(center) != 3:
        raise ValueError('Expected three-dimensional coordinates')
    values = (*low, *high, *center, radius, height)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite geometry')
    if radius <= 0 or height <= 0 or any(a >= b for a, b in zip(low, high)):
        raise ValueError('Invalid box or cylinder dimensions')
    x0, x1 = low[0] - center[0], high[0] - center[0]
    y0, y1 = low[1] - center[1], high[1] - center[1]
    z0, z1 = center[2] - height / 2, center[2] + height / 2
    za, zb = max(low[2], z0), min(high[2], z1)
    overlap_height = max(0.0, zb - za)
    a, b = max(-radius, x0), min(radius, x1)
    area = 0.0
    if b > a and y1 > -radius and y0 < radius:
        breaks = [a, b]
        for y in (y0, y1):
            if abs(y) < radius:
                x = math.sqrt(radius * radius - y * y)
                breaks.extend(v for v in (-x, x) if a < v < b)
        def span(x):
            y = math.sqrt(max(0.0, radius * radius - x * x))
            return max(0.0, min(y1, y) - max(y0, -y))
        area = sum(quad(span, left, right, epsabs=1e-15, epsrel=1e-10)[0]
                   for left, right in zip(sorted(breaks), sorted(breaks)[1:]))
    box_volume = math.prod(b - a for a, b in zip(low, high))
    solid = area * overlap_height
    arc = _arc_measure(x0, x1, y0, y1, radius)
    return {'box_volume_m3': box_volume, 'displaced_volume_m3': solid,
            'free_volume_m3': box_volume - solid,
            'lateral_contact_area_m2': radius * arc * overlap_height,
            'bottom_cap_area_m2': area if low[2] <= z0 < high[2] else 0.0,
            'top_cap_area_m2': area if low[2] < z1 <= high[2] else 0.0}


def aligned_axis_grid(length, max_step, required=()):
    """Partition a centered interval; every requested physical plane is a boundary."""
    if not math.isfinite(length) or not math.isfinite(max_step) or length <= 0 or max_step <= 0:
        raise ValueError('Invalid grid dimensions')
    planes = [-length / 2, length / 2]
    for point in required:
        if not math.isfinite(point) or point < planes[0] - 1e-12 or point > planes[1] + 1e-12:
            raise ValueError('Required plane outside domain')
        planes.append(max(-length / 2, min(length / 2, point)))
    planes = sorted(set(planes))
    result = [planes[0]]
    for a, b in zip(planes, planes[1:]):
        count = max(1, math.ceil((b - a) / max_step - 1e-12))
        result.extend(a + (b - a) * i / count for i in range(1, count + 1))
    return result


def aligned_cylinder_gas_grid(box_size, cylinders, max_step=.0325):
    """Create nonoverlapping cut gas boxes aligned to each cylinder's axial halves.

    Cylinders are dicts with id, center, radius and height; all geometry is in
    metres. Gas flow paths and thermal coupling are deliberately not inferred.
    """
    ids = [c['id'] for c in cylinders]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate cylinder id')
    for i, c in enumerate(cylinders):
        x, y, z = c['center']; r, h = c['radius'], c['height']
        if r <= 0 or h <= 0 or abs(x) + r > box_size[0]/2 + 1e-12 or abs(y) + r > box_size[1]/2 + 1e-12 or abs(z) + h/2 > box_size[2]/2 + 1e-12:
            raise ValueError('Cylinder does not fit box')
        for other in cylinders[:i]:
            ox, oy, oz = other['center']
            z_overlap = abs(z-oz) < (h+other['height'])/2 - 1e-12
            if z_overlap and math.hypot(x-ox, y-oy) < r+other['radius']-1e-12:
                raise ValueError('Overlapping cylinders')
    axial_planes = [v for c in cylinders for v in (c['center'][2]-c['height']/2,
                    c['center'][2], c['center'][2]+c['height']/2)]
    axes = [aligned_axis_grid(box_size[0], max_step), aligned_axis_grid(box_size[1], max_step),
            aligned_axis_grid(box_size[2], max_step, axial_planes)]
    result = []
    for i, (x0, x1) in enumerate(zip(axes[0], axes[0][1:])):
        for j, (y0, y1) in enumerate(zip(axes[1], axes[1][1:])):
            for k, (z0, z1) in enumerate(zip(axes[2], axes[2][1:])):
                low, high = (x0,y0,z0), (x1,y1,z1)
                box_volume = (x1-x0)*(y1-y0)*(z1-z0)
                displaced, contacts = 0., []
                for c in cylinders:
                    hit = cylinder_box_intersection(low, high, c['center'], c['radius'], c['height'])
                    displaced += hit['displaced_volume_m3']
                    if hit['lateral_contact_area_m2'] > 1e-18 or hit['bottom_cap_area_m2'] > 1e-18 or hit['top_cap_area_m2'] > 1e-18:
                        contacts.append({'cylinder': c['id'], 'axial_half': 0 if (z0+z1)/2 < c['center'][2] else 1,
                            'lateral_contact_area_m2': hit['lateral_contact_area_m2'],
                            'bottom_cap_area_m2': hit['bottom_cap_area_m2'], 'top_cap_area_m2': hit['top_cap_area_m2']})
                free = box_volume-displaced
                if free < -1e-14:
                    raise ValueError('Negative free volume')
                result.append({'id': (i,j,k), 'low_m': low, 'high_m': high,
                    'free_volume_m3': max(0.,free), 'contacts': contacts})
    return axes, result


def gas_face_area(low, high, axis, position, cylinders):
    """Open cross-section of a box face after blocking it with solid cylinders."""
    if axis not in (0, 1, 2):
        raise ValueError('Invalid face axis')
    u, v = (axis + 1) % 3, (axis + 2) % 3
    area = (high[u]-low[u])*(high[v]-low[v])
    blocked = 0.
    for c in cylinders:
        center, r, h = c['center'], c['radius'], c['height']
        if axis == 2:
            if not center[2]-h/2 <= position <= center[2]+h/2:
                continue
            # An interior slice returns the XY intersection area independent of z.
            zl, zh = center[2]-h/4, center[2]+h/4
            lo, hi = (low[0],low[1],zl), (high[0],high[1],zh)
            blocked += cylinder_box_intersection(lo,hi,center,r,h)['displaced_volume_m3']/(zh-zl)
        else:
            delta = position-center[axis]
            if abs(delta) >= r:
                continue
            half = math.sqrt(r*r-delta*delta)
            other = 1-axis
            span = max(0., min(high[other],center[other]+half)-max(low[other],center[other]-half))
            zspan = max(0., min(high[2],center[2]+h/2)-max(low[2],center[2]-h/2))
            blocked += span*zspan
    if blocked > area + 1e-12:
        raise ValueError('Cylinder face coverage exceeds cell face')
    return max(0.,area-blocked)


def aligned_gas_faces(cells, cylinders):
    """Build undirected face links for a previously built aligned cut grid."""
    by_id = {c['id']: c for c in cells}
    if len(by_id) != len(cells):
        raise ValueError('Duplicate gas cell id')
    links = []
    for c in cells:
        for axis in range(3):
            target = list(c['id']); target[axis] += 1; target = tuple(target)
            if target not in by_id or c['free_volume_m3'] <= 0 or by_id[target]['free_volume_m3'] <= 0:
                continue
            area = gas_face_area(c['low_m'], c['high_m'], axis, c['high_m'][axis], cylinders)
            if area > 1e-18:
                links.append({'from': c['id'], 'to': target, 'axis': axis, 'area_m2': area})
    return links
