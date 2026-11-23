"""Small local WGS84 polygon membership, including boundary points.

Input coordinates are (longitude, latitude). Antimeridian polygons are unsupported.
Barrier distance must be supplied by a ranging sensor, not derived from zone edges.
"""
import math

def inside_polygon(longitude, latitude, ring):
    if len(ring) < 4 or ring[0] != ring[-1]:
        raise ValueError('A closed polygon ring with at least three vertices is required')
    if not all(math.isfinite(v) for p in ring for v in p):
        raise ValueError('Invalid polygon coordinates')
    if not math.isfinite(longitude) or not math.isfinite(latitude):
        raise ValueError('Missing GPS position')
    if max(p[0] for p in ring)-min(p[0] for p in ring) > 180:
        raise ValueError('Antimeridian polygons unsupported')
    inside = False
    x,y = longitude,latitude
    for (x1,y1),(x2,y2) in zip(ring,ring[1:]):
        cross = (x-x1)*(y2-y1)-(y-y1)*(x2-x1)
        if abs(cross) < 1e-12 and min(x1,x2) <= x <= max(x1,x2) and min(y1,y2) <= y <= max(y1,y2):
            return True
        if (y1 > y) != (y2 > y) and x < (x2-x1)*(y-y1)/(y2-y1)+x1:
            inside = not inside
    return inside
