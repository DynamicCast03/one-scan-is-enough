import math


def point_segment_distance(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(p[0]), float(p[1])
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay

    vv = float(vx * vx + vy * vy)
    if vv <= 0.0:
        return float(math.hypot(wx, wy))
    t = float((wx * vx + wy * vy) / vv)
    if t <= 0.0:
        return float(math.hypot(wx, wy))
    if t >= 1.0:
        return float(math.hypot(px - bx, py - by))
    projx = ax + t * vx
    projy = ay + t * vy
    return float(math.hypot(px - projx, py - projy))
