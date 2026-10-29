"""A* planning on the planning map, the single feasible region of a scene.

The planning map is the navigable BEV resampled to 5 mm cells, eroded by the
safety margin and kept to its largest components. Its free cells are the
only safety constraint: A* searches all of them, simplified chords stay
inside them, and closed-loop runs and rewards use the same map. Planner
distances are in cells of this map.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from omegaconf import DictConfig
from scipy import ndimage

from one_scan_is_enough.geometry.segments import point_segment_distance
from one_scan_is_enough.planning import _native
from one_scan_is_enough.scene import BevMap


def planning_map(bev: BevMap, config: DictConfig) -> BevMap:
    """Return the planning map of a navigable BEV.

    Each BEV cell becomes scale x scale planning cells centered on it. Cells
    within clearance_m of blocked space are blocked, which is the only safety
    margin of planning, runs and rewards, and the largest free components are
    kept by filter_connected_components.
    """
    scale = round(bev.resolution / config.resolution)
    if not np.isclose(scale * config.resolution, bev.resolution):
        raise ValueError("The BEV resolution must be a multiple of the planning resolution.")
    raw = np.repeat(np.repeat(bev.navigable, scale, axis=0), scale, axis=1)
    origin = bev.origin_xz - (scale - 1) / 2 * config.resolution
    clearance_px = max(1, int(np.ceil(config.clearance_m / config.resolution)))
    safe = ((raw == 255) & (compute_distance_field(raw) > clearance_px)).astype(np.uint8) * 255
    navigable = filter_connected_components(safe, config.top_k, config.top_p)
    return BevMap(raw, navigable, origin, config.resolution)


def compute_distance_field(bev_image: np.ndarray) -> np.ndarray:
    """Distance in cells from each free cell to the nearest blocked cell."""
    return ndimage.distance_transform_edt((bev_image == 255).astype(np.uint8))


def compute_cost_map(distance_field, distance_weight, min_clearance):
    """Per-cell obstacle cost growing below min_clearance cells; blocked cells are forbidden."""
    safe_distance = np.maximum(distance_field, 0.1)
    cost_map = distance_weight / safe_distance
    danger_mask = distance_field < min_clearance
    cost_map[danger_mask] *= min_clearance / (safe_distance[danger_mask] + 0.1)
    cost_map[distance_field == 0] = np.inf
    return cost_map


def filter_connected_components(mask_255: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
    """Keep the largest components, at most top_k, until they cover top_p of the area."""
    top_k = max(1, int(top_k))
    top_p = min(max(top_p, 0.0), 1.0)
    binary = (mask_255 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 2:
        return (binary * 255).astype(np.uint8)
    areas = sorted(
        ((int(i), int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, num)),
        key=lambda x: x[1],
        reverse=True,
    )
    total_area = sum(a for _, a in areas)
    if total_area <= 0:
        return np.zeros_like(mask_255, dtype=np.uint8)
    keep_labels = []
    kept_area = 0.0
    for label, area in areas:
        if len(keep_labels) >= top_k:
            break
        if keep_labels and kept_area / total_area >= top_p:
            break
        keep_labels.append(label)
        kept_area += area
    kept = np.zeros_like(binary, dtype=np.uint8)
    for label in keep_labels:
        kept[labels == label] = 255
    return kept


@dataclass
class PlannerCache:
    distance_field: np.ndarray  # (H, W) cells to the nearest blocked cell.
    free_mask: np.ndarray  # (H, W) cells A* may enter, the free cells of the map.
    cost_map: np.ndarray  # (H, W) obstacle cost of entering a cell.
    tolerance_px: float  # Deviation tolerance of path simplification.


def planner_cache(bev: BevMap, config: DictConfig) -> PlannerCache:
    """Distances, free cells and costs of a planning map for a planner configuration.

    config holds distance_weight, min_clearance_m and rdp_tolerance_m.
    """
    distance_field = compute_distance_field(bev.navigable)
    min_clearance_px = max(1, int(config.min_clearance_m / bev.resolution))
    return PlannerCache(
        distance_field,
        bev.navigable == 255,
        compute_cost_map(distance_field, config.distance_weight, min_clearance_px),
        config.rdp_tolerance_m / bev.resolution,
    )


def astar(cache: PlannerCache, start_pixel: tuple, goal_pixel: tuple) -> list | None:
    """Eight-connected A* over the cache; diagonal moves cannot cut corners.

    Runs the compiled search of one_scan_is_enough.planning._native. A cell already in
    the open set is not pushed again when its score improves. Returns the
    (row, col) cells from start to goal, or None when the goal is unreachable.
    """
    cells = _native.astar(
        cache.free_mask.view(np.uint8), cache.cost_map, tuple(start_pixel), tuple(goal_pixel)
    )
    return None if cells is None else [tuple(cell) for cell in cells.tolist()]


def line_clearance_safe(p0, p1, distance_field, min_clearance_px) -> bool:
    """Whether every cell on the straight line keeps min_clearance_px from obstacles."""
    r0, c0 = int(p0[0]), int(p0[1])
    r1, c1 = int(p1[0]), int(p1[1])
    h, w = distance_field.shape
    n_steps = max(abs(r1 - r0), abs(c1 - c0))
    if n_steps == 0:
        if 0 <= r0 < h and 0 <= c0 < w:
            return distance_field[r0, c0] >= min_clearance_px
        return False
    for k in range(n_steps + 1):
        t = k / n_steps
        r = int(round(r0 + t * (r1 - r0)))
        c = int(round(c0 + t * (c1 - c0)))
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        if distance_field[r, c] < min_clearance_px:
            return False
    return True


def rdp_simplify_path(path, distance_field, min_clearance_px, epsilon_px) -> list:
    """Ramer-Douglas-Peucker that also splits segments failing line_clearance_safe."""
    path = list(path)
    if len(path) <= 2:
        return path
    eps = float(epsilon_px)
    keep = {0, len(path) - 1}
    stack = [(0, len(path) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 <= 1:
            continue
        a, b = path[i0], path[i1]
        safe = line_clearance_safe(a, b, distance_field, min_clearance_px)
        best_i, best_d = -1, -1.0
        for i in range(i0 + 1, i1):
            d = point_segment_distance(path[i], a, b)
            if d > best_d:
                best_d, best_i = d, i
        if best_i >= 0 and (not safe or best_d > eps):
            keep.add(best_i)
            stack.append((i0, best_i))
            stack.append((best_i, i1))
    return [path[i] for i in sorted(keep)]


def plan_path(bev: BevMap, cache: PlannerCache, start_xz, goal_xz):
    """Plan between XZ positions in meters; return the simplified XZ path (N, 2) or None.

    Positions truncate to cells.
    """
    origin, resolution = bev.origin_xz, bev.resolution
    start = (
        int((start_xz[1] - origin[1]) / resolution),
        int((start_xz[0] - origin[0]) / resolution),
    )
    goal = (int((goal_xz[1] - origin[1]) / resolution), int((goal_xz[0] - origin[0]) / resolution))
    h, w = bev.navigable.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w and 0 <= goal[0] < h and 0 <= goal[1] < w):
        return None
    path = astar(cache, start, goal)
    if path is None:
        return None
    path = simplify(cache, path)
    return np.asarray([[c * resolution + origin[0], r * resolution + origin[1]] for r, c in path])


def simplify(cache: PlannerCache, path: list) -> list:
    """Simplify an A* path within the tolerance of the cache; chords stay on free cells."""
    if len(path) <= 2:
        return path
    return rdp_simplify_path(path, cache.distance_field, 1, cache.tolerance_px)
