"""Sample scene poses along planned paths in the aligned XZ plane.

Sampling selects waypoints with two motion templates, a forward move and a
turn. It works internally with yaw = atan2(dx, dz), zero facing +Z, and
returns scene poses of one_scan_is_enough.poses, whose heading is pi - yaw.
"""

import logging
import math

import numpy as np
from omegaconf import DictConfig

from one_scan_is_enough.planning import _native

logger = logging.getLogger(__name__)


def _resample(path_xz: list[tuple[float, float]], stride_m: float) -> list[tuple[float, float]]:
    pts = np.asarray(path_xz, dtype=np.float64)
    if len(pts) < 2:
        return [(float(pts[0, 0]), float(pts[0, 1]))] if len(pts) == 1 else []

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total <= 1e-9:
        p = pts[0]
        return [(float(p[0]), float(p[1]))]

    targets = [0.0]
    s = float(stride_m)
    while s < total:
        targets.append(float(s))
        s += float(stride_m)
    targets.append(total)

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    out = []
    si = 0
    for t in targets:
        while si + 1 < len(cum) and cum[si + 1] < t:
            si += 1
        if si >= len(seg_len):
            p = pts[-1]
            out.append((float(p[0]), float(p[1])))
            continue
        denom = float(cum[si + 1] - cum[si])
        r = 0.0 if denom <= 1e-12 else float((t - cum[si]) / denom)
        p = (1.0 - r) * pts[si] + r * pts[si + 1]
        out.append((float(p[0]), float(p[1])))
    return out


def _yaw(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    dx = p1[0] - p0[0]
    dz = p1[1] - p0[1]
    if abs(dx) < 1e-12 and abs(dz) < 1e-12:
        return 0.0
    return float(math.atan2(dx, dz))


def sample_poses(path_xz, config: DictConfig) -> np.ndarray:
    """Return (N, 3) scene poses sampled along a path (M, 2) with move and turn templates.

    Each pose faces along the dense path from its position; the last faces
    along the final segment.
    """
    max_lateral_m = config.max_lateral_m
    max_forward_m = config.max_forward_m
    max_theta_deg = config.max_theta_deg
    # Build the dense reference path.
    dense_xz = _resample(path_xz, stride_m=config.dense_stride_m)
    if not dense_xz:
        return np.empty((0, 3))
    if len(dense_xz) == 1:
        return _scene_poses(dense_xz, [0.0])

    dense_yaws: list[float] = []
    for i in range(len(dense_xz)):
        if i + 1 < len(dense_xz):
            dense_yaws.append(_yaw(dense_xz[i], dense_xz[i + 1]))
        else:
            dense_yaws.append(_yaw(dense_xz[i - 1], dense_xz[i]))

    n_dense = len(dense_xz)
    indices = _native.select_waypoints(
        np.asarray(dense_xz, dtype=np.float64),
        np.asarray(dense_yaws, dtype=np.float64),
        max_lateral_m,
        max_forward_m,
        math.radians(max_theta_deg),
    ).tolist()

    logger.debug("Dual-template waypoint: %d waypoints (dense=%d)", len(indices), n_dense)

    return _scene_poses([dense_xz[i] for i in indices], [dense_yaws[i] for i in indices])


def _scene_poses(points: list[tuple[float, float]], yaws: list[float]) -> np.ndarray:
    """Convert XZ points and yaws to scene poses with headings wrapped to [-pi, pi]."""
    headings = [math.atan2(math.sin(math.pi - yaw), math.cos(math.pi - yaw)) for yaw in yaws]
    return np.array([[x, z, heading] for (x, z), heading in zip(points, headings)])
