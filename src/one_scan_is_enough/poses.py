"""Conversions between scene poses, robot-frame waypoints and local increments.

A scene pose is (X, Z, heading) in the aligned frame, in meters and radians.
Heading zero faces -Z and heading grows with right turns.

A waypoint is (x, y, theta) relative to a reference pose: x to the right and
y forward in meters, theta a right turn in radians. Policy outputs, planner
takeover actions and training labels all use waypoints.

An increment is a waypoint relative to the previous pose of a sequence.
Trajectory files store actions as (T, 2, 3) increments: one frame starts at
each even pose, and each action holds the two increments to the next frame.
"""

import math

import numpy as np


def wrap_angle(angle):
    """Wrap angles in radians to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def to_scene(pose: np.ndarray, waypoints: np.ndarray) -> np.ndarray:
    """Return the scene poses (N, 3) of waypoints (N, 3) relative to a scene pose (3,).

    Output headings are not wrapped.
    """
    x, z, heading = pose
    cos, sin = math.cos(heading), math.sin(heading)
    right, forward, turn = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3).T
    return np.stack(
        [x + cos * right + sin * forward, z + sin * right - cos * forward, heading + turn], axis=1
    )


def to_local(pose: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Return the waypoints (N, 3) of scene poses (N, 3) relative to a scene pose (3,).

    This is the inverse of to_scene. Output turns are wrapped to [-pi, pi].
    """
    x, z, heading = pose
    cos, sin = math.cos(heading), math.sin(heading)
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    dx, dz = poses[:, 0] - x, poses[:, 1] - z
    return np.stack(
        [cos * dx + sin * dz, sin * dx - cos * dz, wrap_angle(poses[:, 2] - heading)], axis=1
    )


def headings_along(points: np.ndarray, heading: float) -> np.ndarray:
    """Return scene poses (N, 3) that face along a polyline of XZ points (N, 2).

    Each point faces along the segment that arrives at it. Points that do not
    move keep the previous heading, and the first point keeps heading.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    headings = np.empty(len(points))
    for index in range(len(points)):
        if index:
            step_x, step_z = points[index] - points[index - 1]
            if abs(step_x) + abs(step_z) >= 1e-6:
                heading = math.atan2(step_x, -step_z)
        headings[index] = heading
    return np.column_stack([points, headings])


def increments(poses: np.ndarray) -> np.ndarray:
    """Return (T, 2, 3) trajectory actions from a scene pose sequence (N, 3).

    T is ceil(N / 2). Increments past the last pose are zero.
    """
    actions = np.zeros(((len(poses) + 1) // 2, 2, 3), dtype=np.float64)
    for index in range(len(poses) - 1):
        actions[index // 2, index % 2] = to_local(poses[index], poses[index + 1])[0]
    return actions


def accumulate(steps: np.ndarray) -> np.ndarray:
    """Return the waypoints (N, 3) reached by increments (N, 3) from the reference pose."""
    waypoints = np.zeros((len(steps), 3), dtype=np.float64)
    x = y = theta = 0.0
    for index, (dx, dy, dtheta) in enumerate(steps):
        # Rotate each increment by the accumulated turn; right turns rotate +y toward +x.
        x, y = (
            x + math.cos(theta) * dx + math.sin(theta) * dy,
            y - math.sin(theta) * dx + math.cos(theta) * dy,
        )
        theta += dtheta
        waypoints[index] = x, y, theta
    return waypoints
