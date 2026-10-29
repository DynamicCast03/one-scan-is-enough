"""Execute predicted waypoints in a scene: the stop rule, the waypoint budget and pose updates.

Scene poses and waypoints follow one_scan_is_enough.poses.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig

from one_scan_is_enough.poses import to_scene

# Every second of the eight predicted waypoints is executable.
EXECUTED = slice(1, None, 2)


def truncate_at_stop(
    waypoints: np.ndarray,
    executed: int,
    min_waypoints: int,
    distance_m: float,
    heading_rad: float,
) -> tuple[np.ndarray, bool]:
    """Cut executed waypoints (N, 3) at the first stop and report whether one occurred.

    A waypoint stops the episode when it differs from the previous executed
    waypoint, or from the start for the first, by less than distance_m along x
    and y and heading_rad in theta. Stops count only once executed earlier
    waypoints plus the waypoint index reach min_waypoints. The stopping
    waypoint and those after it are not executed.
    """
    previous = np.zeros(3)
    for index, waypoint in enumerate(waypoints):
        if executed + index >= min_waypoints:
            gap = np.abs(waypoint - previous)
            if gap[0] < distance_m and gap[1] < distance_m and gap[2] < heading_rad:
                return waypoints[:index], True
        previous = waypoint
    return waypoints, False


@dataclass
class Move:
    """The executed part of one predicted action."""

    poses: np.ndarray  # (K, 3) scene poses of the executed waypoints, before snapping.
    stopped: bool  # Whether the action stops the episode.


class Navigation:
    """One closed-loop navigation run in a scene environment.

    Each step executes up to waypoints_per_decision of the executable
    predicted waypoints, cut by the stop rule of truncate_at_stop and by the
    max_waypoints budget. Every executed waypoint snaps to the nearest free
    planning-map cell and renders one observation. The run is done after a stop or
    once the budget is spent. Create runs with SceneEnvironment.start.
    """

    def __init__(
        self,
        environment,
        start: np.ndarray,
        config: DictConfig,
        observe: Callable[[np.ndarray], None],
    ):
        """Start a run at a scene pose and pass its first observation to observe.

        Args:
            environment: The SceneEnvironment of the run.
            start: The start scene pose (3,).
            config: waypoints_per_decision, max_waypoints and the stop rule
                settings distance_m, heading_deg and min_waypoints.
            observe: Receives each (H, W, 3) uint8 RGB observation in time order.
        """
        self.environment = environment
        self.config = config
        self.observe = observe
        self.pose = np.asarray(start, dtype=np.float64)
        self.trajectory = [self.pose]  # Scene poses (3,) after each executed waypoint.
        self.executed = 0  # Executed waypoints.
        self.path_length = 0.0  # Travelled distance in meters.
        self.stopped = False
        observe(environment.observe(self.pose))

    @property
    def done(self) -> bool:
        return self.stopped or self.executed >= self.config.max_waypoints

    def preview(self, action: np.ndarray) -> Move:
        """Return what step would execute for an (8, 3) action, without moving."""
        stop = self.config.stop
        waypoints, stopped = truncate_at_stop(
            action[EXECUTED][: self.config.waypoints_per_decision],
            self.executed,
            stop.min_waypoints,
            stop.distance_m,
            math.radians(stop.heading_deg),
        )
        waypoints = waypoints[: self.config.max_waypoints - self.executed]
        return Move(to_scene(self.pose, waypoints), stopped)

    def step(self, action: np.ndarray) -> Move:
        """Execute an (8, 3) action, observe every reached pose and return the move."""
        move = self.preview(action)
        for point in move.poses:
            position = self.environment.snap(point[:2])
            self.path_length += float(np.linalg.norm(position - self.pose[:2]))
            self.pose = np.array([*position, point[2]])
            self.trajectory.append(self.pose)
            self.observe(self.environment.observe(self.pose))
        self.executed += len(move.poses)
        self.stopped = move.stopped
        return move
