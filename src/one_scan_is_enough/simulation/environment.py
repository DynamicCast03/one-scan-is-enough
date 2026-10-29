"""A Gaussian scene that renders observations, checks positions and plans local paths."""

from collections.abc import Callable

import numpy as np
import torch
from omegaconf import DictConfig
from scipy.spatial import cKDTree

from one_scan_is_enough.planning.path_planner import plan_path, planner_cache, planning_map
from one_scan_is_enough.planning.path_sampling import sample_poses
from one_scan_is_enough.policy.inputs import WAYPOINTS
from one_scan_is_enough.poses import headings_along, to_local
from one_scan_is_enough.rendering.frames import Renderer
from one_scan_is_enough.rendering.gaussians import load_gaussian_scene
from one_scan_is_enough.scene import Scene
from one_scan_is_enough.simulation.execution import Navigation
from one_scan_is_enough.trajectories import decode_jpeg


class SceneEnvironment:
    """Navigation queries on one calibrated scene package, rendered on one GPU.

    Positions are aligned XZ in meters and poses follow one_scan_is_enough.poses. Runs
    snap to, check positions against and plan takeover actions on the
    planning map of the navigable BEV (one_scan_is_enough.planning.path_planner), the
    same feasible region as reference trajectory planning. Observations pass through the JPEG
    encoding of rendered training frames, so that closed-loop and training
    observations match. Planning needs the planner and path_sampling sections of
    the configuration and runs need the execution section.
    """

    def __init__(self, scene: Scene, config: DictConfig, device: torch.device):
        self.config = config
        transform, self.bev = scene.read_geometry()
        gaussians = load_gaussian_scene(scene.gaussian_path, device)
        self.renderer = Renderer(gaussians, transform, config.rendering)
        self.map = planning_map(self.bev, config.planning_map)
        self.cache = planner_cache(self.map, config.planner)
        self.free = np.argwhere(self.map.navigable > 0)  # (F, 2) rows and columns.
        self.free_tree = cKDTree(self.free.astype(np.float32))

    def start(self, pose: np.ndarray, observe: Callable[[np.ndarray], None]) -> Navigation:
        """Start a closed-loop run at a scene pose; see Navigation."""
        return Navigation(self, pose, self.config.execution, observe)

    @torch.inference_mode()
    def observe(self, pose: np.ndarray) -> np.ndarray:
        """Render the (H, W, 3) uint8 RGB view from a scene pose."""
        return decode_jpeg(self.renderer.jpeg(self.renderer.camera(pose)))

    def snap(self, position: np.ndarray) -> np.ndarray:
        """Return the free planning-map cell nearest to a position."""
        query = self._pixel(position, exact=True).astype(np.float32)
        row, col = self.free[self.free_tree.query(query)[1]]
        return self.map.origin_xz + np.array([col, row]) * self.map.resolution

    def outside(self, position: np.ndarray) -> bool:
        """Whether a position lies outside the free cells of the planning map."""
        row, col = self._pixel(position)
        height, width = self.map.navigable.shape
        if not (0 <= row < height and 0 <= col < width):
            return True
        return not self.cache.free_mask[row, col]

    def plan(self, pose: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Return eight waypoints (8, 3) from a pose along a safe path toward a goal.

        Start and goal snap to free cells. Waypoints face along the path.
        Unreachable goals get a straight line of eight equal steps. Short plans
        repeat their last waypoint and empty plans stay at the start.
        """
        return _eight(self._plan(pose, goal))

    def plan_along(
        self, pose: np.ndarray, reference: np.ndarray, lookahead: int, increment: int
    ) -> np.ndarray:
        """Plan toward the reference position lookahead indices past the nearest one.

        The lookahead grows by increment until the plan reaches eight waypoints
        or targets the reference end.
        """
        nearest = int(np.argmin(np.linalg.norm(reference - pose[:2], axis=1)))
        while True:
            target = min(nearest + lookahead, len(reference) - 1)
            waypoints = self._plan(pose, reference[target])
            if len(waypoints) >= WAYPOINTS or target == len(reference) - 1:
                return _eight(waypoints)
            lookahead += increment

    def _plan(self, pose: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Return all planned waypoints (N, 3) toward a goal; see plan."""
        path = plan_path(
            self.map,
            self.cache,
            self.snap(pose[:2]),
            self.snap(goal),
        )
        if path is None:
            points = np.linspace(pose[:2], goal, WAYPOINTS + 1)
        else:
            points = sample_poses(path, self.config.path_sampling)[:, :2]
        # Waypoints face along the path, which starts at the pose itself.
        points = np.vstack([pose[:2], points[1:].reshape(-1, 2)])
        return to_local(pose, headings_along(points, pose[2])[1:])

    def _pixel(self, position: np.ndarray, exact: bool = False):
        """Return the (row, col) planning-map cell of a position, unrounded when exact."""
        col, row = (
            np.asarray(position, dtype=np.float64) - self.map.origin_xz
        ) / self.map.resolution
        if exact:
            return np.array([row, col])
        return int(round(row)), int(round(col))


def _eight(waypoints: np.ndarray) -> np.ndarray:
    """Keep the first eight waypoints, repeating the last or the start as needed."""
    waypoints = waypoints[:WAYPOINTS]
    if not len(waypoints):
        return np.zeros((WAYPOINTS, 3))
    return np.concatenate([waypoints, np.repeat(waypoints[-1:], WAYPOINTS - len(waypoints), 0)])
