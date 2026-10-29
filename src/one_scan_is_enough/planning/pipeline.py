"""Write planned paths and action targets into a scene's trajectory file."""

from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from one_scan_is_enough.planning.path_planner import planning_map
from one_scan_is_enough.planning.trajectories import plan_trajectories
from one_scan_is_enough.scene import Scene
from one_scan_is_enough.trajectories import TrajectoryFile


def plan(config: DictConfig, directory: Path) -> Path:
    """Create trajectories.h5 from the scene map without changing scene.h5."""
    scene = Scene(directory)
    if scene.trajectories_path.exists():
        raise FileExistsError(f"Trajectory file already exists: {scene.trajectories_path}")
    _, bev = scene.read_geometry()
    trajectories = plan_trajectories(
        planning_map(bev, config.planning_map),
        config.planning,
        config.planner,
        config.path_sampling,
        np.random.default_rng(config.seed),
        config.execution.workers,
    )
    temporary = scene.trajectories_path.with_suffix(".h5.tmp")
    with TrajectoryFile(temporary, "w", scene_name=scene.read_name()) as target:
        for index, trajectory in enumerate(trajectories):
            target.add_planned(f"{index:06d}", trajectory.path_xz, trajectory.poses)
        target.finish("planning")
    temporary.replace(scene.trajectories_path)
    return scene.trajectories_path
