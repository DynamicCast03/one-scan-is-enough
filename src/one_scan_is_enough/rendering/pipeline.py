"""Render trajectories on separate GPUs and publish one trajectory HDF5 file."""

import fcntl
import logging
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from one_scan_is_enough.scene import Scene
from one_scan_is_enough.trajectories import TrajectoryFile

logger = logging.getLogger(__name__)


def _render_worker(
    gpu: int, scene_dir: Path, part: Path, ids: list[str], config: DictConfig
) -> None:
    """Render assigned trajectories into one temporary trajectory file on one GPU."""
    import torch

    from one_scan_is_enough.rendering.frames import Renderer, render_route
    from one_scan_is_enough.rendering.gaussians import load_gaussian_scene
    from one_scan_is_enough.rendering.goal_circle import add_goal_circle, select_goal_views

    torch.cuda.set_device(gpu)
    scene = Scene(scene_dir)
    gaussians = load_gaussian_scene(scene.gaussian_path, torch.device(f"cuda:{gpu}"))
    transform, bev = scene.read_geometry()
    rendering = config.rendering
    with (
        torch.inference_mode(),
        TrajectoryFile(scene.trajectories_path) as planned,
        TrajectoryFile(part, "w") as output,
    ):
        training = Renderer(gaussians, transform, rendering.training)
        for key in tqdm(ids, desc=f"GPU {gpu}", unit="trajectory"):
            poses = planned.poses(key)
            # Training frames start at the even poses, one per action.
            train_indices = np.arange(0, len(poses), 2, dtype=np.int64)
            train_frames = training.jpegs(
                np.stack([training.camera(poses[i]) for i in train_indices])
            )
            route_indices = np.unique(
                np.r_[np.arange(0, len(poses), rendering.route.sample_every), len(poses) - 1]
            ).astype(np.int64)
            marked, ground = add_goal_circle(
                gaussians, transform, bev, poses[-1, :2], rendering.goal_circle
            )
            route_frames = render_route(marked, poses, transform, rendering.route, route_indices)
            target, eyes, views = select_goal_views(
                marked, transform, bev, poses[-1, :2], ground, rendering.goal_views
            )
            if not len(eyes):
                logger.warning("No goal view shows the goal region of trajectory %s", key)
            goal = Renderer(marked, transform, rendering.goal_views)
            goal_frames = goal.jpegs(views)
            output.add_images(
                key,
                training=train_frames,
                route=route_frames,
                goal=(target, eyes, goal_frames),
            )


def render(config: DictConfig, scene_dir: Path) -> Path:
    """Render all trajectories and publish one complete trajectory file.

    Each GPU writes one temporary file. A failed run discards all rendered
    images; the original trajectory file remains available for a fresh run.
    """
    scene = Scene(scene_dir)
    if not config.execution.gpus or len(set(config.execution.gpus)) != len(config.execution.gpus):
        raise ValueError("Select distinct GPU indices.")
    with (scene_dir / ".trajectories.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with TrajectoryFile(scene.trajectories_path) as planned:
            planned.require("planning")
            if planned.complete("rendering"):
                raise FileExistsError(f"Trajectories already rendered: {scene.trajectories_path}")
            keys = planned.keys()
            if not keys:
                raise ValueError("No trajectories to render.")
        assignments = [
            (gpu, keys[index :: len(config.execution.gpus)])
            for index, gpu in enumerate(config.execution.gpus)
            if index < len(keys)
        ]
        parts = scene_dir / ".render_parts"
        temporary = scene.trajectories_path.with_suffix(".h5.tmp")
        if parts.exists():
            shutil.rmtree(parts)
        parts.mkdir()
        try:
            with ProcessPoolExecutor(
                max_workers=len(assignments), mp_context=mp.get_context("spawn")
            ) as executor:
                futures = [
                    executor.submit(
                        _render_worker, gpu, scene_dir, parts / f"gpu_{gpu}.h5", ids, config
                    )
                    for gpu, ids in assignments
                ]
                for future in as_completed(futures):
                    future.result()

            shutil.copyfile(scene.trajectories_path, temporary)
            with TrajectoryFile(temporary, "r+") as target:
                for gpu, ids in tqdm(assignments, desc="Merge rendered images", unit="GPU"):
                    with TrajectoryFile(parts / f"gpu_{gpu}.h5") as source:
                        target.copy_images(source, ids)
                target.finish("rendering")
            temporary.replace(scene.trajectories_path)
        finally:
            temporary.unlink(missing_ok=True)
            shutil.rmtree(parts)
    logger.info("Rendered training, route and goal images: %s", scene.trajectories_path)
    return scene.trajectories_path
