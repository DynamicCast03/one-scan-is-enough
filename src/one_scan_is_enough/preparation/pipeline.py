"""Prepare calibrated Gaussian scenes: ground alignment and a navigable BEV map."""

import logging

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from one_scan_is_enough.geometry.gaussian_bev import build_gaussian_bev
from one_scan_is_enough.geometry.gaussian_ply import compute_scale_cap, iter_geometry
from one_scan_is_enough.geometry.grounding import (
    check_y_direction,
    compute_alignment_transform,
    compute_ground_y_offset,
    get_flip_y_transform,
    segment_ground,
)
from one_scan_is_enough.preparation.bev_editor import calibrate_yaw, edit_navigable
from one_scan_is_enough.scene import Scene

logger = logging.getLogger(__name__)


def sample_centers(path, config, rng, scale_cap):
    """Return sampled Gaussian centers and their opacity-weighted surface areas.

    Independent random priorities give each center the same inclusion chance.
    Only the current batch and the retained pool are held in memory.
    The scale_cap cutoff in meters is also used for BEV projection.
    """
    points = np.empty((0, 3), dtype=np.float32)
    weights = np.empty(0, dtype=np.float32)
    priorities = np.empty(0)
    for batch in tqdm(
        iter_geometry(path, config.batch_size, scale_max=scale_cap),
        desc="Ground centers",
        unit="batch",
    ):
        visible = batch.opacities >= config.opacity_min
        centers = batch.xyz[visible]
        scales = np.sort(batch.scales[visible], axis=1)
        # The two larger axes describe the surface patch; omit the common area factor.
        areas = batch.opacities[visible] * scales[:, 1] * scales[:, 2]
        weights = np.concatenate((weights, areas))
        points = np.concatenate((points, centers))
        priorities = np.concatenate((priorities, rng.random(len(centers))))
        if len(points) > config.ground_pool_size:
            keep = np.argpartition(priorities, config.ground_pool_size - 1)[
                : config.ground_pool_size
            ]
            points, weights, priorities = points[keep], weights[keep], priorities[keep]
    return points, weights


def align_centers(points, weights, config, rng):
    """Select a one-sided plane by Gaussian surface support and align it.

    Returns the original-to-aligned 4x4 transform, with ground at Y=0.
    Voxel sampling reduces density bias in the bounded center pool.
    Each retry draws a new subset from the voxel centroids.
    Scoring uses the original pool and its weights, before voxel sampling.
    """
    support_points = points
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud = cloud.voxel_down_sample(config.voxel_size)
    logger.info("Ground voxel sampling: %s -> %s points", len(points), len(cloud.points))
    points = np.asarray(cloud.points)
    for _ in range(config.resample_attempts):
        indices = rng.choice(len(points), min(len(points), config.num_samples), replace=False)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[indices])
        ground, plane = segment_ground(
            cloud, config, support_points=support_points, support_weights=weights
        )
        if ground is not None:
            break
    else:
        raise RuntimeError("No ground plane satisfies the configured support rules.")
    transform = compute_alignment_transform(plane)
    offset = compute_ground_y_offset(ground, transform)
    if check_y_direction(cloud, transform, offset):
        transform = get_flip_y_transform() @ transform
        offset = compute_ground_y_offset(ground, transform)
    transform[1, 3] = -offset
    return transform


def prepare(config, source, target):
    """Write gaussians.ply and scene.h5 after both manual steps.

    The target must be empty. Failed runs may leave files in the target.
    """
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    package = Scene(target)
    logger.info("Copying Gaussian source: %s", source)
    with source.open("rb") as incoming, package.gaussian_path.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
            outgoing.write(block)

    rng = np.random.default_rng(config.seed)
    o3d.utility.random.seed(config.seed)
    scale_cap = compute_scale_cap(
        package.gaussian_path, config.gaussians.batch_size, config.gaussians.scale_max_quantile
    )
    logger.info("Gaussian scale norm cutoff: %.6f m", scale_cap)
    points, weights = sample_centers(package.gaussian_path, config.gaussians, rng, scale_cap)
    logger.info("Selecting ground from %s Gaussian centers", len(points))
    transform = align_centers(points, weights, config.alignment, rng)
    del points, weights
    bev, views = build_gaussian_bev(
        package.gaussian_path, transform, config.gaussians, config.bev, scale_cap
    )

    yaw = calibrate_yaw(bev, views, config.calibration)
    yaw_transform = np.eye(4)
    yaw_transform[:3, :3] = Rotation.from_euler("y", yaw).as_matrix()
    transform = yaw_transform @ transform
    del bev, views
    bev, views = build_gaussian_bev(
        package.gaussian_path, transform, config.gaussians, config.bev, scale_cap
    )

    bev.navigable = edit_navigable(bev, views, config.calibration)
    del views
    package.write(config.input.scene_name, transform, bev)
    logger.info("Prepared Gaussian scene package: %s", target)
    return target
