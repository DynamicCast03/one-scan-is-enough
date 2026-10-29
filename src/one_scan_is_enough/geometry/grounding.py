"""Estimate a ground plane from Gaussian center samples."""

import logging

import numpy as np
import open3d as o3d
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def segment_ground(cloud, config: DictConfig, *, support_points, support_weights):
    """Select a ground plane from RANSAC candidates.

    Args:
        cloud: Point cloud used for candidate extraction and the one-sided check.
        config: Plane extraction and selection settings. The one-sided check counts
            only points farther than one_side_margin_m from the plane, in meters.
        support_points: Original Gaussian center pool, shape (N, 3), in meters.
        support_weights: Opacity times the two larger Gaussian scales, shape (N,).
            The largest support within the near band fixes the ground direction,
            and the lowest candidate parallel to it within parallel_max_deg is the
            ground, since raised surfaces such as beds can pass the one-sided check.

    Returns:
        Ground inliers and plane coefficients, or (None, None) to request a retry.
    """
    points = np.asarray(cloud.points)
    if len(points) == 0:
        raise ValueError("Empty point cloud")

    def get_plane_stats(plane_model) -> dict:
        a, b, c, d = [float(x) for x in plane_model]
        normal = np.array([a, b, c], dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        normal = normal / norm
        d = float(d) / norm

        signed = points @ normal + d
        pos = int(np.count_nonzero(signed > 0.0))
        neg = int(np.count_nonzero(signed < 0.0))
        up_sign = 1.0 if pos >= neg else -1.0
        signed_up = up_sign * signed
        # Ignore points near the plane, since a slightly bent floor puts its
        # points on both sides.
        up = int(np.count_nonzero(signed_up > config.one_side_margin_m))
        down = int(np.count_nonzero(signed_up < -config.one_side_margin_m))
        one_side_ratio = up / (up + down)
        near_cnt = int(np.count_nonzero((signed_up >= 0.0) & (signed_up <= config.near_distance_m)))

        # Keep the side direction from the candidate cloud for all scoring points.
        distances = up_sign * (support_points @ normal + d)
        near = (distances >= 0.0) & (distances <= config.near_distance_m)
        return {
            "up": up_sign * normal,
            "height": -up_sign * d,
            "one_side_ratio": one_side_ratio,
            "near_cnt": near_cnt,
            "support": float(np.sum(support_weights[near], dtype=np.float64)),
        }

    remaining_cloud = o3d.geometry.PointCloud(cloud)
    remaining_indices = np.arange(len(points), dtype=np.int64)
    candidates = []
    for _ in range(config.max_plane_candidates):
        plane_model, inliers = remaining_cloud.segment_plane(
            distance_threshold=config.distance_threshold,
            ransac_n=config.ransac_n,
            num_iterations=config.num_iterations,
        )
        inliers = np.asarray(inliers, dtype=np.int64)
        if len(inliers) == 0:
            break
        orig_inliers = remaining_indices[inliers]
        inlier_ratio = float(len(orig_inliers)) / float(len(points))
        # Small planes are removed but skipped, since larger ones such as a floor
        # partly hidden by furniture can come later.
        if inlier_ratio >= config.min_plane_inlier_ratio:
            candidates.append((plane_model, orig_inliers))

        keep_mask = np.ones(len(remaining_indices), dtype=bool)
        keep_mask[inliers] = False
        remaining_indices = remaining_indices[keep_mask]
        remaining_cloud = remaining_cloud.select_by_index(inliers.tolist(), invert=True)
        if len(remaining_indices) < 10:
            break

    if not candidates:
        return None, None

    stats = []
    for plane_model, inlier_indices in candidates:
        m = get_plane_stats(plane_model)
        stats.append((plane_model, inlier_indices, m))

    filtered = [
        (pm, idx, m) for pm, idx, m in stats if float(m["one_side_ratio"]) >= config.one_side_ratio
    ]
    if not filtered:
        logger.info(
            "No plane satisfies one_side_ratio>=%.2f; resampling the point cloud",
            config.one_side_ratio,
        )
        return None, None
    up = max(filtered, key=lambda x: x[2]["support"])[2]["up"]
    min_cos = np.cos(np.radians(config.parallel_max_deg))
    parallel = [item for item in filtered if abs(float(item[2]["up"] @ up)) >= min_cos]
    # Heights along the selected up direction; parallel planes may face either way.
    best_plane_model, best_inlier_indices, best_m = min(
        parallel, key=lambda x: x[2]["height"] * np.sign(float(x[2]["up"] @ up))
    )
    logger.info(
        "Ground candidates (height, support): %s",
        [(round(float(m["height"]), 3), round(m["support"], 3)) for _, _, m in parallel],
    )
    logger.info(
        "Selected ground: support=%.6f one_side_ratio=%.4f near_cnt=%d",
        best_m["support"],
        float(best_m["one_side_ratio"]),
        int(best_m["near_cnt"]),
    )

    inlier_list = [int(x) for x in np.asarray(best_inlier_indices, dtype=np.int64).tolist()]
    ground_cloud = cloud.select_by_index(inlier_list)
    return ground_cloud, best_plane_model


def compute_alignment_transform(plane_model):
    """
    Compute 4x4 transform matrix to align ground normal with Y-axis.
    """
    a, b, c, _ = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    if normal[1] < 0:
        normal = -normal

    target = np.array([0.0, 1.0, 0.0])
    dot = np.dot(normal, target)
    if np.abs(dot - 1.0) < 1e-6:
        return np.eye(4)

    axis = np.cross(normal, target)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    transform = np.eye(4)
    transform[:3, :3] = R
    return transform


def compute_ground_y_offset(ground_cloud, rotation_transform):
    """Compute Y offset to place ground at Y=0."""
    rotated_ground = o3d.geometry.PointCloud(ground_cloud)
    rotated_ground.transform(rotation_transform)
    points = np.asarray(rotated_ground.points)
    ground_y = np.median(points[:, 1])
    return ground_y


def check_y_direction(point_cloud, rotation_transform, ground_y_offset):
    """Check if Y-axis points towards the majority of points (upwards)."""
    rotated_cloud = o3d.geometry.PointCloud(point_cloud)
    rotated_cloud.transform(rotation_transform)
    rotated_cloud.translate([0, -ground_y_offset, 0])
    points = np.asarray(rotated_cloud.points)
    points_above = np.sum(points[:, 1] > 0)
    points_below = np.sum(points[:, 1] < 0)
    return points_below > points_above


def get_flip_y_transform():
    """Transform matrix for 180 degree rotation around X-axis."""
    transform = np.eye(4)
    transform[1, 1] = -1
    transform[2, 2] = -1
    return transform
