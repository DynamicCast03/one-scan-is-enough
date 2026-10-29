"""Mark a goal with a red circle on the local ground and select views of its surroundings.

The goal region is a box around the goal from the ground up. Candidate cameras
stand at camera_height_m on a horizontal grid around the goal and look at the
box center. Cameras that do not frame the whole box are dropped, and a greedy
maximum coverage selects up to config.maximum views that together show the
most region surface.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from omegaconf import DictConfig

from one_scan_is_enough.scene import BevMap

from .frames import Renderer, append_markers, intrinsics_matrix
from .gaussians import GaussianScene

# Pixels with lower accumulated opacity have no reliable depth.
MIN_DEPTH_ALPHA = 0.5
# Probe views rasterized together; probes are small, so batches can be large.
PROBE_BATCH_VIEWS = 128


def _aligned_means(scene: GaussianScene, transform: np.ndarray) -> torch.Tensor:
    """Return (G, 3) Gaussian centers in the aligned frame."""
    device = scene.means.device
    rotation = torch.as_tensor(transform[:3, :3], dtype=torch.float32, device=device)
    offset = torch.as_tensor(transform[:3, 3], dtype=torch.float32, device=device)
    return scene.means @ rotation.T + offset


def _open_arcs(bev: BevMap, goal_xz: np.ndarray, angles: np.ndarray, radius: float) -> np.ndarray:
    """Return which circle angles the raw BEV leaves unobstructed from the goal."""
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    steps = np.linspace(0.0, 1.0, math.ceil(radius / bev.resolution) + 1)
    rays = goal_xz + steps[None, :, None] * radius * directions[:, None, :]
    pixels = np.round((rays - bev.origin_xz) / bev.resolution).astype(np.int64)
    cols, rows = pixels[..., 0], pixels[..., 1]
    height, width = bev.raw.shape
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    free = bev.raw[rows.clip(0, height - 1), cols.clip(0, width - 1)] == 255
    return (inside & free).all(axis=1)


def _ground_height(
    scene: GaussianScene,
    transform: np.ndarray,
    goal_xz: np.ndarray,
    open_arcs: np.ndarray,
    config: DictConfig,
) -> float:
    """Return the aligned ground height along the open circle arcs.

    It is the ground_quantile height of opaque Gaussians within ground_band_m
    of the circle and below ground_max_m.
    """
    device = scene.means.device
    aligned = _aligned_means(scene, transform)
    dx = aligned[:, 0] - float(goal_xz[0])
    dz = aligned[:, 2] - float(goal_xz[1])
    near = (
        (((dx.square() + dz.square()).sqrt() - config.radius_m).abs() < config.ground_band_m)
        & (scene.opacities > 0.5)
        & (aligned[:, 1] < config.ground_max_m)
    )
    angles = torch.atan2(dz[near], dx[near]) % (2.0 * math.pi)
    bins = (angles / (2.0 * math.pi) * len(open_arcs)).round().long() % len(open_arcs)
    heights = aligned[near, 1][torch.as_tensor(open_arcs, device=device)[bins]]
    if not len(heights):
        return 0.0
    return torch.quantile(heights, config.ground_quantile).item()


@torch.inference_mode()
def add_goal_circle(
    scene: GaussianScene,
    transform: np.ndarray,
    bev: BevMap,
    goal_xz: np.ndarray,
    config: DictConfig,
) -> tuple[GaussianScene, float]:
    """Append a red ground circle of radius_m around the goal.

    Arcs that the raw BEV blocks from the goal are dropped, so the circle never
    shows through a wall or furniture. The circle floats lift_m above the
    local ground height.

    Returns:
        The scene with the circle, and the local ground height.
    """
    count = math.ceil(2.0 * math.pi * config.radius_m / config.stride_m)
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    open_arcs = _open_arcs(bev, goal_xz, angles, config.radius_m)
    ground = _ground_height(scene, transform, goal_xz, open_arcs, config)
    directions = np.column_stack([np.cos(angles[open_arcs]), np.sin(angles[open_arcs])])
    offsets = np.linspace(-config.width_m / 2, config.width_m / 2, config.width_samples)
    xz = (goal_xz + (config.radius_m + offsets[None, :, None]) * directions[:, None, :]).reshape(
        -1, 2
    )
    points = np.column_stack([xz[:, 0], np.full(len(xz), ground + config.lift_m), xz[:, 1]])
    marked = append_markers(
        scene, points, transform, config.scale_m, (1.0, 0.0, 0.0), config.opacity
    )
    return marked, ground


def _look_at(eyes: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return (N, 3, 3) aligned-to-camera rotations that look from eyes at target.

    Rows are camera right, down, and forward in the aligned frame, with Y up
    and no roll. No eye is directly above or below target.
    """
    forward = target - eyes
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    return np.stack([right, np.cross(forward, right), forward], 1)


def _candidate_cameras(
    corners: np.ndarray,
    ground: float,
    bev: BevMap,
    intrinsics: np.ndarray,
    config: DictConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the cameras around the region that frame the region.

    Cameras stand camera_height_m above the ground on a square grid of
    grid_step_m within radius_m of the goal, and look at the box center. The
    grid is offset by half a step, so that no camera is directly above the
    center. A camera is kept when:
    - it stands over navigable floor, so that it stays inside the rooms;
    - all projected box corners are in front of the camera and inside the
      image, at least edge_margin_px from the border;
    - the shorter side of the projected box covers at most max_short_ratio
      of the image side, so that the image also shows the surroundings.
    Furniture is not tested here; views behind it show little region surface.

    Returns:
        Aligned camera centers (N, 3) and aligned-to-camera rotations (N, 3, 3).
    """
    center = corners.mean(0)
    step = config.grid_step_m
    half = int(np.ceil(config.radius_m / step))
    axis = (np.arange(-half, half) + 0.5) * step
    offsets = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    xz = center[[0, 2]] + offsets[np.linalg.norm(offsets, axis=1) <= config.radius_m]
    eyes = np.column_stack([xz[:, 0], np.full(len(xz), ground + config.camera_height_m), xz[:, 1]])
    pixels = np.round((xz - bev.origin_xz) / bev.resolution).astype(np.int64)
    rows, cols = pixels[:, 1], pixels[:, 0]
    height, width = bev.navigable.shape
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    eyes = eyes[inside & (bev.navigable[rows.clip(0, height - 1), cols.clip(0, width - 1)] == 255)]
    rotations = _look_at(eyes, center)

    camera = np.einsum("nij,nkj->nki", rotations, corners[None] - eyes[:, None])  # (N, 8, 3)
    in_front = (camera[..., 2] > 1e-3).all(1)
    depth = np.where(in_front[:, None], camera[..., 2], 1.0)
    u = intrinsics[0, 0] * camera[..., 0] / depth + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera[..., 1] / depth + intrinsics[1, 2]
    width, height = 2 * intrinsics[0, 2], 2 * intrinsics[1, 2]
    margin = config.edge_margin_px
    inside = (
        (u.min(1) >= margin)
        & (u.max(1) <= width - margin)
        & (v.min(1) >= margin)
        & (v.max(1) <= height - margin)
    )
    short = np.minimum(np.ptp(u, 1) / width, np.ptp(v, 1) / height)
    keep = in_front & inside & (short <= config.max_short_ratio)
    return eyes[keep], rotations[keep]


def _world_views(eyes: np.ndarray, rotations: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Return (N, 4, 4) view matrices in the original Gaussian frame."""
    views = np.tile(np.eye(4, dtype=np.float32), (len(eyes), 1, 1))
    views[:, :3, :3] = rotations @ transform[:3, :3]
    views[:, :3, 3] = np.einsum("nij,nj->ni", rotations, transform[:3, 3] - eyes)
    return views


def _subset(scene: GaussianScene, mask: torch.Tensor) -> GaussianScene:
    """Return the Gaussians selected by a boolean mask (G,)."""
    return GaussianScene(
        means=scene.means[mask],
        quaternions=scene.quaternions[mask],
        scales=scene.scales[mask],
        opacities=scene.opacities[mask],
        spherical_harmonics=scene.spherical_harmonics[mask],
        sh_degree=scene.sh_degree,
    )


def _visible_surface(
    scene: GaussianScene,
    region: GaussianScene,
    eyes: np.ndarray,
    rotations: np.ndarray,
    transform: np.ndarray,
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the region surface that each candidate view shows.

    Views render in batches at probe resolution, twice: once with the whole scene
    and once with only the region Gaussians. A region pixel is visible when
    the whole-scene depth matches the region-only depth within
    depth_tolerance_m. Visible pixels are back-projected to aligned points and quantized into
    voxels of voxel_size_m, so that the same surface patch maps to the same
    voxel from any view.

    Returns:
        Per visible pixel, the view index (P,) and voxel key (P,).
    """
    device = scene.means.device
    downsample = config.probe_downsample
    renderer = Renderer(
        scene, transform, config, size=(config.width // downsample, config.height // downsample)
    )
    width, height = renderer.size
    intrinsics = renderer.intrinsics
    v, u = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
    )
    rays = torch.stack(
        [
            (u + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0],
            (v + 0.5 - intrinsics[1, 2]) / intrinsics[1, 1],
            torch.ones_like(u, dtype=torch.float32),
        ],
        -1,
    )  # (H, W, 3), camera frame at unit depth.

    def depth(gaussians: GaussianScene, views: np.ndarray) -> torch.Tensor:
        rendered, alphas = renderer.rasterize(views, gaussians, mode="ED")
        return torch.where(alphas >= MIN_DEPTH_ALPHA, rendered[..., 0], 0.0)

    views = _world_views(eyes, rotations, transform)
    eyes_t = torch.as_tensor(eyes, dtype=torch.float32, device=device)
    rotations_t = torch.as_tensor(rotations, dtype=torch.float32, device=device)
    view_ids, keys = [], []
    for start in range(0, len(views), PROBE_BATCH_VIEWS):
        batch = views[start : start + PROBE_BATCH_VIEWS]
        scene_depth, region_depth = depth(scene, batch), depth(region, batch)
        visible = (
            (region_depth > 0)
            & (scene_depth > 0)
            & ((scene_depth - region_depth).abs() < config.depth_tolerance_m)
        )
        index, y, x = visible.nonzero(as_tuple=True)
        index += start
        camera = rays[y, x] * scene_depth[visible][:, None]
        aligned = torch.einsum("pi,pij->pj", camera, rotations_t[index]) + eyes_t[index]
        # Pack each voxel index into one int64 key with 21 bits per axis.
        voxels = torch.floor(aligned / config.voxel_size_m).long() + (1 << 20)
        keys.append((voxels[:, 0] << 42) | (voxels[:, 1] << 21) | voxels[:, 2])
        view_ids.append(index)
    return torch.cat(view_ids), torch.cat(keys)


def _select_views(
    view_ids: torch.Tensor, keys: torch.Tensor, count: int, config: DictConfig
) -> list[int]:
    """Select views that together show the most region surface.

    Greedy maximum coverage: each step adds the view with the most voxels that no selected
    view shows. Selection stops after maximum views or when no view adds a voxel.
    """
    if len(keys) == 0:
        return []
    voxels, columns = torch.unique(keys, return_inverse=True)
    shown = torch.zeros(count, len(voxels), dtype=torch.bool, device=keys.device)
    shown[view_ids, columns] = True
    covered = torch.zeros(len(voxels), dtype=torch.bool, device=keys.device)
    selected: list[int] = []
    while len(selected) < config.maximum:
        gain = (shown & ~covered).sum(1)
        best = int(gain.argmax())
        if gain[best] == 0:
            break
        selected.append(best)
        covered |= shown[best]
    return selected


@torch.inference_mode()
def select_goal_views(
    scene: GaussianScene,
    transform: np.ndarray,
    bev: BevMap,
    goal_xz: np.ndarray,
    ground: float,
    config: DictConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select views of the goal region.

    The region extends region_half_m from the goal along X and Z, and from
    the ground up to region_height_m above it.

    Returns:
        The aligned look-at target (3,), and for the selected views, aligned
        camera centers (K, 3) and view matrices (K, 4, 4) in the original
        Gaussian frame. K is zero when no candidate frames the region.
    """
    half, top = config.region_half_m, config.region_height_m
    lower = np.array([goal_xz[0] - half, ground, goal_xz[1] - half])
    upper = np.array([goal_xz[0] + half, ground + top, goal_xz[1] + half])
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    corners = np.where(corners, upper, lower)
    target = (lower + upper) / 2
    intrinsics = intrinsics_matrix(config.width, config.height, config.vertical_fov_deg)
    eyes, rotations = _candidate_cameras(corners, ground, bev, intrinsics, config)
    aligned = _aligned_means(scene, transform)
    mask = (
        (aligned >= torch.as_tensor(lower, dtype=torch.float32, device=aligned.device))
        & (aligned <= torch.as_tensor(upper, dtype=torch.float32, device=aligned.device))
    ).all(1)
    selected: list[int] = []
    if len(eyes) and mask.any():
        visible = _visible_surface(scene, _subset(scene, mask), eyes, rotations, transform, config)
        selected = _select_views(*visible, len(eyes), config)
    eyes, rotations = eyes[selected], rotations[selected]
    return target, eyes, _world_views(eyes, rotations, transform)
