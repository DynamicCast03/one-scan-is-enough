"""Render views of Gaussian scenes, route markers and first-person route frames."""

from __future__ import annotations

import functools
import math

import numpy as np
import torch
from gsplat import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    spherical_harmonics,
)
from omegaconf import DictConfig
from torchvision.io import encode_jpeg

from .gaussians import GaussianScene

# Gaussians up to this scale norm, in meters, define the scene content extent.
CONTENT_MAX_SCALE_M = 1.0
# View depth, in meters, to which oversized near Gaussians are pushed.
PUSH_DEPTH_M = 1.0e4
# Screen tile size of the gsplat rasterizer, in pixels.
TILE_SIZE = 16
# Views rasterized together; memory grows with the Gaussians each view projects.
BATCH_VIEWS = 16


def append_markers(
    scene: GaussianScene,
    points: np.ndarray,
    transform: np.ndarray,
    scale: float,
    color: tuple[float, float, float],
    opacity: float,
) -> GaussianScene:
    """Append isotropic single-color Gaussians at (N, 3) aligned XYZ points."""
    world = (transform[:3, :3].T @ (points - transform[:3, 3]).T).T
    device = scene.means.device
    count = len(world)
    sh = torch.zeros((count, scene.spherical_harmonics.shape[1], 3), device=device)
    sh[:, 0] = (torch.tensor(color, device=device) - 0.5) / 0.28209479177387814
    quaternions = torch.zeros((count, 4), device=device)
    quaternions[:, 0] = 1.0
    return GaussianScene(
        means=torch.cat([scene.means, torch.as_tensor(world, dtype=torch.float32, device=device)]),
        quaternions=torch.cat([scene.quaternions, quaternions]),
        scales=torch.cat([scene.scales, torch.full((count, 3), scale, device=device)]),
        opacities=torch.cat([scene.opacities, torch.full((count,), opacity, device=device)]),
        spherical_harmonics=torch.cat([scene.spherical_harmonics, sh]),
        sh_degree=scene.sh_degree,
    )


def _route_markers(
    scene: GaussianScene, poses: np.ndarray, transform: np.ndarray, config: DictConfig
) -> tuple[GaussianScene, torch.Tensor, np.ndarray]:
    """Append orange ground markers and return their arc lengths in meters."""
    xz = poses[:, :2]
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))]
    if distances[-1] <= 0:
        raise ValueError("Route marker needs a path with positive length.")
    pose_distances = distances
    unique = np.r_[True, np.diff(distances) > 1e-9]
    distances, xz = distances[unique], xz[unique]
    locations = np.r_[
        np.arange(0.0, distances[-1], max(0.005, config.path_stride_m)), distances[-1]
    ]
    centers = np.column_stack([np.interp(locations, distances, xz[:, i]) for i in range(2)])
    tangents = np.gradient(centers, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    right = np.column_stack([tangents[:, 1], -tangents[:, 0]])
    offsets = np.linspace(
        -config.path_width_m / 2, config.path_width_m / 2, config.path_width_samples
    )
    marker_xz = (centers[:, None, :] + offsets[None, :, None] * right[:, None, :]).reshape(-1, 2)
    marker_s = np.repeat(locations, len(offsets))
    aligned = np.column_stack(
        [marker_xz[:, 0], np.full(len(marker_xz), config.path_height_m), marker_xz[:, 1]]
    )
    augmented = append_markers(
        scene, aligned, transform, config.path_scale_m, (1.0, 0.65, 0.0), config.path_opacity
    )
    device = scene.means.device
    return augmented, torch.as_tensor(marker_s, dtype=torch.float32, device=device), pose_distances


def _content_bounds(scene: GaussianScene, transform: np.ndarray) -> np.ndarray:
    """Return aligned XZ bounds, (2, 2) as [lower, upper], of the scene content.

    Content means Gaussians with scale norms at or below CONTENT_MAX_SCALE_M.
    The 0.5% and 99.5% quantiles of a 1/16 center sample ignore sparse outliers.
    """
    device = scene.means.device
    centers = scene.means[scene.scales.norm(dim=1) <= CONTENT_MAX_SCALE_M][::16]
    rotation = torch.as_tensor(transform[:3, :3], dtype=torch.float32, device=device)
    offset = torch.as_tensor(transform[:3, 3], dtype=torch.float32, device=device)
    xz = (centers @ rotation.T + offset)[:, [0, 2]]
    lower = xz.kthvalue(max(1, int(0.005 * len(xz))), dim=0).values
    upper = xz.kthvalue(max(1, int(0.995 * len(xz))), dim=0).values
    return torch.stack([lower, upper]).cpu().numpy()


def _cull_depth_limit(pose: np.ndarray, bounds: np.ndarray, margin_m: float) -> float:
    """Return the view depth beyond which Gaussians count as background.

    It is the horizontal distance from the camera to the farthest content
    corner plus margin_m. Nearer Gaussians can occlude scene content.
    """
    corners = np.array([[bounds[i, 0], bounds[j, 1]] for i in (0, 1) for j in (0, 1)])
    return float(np.linalg.norm(corners - pose[:2], axis=1).max() + margin_m)


def _project(
    gaussians: GaussianScene,
    opacities: torch.Tensor,
    views: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    depth_limits: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, ...]:
    """Project Gaussians into views (C, 4, 4) and move oversized near ones behind everything.

    Oversized Gaussians fail screen-space covariance culling: the infinity
    norm of their pixel covariance exceeds alpha * width * height. Those with
    a center depth below the depth limit of their view also lie in front of
    scene content, while distant background Gaussians stay in place. Their depth is
    raised to PUSH_DEPTH_M, as if their center and scale were scaled about
    the camera: the projection is unchanged, but they only fill otherwise
    empty pixels. Opacities (N,) bound the projected radii.

    Returns:
        Packed camera ids, Gaussian ids, radii, pixel means, depths and conics.
    """
    _, camera_ids, gaussian_ids, radii, means2d, depths, conics, _ = fully_fused_projection(
        gaussians.means,
        None,
        gaussians.quaternions,
        gaussians.scales,
        views,
        intrinsics,
        width,
        height,
        packed=True,
        opacities=opacities,
    )
    # Invert each conic back to its pixel-space covariance.
    a, b, c = conics.unbind(-1)
    det = a * c - b * b
    xx, xy, yy = c / det, -b / det, a / det
    norm = torch.maximum(xx.abs() + xy.abs(), xy.abs() + yy.abs())
    pushed = (norm > alpha * width * height) & (depths < depth_limits[camera_ids])
    depths = torch.where(pushed, depths.clamp(min=PUSH_DEPTH_M), depths)
    return camera_ids, gaussian_ids, radii, means2d, depths, conics


def _lanczos(x: float) -> float:
    """Lanczos kernel with a = 3."""
    if not -3.0 <= x < 3.0:
        return 0.0
    if x == 0.0:
        return 1.0
    return math.sin(math.pi * x) * math.sin(math.pi * x / 3.0) / (math.pi * x * math.pi * x / 3.0)


@functools.cache
def lanczos_weights(size_in: int, size_out: int) -> np.ndarray:
    """Return the (size_out, size_in) antialiased Lanczos resampling weights of PIL.

    Each output pixel averages the input pixels within three output pixels of
    its center, with weights normalized over the pixels inside the image.
    """
    scale = size_in / size_out
    support = 3.0 * max(scale, 1.0)
    weights = np.zeros((size_out, size_in))
    for out in range(size_out):
        center = (out + 0.5) * scale
        low = max(int(center - support + 0.5), 0)
        high = min(int(center + support + 0.5), size_in)
        row = np.array([_lanczos((x - center + 0.5) / max(scale, 1.0)) for x in range(low, high)])
        weights[out, low:high] = row / row.sum()
    return weights


class _Resample:
    """Resample the last axis of float images with the weights of lanczos_weights, then round.

    For an integer downsampling factor, all rows of the weights are one
    kernel shifted by the factor, cut and renormalized at the image border.
    The pass is then a strided convolution that skips the zero weights.
    Other sizes multiply by the dense weights.
    """

    def __init__(self, size_in: int, size_out: int, device: torch.device):
        weights = lanczos_weights(size_in, size_out)
        self.dense = torch.as_tensor(weights, dtype=torch.float32, device=device)
        self.band = None
        factor = size_in // size_out
        if factor < 2 or factor * size_out != size_in:
            return
        middle = size_out // 2
        support = np.flatnonzero(weights[middle])
        kernel = weights[middle, support[0] : support[-1] + 1]
        left = middle * factor - support[0]
        right = (size_out - 1) * factor + len(kernel) - size_in - left
        # Rebuild the weights from the kernel to confirm the shared-kernel structure.
        rebuilt = np.zeros((size_out, size_in + left + right))
        for out in range(size_out):
            rebuilt[out, out * factor : out * factor + len(kernel)] = kernel
        rebuilt = rebuilt[:, left : left + size_in]
        if not np.allclose(rebuilt / rebuilt.sum(1, keepdims=True), weights, atol=1e-7):
            return
        self.factor, self.pads = factor, (left, right)
        self.band = torch.as_tensor(kernel, dtype=torch.float32, device=device).view(1, 1, -1)
        self.norm = self._convolve(torch.ones(1, size_in, device=device))

    def _convolve(self, x: torch.Tensor) -> torch.Tensor:
        flat = torch.nn.functional.pad(x.reshape(-1, 1, x.shape[-1]), self.pads)
        # Full float32 precision keeps the rounding of each pass that of the dense weights.
        with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
            out = torch.nn.functional.conv1d(flat, self.band, stride=self.factor)
        return out.view(*x.shape[:-1], -1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.dense.T if self.band is None else self._convolve(x) / self.norm
        return out.round().clamp(0, 255)


def intrinsics_matrix(width: int, height: int, vertical_fov_deg: float) -> np.ndarray:
    """Return the (3, 3) pinhole intrinsics with a centered principal point."""
    focal = height / (2.0 * math.tan(math.radians(vertical_fov_deg) / 2.0))
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], np.float32
    )


def _world_to_camera(pose: np.ndarray, transform: np.ndarray, height: float) -> np.ndarray:
    """Return a float32 4x4 camera view matrix in the original Gaussian frame.

    Pose is a scene pose of one_scan_is_enough.poses, transform maps original world to
    aligned coordinates, and the camera height is in meters above the aligned ground.
    """
    x, z, heading = pose
    rotation_align = transform[:3, :3]
    position = rotation_align.T @ (np.array([x, height, z]) - transform[:3, 3])
    forward = rotation_align.T @ np.array([math.sin(heading), 0.0, -math.cos(heading)])
    up = rotation_align.T @ np.array([0.0, 1.0, 0.0])
    forward /= np.linalg.norm(forward)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    rotation = np.stack([right, down, forward])
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ position
    return view


class Renderer:
    """Render views of one Gaussian scene with one pinhole camera configuration.

    The configuration gives the output width and height, supersample,
    vertical_fov_deg and culling. Pose views also need camera_height_m and
    JPEG output needs jpeg_quality. Rasterization uses the supersampled size,
    or size when it is given. Images are composited on white and downsampled
    to the output size. Oversized near Gaussians are pushed back per view as
    set in culling, against the content bounds of the construction scene.
    Views render together in batches, and JPEGs are encoded on the GPU.
    Call the methods in torch inference mode.
    """

    def __init__(
        self,
        scene: GaussianScene,
        transform: np.ndarray,
        config: DictConfig,
        size: tuple[int, int] | None = None,
    ):
        self.scene = scene
        self.transform = transform  # (4, 4), original world to aligned frame.
        self.config = config
        # Rasterization (width, height) in pixels.
        self.size = size or (config.width * config.supersample, config.height * config.supersample)
        self.intrinsics = intrinsics_matrix(*self.size, config.vertical_fov_deg)
        self.bounds = _content_bounds(scene, transform)
        # Downsampling to the output size runs on the GPU with the weights of PIL's Lanczos.
        self.resample = None
        if self.size != (config.width, config.height):
            device = scene.means.device
            self.resample = tuple(
                _Resample(size_in, size_out, device)
                for size_in, size_out in zip(self.size, (config.width, config.height))
            )

    def camera(self, pose: np.ndarray) -> np.ndarray:
        """Return the 4x4 view matrix of a scene pose at camera_height_m."""
        return _world_to_camera(pose, self.transform, self.config.camera_height_m)

    def rasterize(
        self,
        views: np.ndarray,
        gaussians: GaussianScene | None = None,
        opacities: torch.Tensor | None = None,
        mode: str = "RGB",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rasterize world views together and return (C, H, W, D) channels and (C, H, W) alphas.

        Memory grows with the view count, so callers pass at most BATCH_VIEWS views.

        Args:
            views: View matrices (C, 4, 4) in the original Gaussian frame.
            gaussians: Gaussians to draw instead of the scene, such as a subset of it.
            opacities: Opacities (N,) or per view (C, N) to use instead of
                those of the drawn Gaussians.
            mode: RGB for colors, or ED for expected depth.
        """
        gaussians = self.scene if gaussians is None else gaussians
        opacities = gaussians.opacities if opacities is None else opacities
        width, height = self.size
        device = gaussians.means.device
        views = np.asarray(views, dtype=np.float32)
        count = len(views)
        views_t = torch.as_tensor(views, device=device)
        intrinsics = torch.as_tensor(self.intrinsics, device=device).expand(count, 3, 3)
        centers = -np.einsum("nji,nj->ni", views[:, :3, :3], views[:, :3, 3])
        eyes_xz = (centers @ self.transform[:3, :3].T + self.transform[:3, 3])[:, [0, 2]]
        culling = self.config.culling
        limits = [_cull_depth_limit(eye, self.bounds, culling.depth_margin_m) for eye in eyes_xz]
        per_view = opacities.dim() == 2
        camera_ids, gaussian_ids, radii, means2d, depths, conics = _project(
            gaussians,
            opacities.amax(0) if per_view else opacities,
            views_t,
            intrinsics,
            width,
            height,
            torch.tensor(limits, dtype=torch.float32, device=device),
            culling.alpha,
        )
        if mode == "ED":
            colors = depths[:, None]
        elif mode == "RGB":
            # Pushing keeps view directions, so colors use the original centers.
            directions = (
                gaussians.means[gaussian_ids]
                - torch.as_tensor(centers, dtype=torch.float32, device=device)[camera_ids]
            )
            colors = spherical_harmonics(
                gaussians.sh_degree,
                directions,
                gaussians.spherical_harmonics[gaussian_ids],
                masks=(radii > 0).all(-1),
            )
            colors = (colors + 0.5).clamp(min=0.0)
        else:
            raise ValueError(f"Unknown render mode: {mode}")
        tiles = (math.ceil(width / TILE_SIZE), math.ceil(height / TILE_SIZE))
        _, intersections, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            TILE_SIZE,
            *tiles,
            packed=True,
            n_images=count,
            image_ids=camera_ids,
            gaussian_ids=gaussian_ids,
        )
        offsets = isect_offset_encode(intersections, count, *tiles).reshape(count, *tiles[::-1])
        rendered, alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities[camera_ids, gaussian_ids] if per_view else opacities[gaussian_ids],
            width,
            height,
            TILE_SIZE,
            offsets,
            flatten_ids,
            packed=True,
        )
        if mode == "ED":
            rendered = rendered / alphas.clamp(min=1e-10)
        return rendered, alphas[..., 0]

    def images(self, views: np.ndarray, opacities: torch.Tensor | None = None) -> torch.Tensor:
        """Render views as (C, 3, H, W) uint8 RGB images of the output size; see rasterize."""
        rendered, alphas = self.rasterize(views, opacities=opacities)
        # Composite here: packed gsplat 1.5.3 rejects batched backgrounds.
        rgb = (rendered + (1.0 - alphas[..., None])).clamp(0, 1).mul(255).byte()
        rgb = rgb.permute(0, 3, 1, 2)
        if self.resample is not None:
            # Horizontal then vertical pass, rounding after each as PIL does.
            columns, rows = self.resample
            rgb = rows(columns(rgb.float()).transpose(-1, -2)).transpose(-1, -2).byte()
        return rgb.contiguous()

    def jpegs(self, views: np.ndarray, opacities: torch.Tensor | None = None) -> list[bytes]:
        """Render views as JPEG bytes at jpeg_quality, encoded on the GPU.

        Views render in batches of BATCH_VIEWS; per-view opacities (C, N) are split alike.
        """
        frames: list[bytes] = []
        for start in range(0, len(views), BATCH_VIEWS):
            batch = slice(start, start + BATCH_VIEWS)
            chunk = (
                opacities[batch] if opacities is not None and opacities.dim() == 2 else opacities
            )
            images = list(self.images(views[batch], chunk))
            # The encoder reads on its own stream without waiting for the render kernels.
            torch.cuda.current_stream(images[0].device).synchronize()
            encoded = encode_jpeg(images, quality=self.config.jpeg_quality)
            frames += [data.cpu().numpy().tobytes() for data in encoded]
        return frames

    def jpeg(self, view: np.ndarray) -> bytes:
        """Render one view as JPEG bytes at jpeg_quality."""
        return self.jpegs(view[None])[0]


@torch.inference_mode()
def render_route(
    scene: GaussianScene,
    poses: np.ndarray,
    transform: np.ndarray,
    config: DictConfig,
    indices: np.ndarray,
) -> list[bytes]:
    """Render selected pose views (N, 3) as JPEGs with markers along the future route.

    The configuration is that of Renderer with the path_* marker settings.
    """
    base_opacities = scene.opacities
    scene, marker_s, distances = _route_markers(scene, poses, transform, config)
    renderer = Renderer(scene, transform, config)
    # Hide past markers and fade the path immediately behind each camera.
    fade = max(1e-6, config.path_fade_m)
    ahead = distances[indices] - fade
    visibility = (
        (
            marker_s[None]
            - torch.as_tensor(ahead, dtype=torch.float32, device=marker_s.device)[:, None]
        )
        / fade
    ).clamp(0.0, 1.0)
    marker_alpha = config.path_opacity_min + visibility * (
        config.path_opacity - config.path_opacity_min
    )
    opacities = torch.cat([base_opacities.expand(len(indices), -1), marker_alpha], 1)
    views = np.stack([renderer.camera(poses[index]) for index in indices])
    return renderer.jpegs(views, opacities)
