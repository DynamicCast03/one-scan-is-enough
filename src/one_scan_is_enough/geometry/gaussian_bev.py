"""Project bounded Gaussian batches into aligned XZ occupancy and color maps."""

import logging
import math

import cv2
import numpy as np
from numba import njit
from tqdm import tqdm

from one_scan_is_enough.geometry.gaussian_ply import iter_geometry
from one_scan_is_enough.scene import BevMap

# Gray level of color-view cells that no Gaussian covers.
EMPTY_COLOR = 255

logger = logging.getLogger(__name__)


def aligned_batches(path, transform, config, scale_cap, *, with_colors=False):
    """Yield centers, covariance matrices, opacity, and colors in the aligned frame.

    Colors are None unless with_colors is set.
    """
    rotation = transform[:3, :3]
    for batch in iter_geometry(
        path, config.batch_size, scale_max=scale_cap, with_colors=with_colors
    ):
        keep = batch.opacities >= config.opacity_min
        centers = batch.xyz[keep] @ rotation.T + transform[:3, 3]  # (B, 3)
        axes = rotation @ batch.rotations[keep].astype(np.float64)  # (B, 3, 3)
        axes *= batch.scales[keep, None, :]
        covariance = axes @ axes.transpose(0, 2, 1)  # (B, 3, 3)
        # A one-micrometer variance floor keeps thin ellipsoids invertible.
        covariance[:, np.arange(3), np.arange(3)] += 1e-12
        colors = batch.colors[keep] if with_colors else None
        yield centers, covariance, batch.opacities[keep], colors


def map_bounds(path, transform, config, scale_cap):
    """Return XZ limits of all retained, truncated Gaussian ellipsoids."""
    lower = np.full(2, np.inf)
    upper = np.full(2, -np.inf)
    for centers, covariance, _, _ in aligned_batches(path, transform, config, scale_cap):
        if not len(centers):
            continue
        radii = config.support_sigma * np.sqrt(covariance[:, [0, 2], [0, 2]])
        lower = np.minimum(lower, (centers[:, [0, 2]] - radii).min(axis=0))
        upper = np.maximum(upper, (centers[:, [0, 2]] + radii).max(axis=0))
    if not np.isfinite(lower).all():
        raise ValueError("No Gaussians remain after opacity filtering.")
    return lower, upper


@njit(cache=True)
def splat_batch(
    centers,
    covariance,
    opacity,
    colors,
    origin,
    resolution,
    y_min,
    y_max,
    ground_y_min,
    ground_y_max,
    top_y_max,
    top_alpha_min,
    support_sigma,
    occupancy,
    coverage,
    ground_sum,
    ground_weight,
    top_height,
    top_color,
):
    """Update maps in place with slab maxima, scene support, and two color views.

    Cell (row, col) is centered at origin + index * resolution. Occupancy uses alpha union; coverage uses the same alpha union across all heights.
    The ground view averages colors of centers in [ground_y_min, ground_y_max],
    weighted by footprint alpha. The top view keeps, per cell, the color of the
    highest center below top_y_max whose footprint alpha reaches top_alpha_min.
    Each call writes sequentially to avoid races between overlapping splats.
    """
    height, width = occupancy.shape
    cutoff = support_sigma * support_sigma
    for i in range(len(centers)):
        mu = centers[i]
        cov = covariance[i]
        rx = support_sigma * math.sqrt(cov[0, 0])
        rz = support_sigma * math.sqrt(cov[2, 2])
        ry = support_sigma * math.sqrt(cov[1, 1])
        intersects = mu[1] + ry >= y_min and mu[1] - ry <= y_max
        in_ground = ground_y_min <= mu[1] <= ground_y_max
        in_top = mu[1] <= top_y_max
        x0 = max(0, int(math.ceil((mu[0] - rx - origin[0]) / resolution)))
        x1 = min(width - 1, int(math.floor((mu[0] + rx - origin[0]) / resolution)))
        z0 = max(0, int(math.ceil((mu[2] - rz - origin[1]) / resolution)))
        z1 = min(height - 1, int(math.floor((mu[2] + rz - origin[1]) / resolution)))

        # Partition covariance into horizontal and conditional height terms.
        det = cov[0, 0] * cov[2, 2] - cov[0, 2] ** 2
        inv_xx = cov[2, 2] / det
        inv_xz = -cov[0, 2] / det
        inv_zz = cov[0, 0] / det
        slope_x = cov[1, 0] * inv_xx + cov[1, 2] * inv_xz
        slope_z = cov[1, 0] * inv_xz + cov[1, 2] * inv_zz
        variance = max(1e-12, cov[1, 1] - slope_x * cov[0, 1] - slope_z * cov[2, 1])
        for row in range(z0, z1 + 1):
            dz = origin[1] + row * resolution - mu[2]
            for col in range(x0, x1 + 1):
                dx = origin[0] + col * resolution - mu[0]
                q = inv_xx * dx * dx + 2 * inv_xz * dx * dz + inv_zz * dz * dz
                if q > cutoff:
                    continue
                full_alpha = opacity[i] * math.exp(-0.5 * q)
                coverage[row, col] += (1.0 - coverage[row, col]) * full_alpha
                if in_ground:
                    for channel in range(3):
                        ground_sum[row, col, channel] += full_alpha * colors[i, channel]
                    ground_weight[row, col] += full_alpha
                if in_top and full_alpha >= top_alpha_min and mu[1] > top_height[row, col]:
                    top_height[row, col] = mu[1]
                    for channel in range(3):
                        top_color[row, col, channel] = int(colors[i, channel] * 255.0 + 0.5)
                if not intersects:
                    continue
                conditional_y = mu[1] + slope_x * dx + slope_z * dz
                nearest_y = min(y_max, max(y_min, conditional_y))
                distance = q + (nearest_y - conditional_y) ** 2 / variance
                if distance <= cutoff:
                    alpha = opacity[i] * math.exp(-0.5 * distance)
                    occupancy[row, col] += (1.0 - occupancy[row, col]) * alpha


def build_gaussian_bev(path, transform, gaussian_config, bev_config, scale_cap):
    """Build the raw and navigable BEV maps and RGB views from Gaussians.

    Gaussian work buffers are bounded by batch_size. The scale_cap cutoff,
    in meters, is shared with ground sampling. Map size follows scene bounds.
    Otsu thresholds each score map over the complete grid. Obstacle closing
    and navigable component filtering clean the map. Planning owns clearance.
    Returns the map and a dict of (H, W, 3) uint8 RGB views, "top" and "ground",
    on the same grid; they are for manual annotation only.
    """
    logger.info("Computing Gaussian map bounds")
    lower, upper = map_bounds(path, transform, gaussian_config, scale_cap)
    resolution = float(bev_config.resolution)
    origin = lower - 10 * resolution  # (2,), aligned XZ coordinates in meters.
    width, height = np.ceil((upper - lower) / resolution + 20).astype(np.int64)
    logger.info("Projecting Gaussians into %s x %s cells", width, height)
    occupancy = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)
    ground_sum = np.zeros((height, width, 3), dtype=np.float32)
    ground_weight = np.zeros((height, width), dtype=np.float32)
    top_height = np.full((height, width), -np.inf, dtype=np.float32)
    top_color = np.full((height, width, 3), EMPTY_COLOR, dtype=np.uint8)
    for centers, covariance, opacity, colors in tqdm(
        aligned_batches(path, transform, gaussian_config, scale_cap, with_colors=True),
        desc="Gaussian BEV",
        unit="batch",
    ):
        splat_batch(
            centers,
            covariance,
            opacity,
            colors,
            origin,
            resolution,
            bev_config.y_min,
            bev_config.y_max,
            bev_config.ground_color_y_min,
            bev_config.ground_color_y_max,
            bev_config.top_color_y_max,
            bev_config.top_color_alpha_min,
            gaussian_config.support_sigma,
            occupancy,
            coverage,
            ground_sum,
            ground_weight,
            top_height,
            top_color,
        )
    del top_height
    views = {
        "top": top_color,
        "ground": average_colors(ground_sum, ground_weight),
    }
    del ground_sum, ground_weight
    obstacles = otsu_mask(occupancy, "obstacles")
    del occupancy
    observed = otsu_mask(coverage, "coverage")
    del coverage
    # Fill coverage holes while keeping obstacles in the separate height-slab map.
    contours, _ = cv2.findContours(observed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    observed.fill(0)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(observed, [contour], 0, 255, thickness=-1)
    raw = cv2.bitwise_and(observed, cv2.bitwise_not(obstacles))
    del observed, obstacles
    # Close small obstacle gaps so enclosed free-space pockets can be removed.
    # The configured width is rounded up to an odd number of cells.
    kernel_size = int(np.ceil(bev_config.obstacle_close_size_m / resolution)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blocked = cv2.morphologyEx(cv2.bitwise_not(raw), cv2.MORPH_CLOSE, kernel)
    safe = cv2.bitwise_not(blocked)
    del blocked
    logger.info("Closing obstacles with a %s x %s cell kernel", kernel_size, kernel_size)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(safe, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] * resolution**2 >= bev_config.min_component_area_m2
    keep[0] = False
    safe = keep[labels].astype(np.uint8) * 255
    logger.info("Navigable components: %s -> %s", count - 1, int(keep.sum()))
    return BevMap(raw, safe, origin, resolution), views


def average_colors(color_sum, weight):
    """Return uint8 weighted mean colors; cells without weight get EMPTY_COLOR.

    The float sums are divided in place.
    """
    empty = weight < 1e-3
    color_sum /= np.maximum(weight, 1e-3)[..., None]
    colors = np.rint(np.clip(color_sum, 0.0, 1.0) * 255).astype(np.uint8)
    colors[empty] = EMPTY_COLOR
    return colors


def otsu_mask(scores, name):
    """Return a 0/255 mask from a score map in [0, 1].

    Quantize to 256 levels and apply Otsu over all grid cells, including zeros.
    Cells strictly above the selected threshold receive 255.
    """
    quantized = np.rint(scores * 255).astype(np.uint8)
    threshold, mask = cv2.threshold(quantized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    logger.info("Otsu %s threshold: %.6f (%s/255)", name, threshold / 255, int(threshold))
    return mask
