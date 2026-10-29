"""Export TUM RGB-D directories with camera poses as PGSR scene inputs.

The poses are a TUM trajectory, camera to world in OpenCV camera axes: the
ORB-SLAM3 CameraTrajectory.txt or the ARKit groundtruth.txt of a Stray Scanner
capture. The output holds images/, depths/ in millimeters and sparse/ with
cameras.txt, images.txt, depth_params.json of unit scales and points3D.ply,
a colored point cloud backprojected from the depth maps.
"""

import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from one_scan_is_enough.capture.tum import Camera, read_associations, read_camera, read_trajectory

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    name: str  # Output image name.
    rgb: Path
    depth: Path
    position: np.ndarray  # (3,) camera to world translation.
    quaternion: np.ndarray  # (4,) camera to world rotation as x, y, z, w.


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


def matrix_to_quaternion(r: np.ndarray) -> np.ndarray:
    """Return a rotation matrix as a w, x, y, z quaternion."""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        return np.array(
            [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
        )
    if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        return np.array(
            [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
        )
    if r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        return np.array(
            [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
        )
    s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
    return np.array(
        [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    )


def world_to_camera(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """Return the COLMAP translation and w, x, y, z quaternion of a frame."""
    rotation = quaternion_to_matrix(frame.quaternion).T
    return -rotation @ frame.position, matrix_to_quaternion(rotation)


def depth_samples(depth_path: Path, camera: Camera, config: DictConfig):
    """Return the pixel grid (u, v) and depths in meters within the depth range."""
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    if depth.shape[1::-1] != (camera.width, camera.height):
        depth = cv2.resize(depth, (camera.width, camera.height), interpolation=cv2.INTER_NEAREST)
    step = config.image_downsample
    depth = depth[::step, ::step]
    v, u = np.meshgrid(
        np.arange(0, depth.shape[0], config.stride),
        np.arange(0, depth.shape[1], config.stride),
        indexing="ij",
    )
    u, v = u.ravel(), v.ravel()
    z = depth[v, u].astype(np.float32) / camera.depth_scale
    keep = (z > 0) & (z < config.max_depth_m)
    return u[keep], v[keep], z[keep]


def count_points(args) -> int:
    depth_path, camera, config = args
    samples = depth_samples(depth_path, camera, config)
    return 0 if samples is None else len(samples[2])


def backproject(args):
    """Backproject at most quota depth samples of a frame to world points with colors."""
    frame, camera, config, quota, seed = args
    samples = depth_samples(frame.depth, camera, config)
    rgb = cv2.imread(str(frame.rgb))
    if samples is None or rgb is None or quota == 0 or len(samples[2]) == 0:
        return None
    u, v, z = samples
    if len(z) > quota:
        keep = np.random.default_rng(seed).choice(len(z), quota, replace=False)
        u, v, z = u[keep], v[keep], z[keep]
    step = config.image_downsample
    rgb = rgb[::step, ::step]
    points = np.stack(
        [
            (u - camera.cx / step) * z / (camera.fx / step),
            (v - camera.cy / step) * z / (camera.fy / step),
            z,
        ],
        1,
    )
    world = points @ quaternion_to_matrix(frame.quaternion).T + frame.position
    return world, rgb[v, u][:, ::-1]


def write_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write points with zero normals and uint8 colors, the PLY that PGSR reads."""
    vertices = np.empty(
        len(points),
        dtype=[(n, "f4") for n in ("x", "y", "z", "nx", "ny", "nz")]
        + [(n, "u1") for n in ("red", "green", "blue")],
    )
    for i, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, i]
        vertices["n" + name] = 0.0
    for i, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, i]
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def export(config: DictConfig, tum: Path, trajectory_path: Path, output: Path) -> Path:
    """Write the PGSR input of every config.frame_step-th associated frame."""
    camera = read_camera(tum / "tum.yaml")
    trajectory = read_trajectory(trajectory_path)
    stamps = np.array(sorted(trajectory))
    extension = ".jpg" if config.jpeg else ".png"
    frames = []
    for index, item in enumerate(read_associations(tum / "association.txt")):
        if index % config.frame_step:
            continue
        rgb, depth = tum / item.rgb, tum / item.depth
        if not rgb.exists() or not depth.exists():
            continue
        # Each frame takes the pose nearest in time.
        position, quaternion = trajectory[stamps[np.argmin(np.abs(stamps - item.timestamp))]]
        frames.append(Frame(f"{index:06d}{extension}", rgb, depth, position, quaternion))
    logger.info(
        "Exporting %d frames of %s with poses from %s", len(frames), tum, trajectory_path.name
    )

    sparse = output / "sparse"
    for directory in (sparse, output / "images", output / "depths"):
        directory.mkdir(parents=True, exist_ok=True)
    (sparse / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {camera.width} {camera.height} {camera.fx} {camera.fy} {camera.cx} {camera.cy}\n"
    )
    with open(sparse / "images.txt", "w") as file:
        file.write("# Image list with two lines of data per image:\n")
        file.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        file.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id, frame in enumerate(frames, start=1):
            t, q = world_to_camera(frame)
            file.write(
                f"{image_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {frame.name}\n\n"
            )
    depth_params = {frame.name.rsplit(".", 1)[0]: {"scale": 1.0, "offset": 0.0} for frame in frames}
    (sparse / "depth_params.json").write_text(json.dumps(depth_params, indent=2))

    def copy(frame: Frame) -> None:
        image_path = output / "images" / frame.name
        # Images already in the output format are copied without re-encoding.
        if frame.rgb.suffix.lower() == image_path.suffix:
            shutil.copy2(frame.rgb, image_path)
        else:
            params = [cv2.IMWRITE_JPEG_QUALITY, 95] if config.jpeg else []
            cv2.imwrite(str(image_path), cv2.imread(str(frame.rgb), cv2.IMREAD_COLOR), params)
        shutil.copy2(frame.depth, output / "depths" / (frame.name.rsplit(".", 1)[0] + ".png"))

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        list(tqdm(executor.map(copy, frames), total=len(frames), desc="Images", unit="frame"))

    # Each frame contributes points in proportion to its valid depth samples.
    cloud = config.point_cloud
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        counts = np.array(
            list(
                tqdm(
                    executor.map(count_points, [(f.depth, camera, cloud) for f in frames]),
                    total=len(frames),
                    desc="Depth samples",
                    unit="frame",
                )
            ),
            dtype=np.int64,
        )
        total = int(counts.sum())
        quotas = counts
        if total > cloud.target_points:
            expected = counts * (cloud.target_points / total)
            quotas = np.floor(expected).astype(np.int64)
            remaining = int(cloud.target_points - quotas.sum())
            if remaining > 0:
                quotas[np.argpartition(expected - quotas, -remaining)[-remaining:]] += 1
        tasks = [
            (frame, camera, cloud, int(quota), [config.seed, i])
            for i, (frame, quota) in enumerate(zip(frames, quotas))
        ]
        results = [
            r
            for r in tqdm(
                executor.map(backproject, tasks), total=len(tasks), desc="Point cloud", unit="frame"
            )
            if r is not None
        ]
    points = np.concatenate([r[0] for r in results])
    colors = np.concatenate([r[1] for r in results])
    write_point_cloud(sparse / "points3D.ply", points, colors)
    logger.info("Wrote %d frames and %d points to %s", len(frames), len(points), output)
    return output
