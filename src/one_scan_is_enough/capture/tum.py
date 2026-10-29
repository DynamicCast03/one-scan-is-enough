"""TUM RGB-D layouts: associations, trajectories and ORB-SLAM3 camera settings.

A TUM directory holds rgb/ and depth/ images named by timestamp, association.txt
pairing them, tum.yaml with the pinhole camera for ORB-SLAM3, and optionally
trajectories in TUM format: timestamp tx ty tz qx qy qz qw, camera to world in
OpenCV camera axes. Depth images are uint16 PNGs in millimeters.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    """Pinhole intrinsics in pixels of images width x height."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float = 1000.0  # Depth image units per meter.


@dataclass
class Association:
    timestamp: float
    rgb: str  # Path relative to the TUM directory.
    depth: str


def write_camera(path: Path, camera: Camera, fps: int) -> None:
    """Write tum.yaml, the ORB-SLAM3 RGB-D settings of a camera."""
    lines = [
        "%YAML:1.0",
        "",
        'File.version: "1.0"',
        "",
        'Camera.type: "PinHole"',
        "",
        "# Camera calibration and distortion parameters (OpenCV)",
        f"Camera1.fx: {camera.fx}",
        f"Camera1.fy: {camera.fy}",
        f"Camera1.cx: {camera.cx}",
        f"Camera1.cy: {camera.cy}",
        "",
        "Camera1.k1: 0.0",
        "Camera1.k2: 0.0",
        "Camera1.p1: 0.0",
        "Camera1.p2: 0.0",
        "Camera1.k3: 0.0",
        "",
        f"Camera.fps: {fps}",
        "Camera.RGB: 1",
        f"Camera.width: {camera.width}",
        f"Camera.height: {camera.height}",
        "",
        "ORBextractor.nFeatures: 1000",
        "ORBextractor.scaleFactor: 1.2",
        "ORBextractor.nLevels: 8",
        "ORBextractor.iniThFAST: 20",
        "ORBextractor.minThFAST: 7",
        f"RGBD.DepthMapFactor: {camera.depth_scale}",
        "Stereo.ThDepth: 40.0",
        "Stereo.b: 0.075",
        "Viewer.KeyFrameSize: 0.05",
        "Viewer.KeyFrameLineWidth: 1.0",
        "Viewer.GraphLineWidth: 0.9",
        "Viewer.PointSize: 2.0",
        "Viewer.CameraSize: 0.08",
        "Viewer.CameraLineWidth: 3.0",
        "Viewer.ViewpointX: 0.0",
        "Viewer.ViewpointY: -0.7",
        "Viewer.ViewpointZ: -1.8",
        "Viewer.ViewpointF: 500.0",
    ]
    path.write_text("\n".join(lines) + "\n")


def read_camera(path: Path) -> Camera:
    """Read the camera of a tum.yaml."""
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "%")) or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        try:
            values[key] = int(value) if "." not in value else float(value)
        except ValueError:
            values[key] = value.strip('"')
    return Camera(
        fx=values["Camera1.fx"],
        fy=values["Camera1.fy"],
        cx=values["Camera1.cx"],
        cy=values["Camera1.cy"],
        width=values["Camera.width"],
        height=values["Camera.height"],
        depth_scale=values["RGBD.DepthMapFactor"],
    )


def write_associations(path: Path, associations: list[Association]) -> None:
    with open(path, "w") as file:
        for item in associations:
            stamp = f"{item.timestamp:.6f}"
            file.write(f"{stamp} {item.rgb} {stamp} {item.depth}\n")


def read_associations(path: Path) -> list[Association]:
    associations = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if line.startswith("#") or len(parts) < 4:
            continue
        associations.append(Association(float(parts[0]), parts[1], parts[3]))
    return associations


def write_trajectory(path: Path, timestamps, positions, quaternions) -> None:
    """Write camera-to-world poses; quaternions are (N, 4) as x, y, z, w."""
    with open(path, "w") as file:
        for stamp, (tx, ty, tz), (qx, qy, qz, qw) in zip(timestamps, positions, quaternions):
            file.write(
                f"{stamp:.6f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )


def read_trajectory(path: Path) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Return camera-to-world (position, x-y-z-w quaternion) by timestamp."""
    trajectory = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if line.startswith("#") or len(parts) < 8:
            continue
        values = [float(value) for value in parts[:8]]
        trajectory[values[0]] = (np.array(values[1:4]), np.array(values[4:8]))
    return trajectory
