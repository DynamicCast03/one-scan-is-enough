"""Convert Stray Scanner captures to TUM RGB-D directories.

A capture holds rgb.mp4, depth/NNNNNN.png at a lower resolution in millimeters,
camera_matrix.csv with the RGB intrinsics and odometry.csv with ARKit camera
poses per frame, camera to world in OpenCV camera axes as the Stray tools use
them. The conversion selects frames, extracts their RGB images, upsamples their
depth to the RGB resolution with an RGB-guided filter and writes the ARKit
poses as groundtruth.txt next to the ORB-SLAM3 inputs.
"""

import logging
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from one_scan_is_enough.capture.tum import (
    Association,
    Camera,
    write_associations,
    write_camera,
    write_trajectory,
)

logger = logging.getLogger(__name__)

# Guided filter of depth upsampling: radius in pixels and regularization eps.
GUIDE_RADIUS = 4
GUIDE_EPS = 0.01


@dataclass
class Frame:
    index: int  # Video frame index, also the depth image index.
    timestamp: float
    position: np.ndarray  # (3,) camera position in the ARKit world.
    quaternion: np.ndarray  # (4,) camera-to-world rotation as x, y, z, w.


def read_odometry(path: Path) -> dict[int, Frame]:
    """Read odometry.csv rows keyed by frame index."""
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    header = [field.strip().lower() for field in lines[0].split(",")]
    column = {
        name: header.index(name)
        for name in ("timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw")
    }
    frames = {}
    for line in lines[1:]:
        fields = [field.strip() for field in line.split(",")]
        index = int(fields[column["frame"]])
        if index in frames:
            raise ValueError(f"Duplicated odometry frame {index} in {path}")
        frames[index] = Frame(
            index,
            float(fields[column["timestamp"]]),
            np.array([float(fields[column[name]]) for name in ("x", "y", "z")]),
            np.array([float(fields[column[name]]) for name in ("qx", "qy", "qz", "qw")]),
        )
    return frames


def read_camera(path: Path, width: int, height: int) -> Camera:
    """Read the 3x3 intrinsics of camera_matrix.csv for RGB frames of width x height."""
    rows = [
        line.replace(",", " ").split() for line in path.read_text().splitlines() if line.strip()
    ]
    return Camera(
        fx=float(rows[0][0]),
        fy=float(rows[1][1]),
        cx=float(rows[0][2]),
        cy=float(rows[1][2]),
        width=width,
        height=height,
    )


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian of the gray image; higher is sharper."""
    return float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def select_frames(
    frames: list[Frame], video: Path, video_fps: float, config: DictConfig, workers: int
) -> list[Frame]:
    """Keep frames in config.frame_range, then downsample them to config.fps.

    Downsampling keeps the first frame of each interval, or with
    config.sharpest the sharpest one and at least config.min_frames frames.
    """
    if config.frame_range is not None:
        start, end = config.frame_range
        frames = [frame for frame in frames if start <= frame.index <= end]
    if config.fps is None or abs(video_fps - config.fps) < 1e-6:
        return frames
    if config.fps > video_fps:
        raise ValueError(f"Target FPS {config.fps} exceeds the video FPS {video_fps}")
    interval = video_fps / config.fps
    if not config.sharpest:
        selected, position = [], 0.0
        for row, frame in enumerate(frames):
            if row >= int(position):
                selected.append(frame)
                position += interval
        return selected

    scores = dict(
        zip(
            (frame.index for frame in frames),
            video_sharpness(video, {f.index for f in frames}, workers),
        )
    )
    frames = [frame for frame in frames if not math.isnan(scores[frame.index])]
    chosen = []
    for interval_index in range(int(math.ceil(len(frames) / interval))):
        start = int(math.floor(interval_index * interval))
        end = min(max(int(math.floor((interval_index + 1) * interval)), start + 1), len(frames))
        if start >= len(frames):
            break
        chosen.append(max(range(start, end), key=lambda row: (scores[frames[row].index], -row)))
    if len(chosen) < config.min_frames:
        rest = sorted(
            set(range(len(frames))) - set(chosen),
            key=lambda row: scores[frames[row].index],
            reverse=True,
        )
        chosen += rest[: config.min_frames - len(chosen)]
    return [frames[row] for row in sorted(set(chosen))]


def video_sharpness(video: Path, indices: set[int], workers: int) -> list[float]:
    """Return sharpness by ascending frame index; undecodable frames are NaN."""
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index in tqdm(range(total), desc="Sharpness", unit="frame"):
            ok, image = capture.read()
            if ok and image is not None and index in indices:
                futures[index] = executor.submit(sharpness, image)
    capture.release()
    return [
        futures[index].result() if index in futures else float("nan") for index in sorted(indices)
    ]


def upsample_depth(depth: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Resize depth to the RGB size and refine it with the RGB-guided filter."""
    height, width = rgb.shape[:2]
    upsampled = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    values = upsampled.astype(np.float32)
    peak = values.max()
    normalized = values / peak if peak > 0 else values
    filtered = cv2.ximgproc.guidedFilter(
        guide=rgb.astype(np.float32) / 255.0,
        src=normalized,
        radius=GUIDE_RADIUS,
        eps=GUIDE_EPS * GUIDE_EPS,
    )
    if peak > 0:
        filtered = filtered * peak
    return filtered.astype(depth.dtype)


def convert(config: DictConfig, capture: Path, output: Path) -> Path:
    """Write the TUM directory of a Stray Scanner capture and return it."""
    video = capture / "rgb.mp4"
    reader = cv2.VideoCapture(str(video))
    total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(reader.get(cv2.CAP_PROP_FPS))
    width, height = (
        int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    reader.release()
    depth_dir = capture / "depth"
    depth_indices = sorted(int(path.stem) for path in depth_dir.glob("*.png"))
    if depth_indices != list(range(total)):
        raise ValueError(f"Depth images of {capture} do not match its {total} video frames")
    odometry = read_odometry(capture / "odometry.csv")
    frames = [odometry[index] for index in range(total) if index in odometry]
    logger.info(
        "Capture %s: %d frames at %.3f FPS, %dx%d, %d with poses",
        capture,
        total,
        video_fps,
        width,
        height,
        len(frames),
    )
    frames = select_frames(frames, video, video_fps, config, config.workers)
    logger.info("Selected %d frames", len(frames))

    extension = ".jpg" if config.jpeg else ".png"
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    (output / "depth").mkdir(parents=True, exist_ok=True)
    by_index = {frame.index: frame for frame in frames}
    written = []

    def write_frame(frame: Frame, image: np.ndarray) -> Frame | None:
        name = f"{frame.timestamp:.6f}"
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if config.jpeg else [cv2.IMWRITE_PNG_COMPRESSION, 1]
        rgb_path = output / "rgb" / f"{name}{extension}"
        if not cv2.imwrite(str(rgb_path), image, params):
            return None
        # The decoded video frame guides the depth upsampling filter.
        depth = cv2.imread(str(depth_dir / f"{frame.index:06d}.png"), cv2.IMREAD_UNCHANGED)
        if not cv2.imwrite(str(output / "depth" / f"{name}.png"), upsample_depth(depth, image)):
            return None
        return frame

    # Frames run in parallel workers, so OpenCV runs single-threaded within each.
    cv2.setNumThreads(1)
    reader = cv2.VideoCapture(str(video))
    # Decoding outpaces writing, so at most two frames per worker wait in memory.
    pending = deque()
    with (
        ThreadPoolExecutor(max_workers=config.workers) as executor,
        tqdm(total=len(frames), desc="Frames", unit="frame") as progress,
    ):

        def collect() -> None:
            frame = pending.popleft().result()
            if frame is not None:
                written.append(frame)
            progress.update()

        for index in range(total):
            ok, image = reader.read()
            if ok and image is not None and index in by_index:
                if len(pending) >= 2 * config.workers:
                    collect()
                pending.append(executor.submit(write_frame, by_index[index], image))
        while pending:
            collect()
    reader.release()
    if len(written) < len(frames):
        logger.warning(
            "Dropped %d frames that failed to decode or write", len(frames) - len(written)
        )

    write_camera(
        output / "tum.yaml",
        read_camera(capture / "camera_matrix.csv", width, height),
        int(round(config.fps if config.fps is not None else video_fps)),
    )
    write_associations(
        output / "association.txt",
        [
            Association(
                frame.timestamp,
                f"rgb/{frame.timestamp:.6f}{extension}",
                f"depth/{frame.timestamp:.6f}.png",
            )
            for frame in written
        ],
    )
    write_trajectory(
        output / "groundtruth.txt",
        [frame.timestamp for frame in written],
        [frame.position for frame in written],
        [frame.quaternion for frame in written],
    )
    logger.info("Wrote %d frames to %s", len(written), output)
    return output
