"""Convert RealSense recordings to TUM RGB-D directories.

A recording holds rgb/N.jpg and depth/N.png aligned to the color camera, with
contiguous indices N, and color_intrinsics.json. It has no poses, so its TUM
directory is meant for ORB-SLAM3. Timestamps are frame indices over the
recording FPS.
"""

import json
import logging
import shutil
from pathlib import Path

from omegaconf import DictConfig
from tqdm import tqdm

from one_scan_is_enough.capture.tum import Association, Camera, write_associations, write_camera

logger = logging.getLogger(__name__)


def indexed_files(directory: Path, pattern: str) -> list[Path]:
    """Return files named by contiguous integers from 0, in index order."""
    files = sorted(
        (path for path in directory.glob(pattern) if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if not files or [int(path.stem) for path in files] != list(range(len(files))):
        raise ValueError(f"{directory} needs files {pattern} indexed contiguously from 0")
    return files


def convert(config: DictConfig, recording: Path, output: Path) -> Path:
    """Write the TUM directory of a RealSense recording and return it."""
    intrinsics = json.loads((recording / "color_intrinsics.json").read_text())
    rgb_files = indexed_files(recording / "rgb", "*.jpg")
    depth_files = indexed_files(recording / "depth", "*.png")
    if len(rgb_files) != len(depth_files):
        raise ValueError(
            f"{recording} has {len(rgb_files)} RGB and {len(depth_files)} depth images"
        )
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    (output / "depth").mkdir(parents=True, exist_ok=True)
    associations = []
    for index, (rgb, depth) in enumerate(
        tqdm(list(zip(rgb_files, depth_files)), desc="Frames", unit="frame")
    ):
        timestamp = index / config.fps
        stem = f"{timestamp:.6f}"
        for source, target in (
            (rgb, output / "rgb" / f"{stem}.jpg"),
            (depth, output / "depth" / f"{stem}.png"),
        ):
            if target.exists() or target.is_symlink():
                target.unlink()
            if config.symlink:
                target.symlink_to(source.resolve())
            else:
                shutil.copy2(source, target)
        associations.append(Association(timestamp, f"rgb/{stem}.jpg", f"depth/{stem}.png"))

    camera = Camera(
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["ppx"],
        cy=intrinsics["ppy"],
        width=intrinsics["width"],
        height=intrinsics["height"],
        depth_scale=config.depth_scale,
    )
    write_camera(output / "tum.yaml", camera, int(round(config.fps)))
    write_associations(output / "association.txt", associations)
    for name in ("color_intrinsics.json", "depth_intrinsics.json"):
        if (recording / name).exists():
            shutil.copy2(recording / name, output / name)
    logger.info("Wrote %d frames to %s; run ORB-SLAM3 for poses", len(associations), output)
    return output
