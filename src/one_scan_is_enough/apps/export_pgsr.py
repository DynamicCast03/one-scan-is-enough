"""Hydra entry point for exporting a TUM RGB-D directory with poses as PGSR inputs."""

from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# TUM trajectory file of each pose source.
POSE_FILES = {"orbslam3": "CameraTrajectory.txt", "arkit": "groundtruth.txt"}


@hydra.main(version_base="1.3", config_path="../config", config_name="apps/export_pgsr")
def main(config: DictConfig) -> None:
    """Write the images, depths, cameras, poses and point cloud that PGSR trains on."""
    from one_scan_is_enough.capture.colmap import export

    OmegaConf.resolve(config)
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Missing configuration: {sorted(missing)}")
    if config.input.poses not in POSE_FILES:
        raise ValueError(f"Unknown pose source: {config.input.poses}")
    tum = Path(to_absolute_path(config.input.tum_dir))
    export(
        config.export,
        tum,
        tum / POSE_FILES[config.input.poses],
        Path(to_absolute_path(config.output.scene_dir)),
    )


if __name__ == "__main__":
    main()
