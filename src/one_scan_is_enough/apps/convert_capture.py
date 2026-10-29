"""Hydra entry point for converting an RGB-D capture to a TUM RGB-D directory."""

from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="apps/convert_capture")
def main(config: DictConfig) -> None:
    """Convert a Stray Scanner or RealSense capture for ORB-SLAM3 and export-pgsr."""
    from one_scan_is_enough.capture import realsense, stray

    OmegaConf.resolve(config)
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Missing configuration: {sorted(missing)}")
    capture = Path(to_absolute_path(config.input.capture_dir))
    output = Path(to_absolute_path(config.output.tum_dir))
    if config.input.format == "stray":
        stray.convert(config.stray, capture, output)
    elif config.input.format == "realsense":
        realsense.convert(config.realsense, capture, output)
    else:
        raise ValueError(f"Unknown capture format: {config.input.format}")


if __name__ == "__main__":
    main()
