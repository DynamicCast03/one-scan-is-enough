"""Hydra entry point for a complete, calibrated scene package."""

from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="apps/prepare_scene")
def main(config: DictConfig) -> None:
    """Resolve input paths and pass each module its own configuration."""
    from one_scan_is_enough.preparation.pipeline import prepare

    OmegaConf.resolve(config)
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Missing configuration: {sorted(missing)}")
    prepare(
        config,
        Path(to_absolute_path(config.input.gaussian_ply)),
        Path(to_absolute_path(config.output.scene_dir)),
    )


if __name__ == "__main__":
    main()
