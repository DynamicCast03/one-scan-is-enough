"""Hydra entry point for planning paths in a calibrated scene."""

from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="apps/plan_trajectories")
def main(config: DictConfig) -> None:
    """Plan trajectories and create the trajectory HDF5 file."""
    from one_scan_is_enough.planning.pipeline import plan

    OmegaConf.resolve(config)
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Missing configuration: {sorted(missing)}")
    plan(config, Path(to_absolute_path(config.input.scene_dir)))


if __name__ == "__main__":
    main()
