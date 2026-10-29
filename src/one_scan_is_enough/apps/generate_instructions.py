"""Hydra entry point for labeling rendered trajectories through a model."""

from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="apps/generate_instructions")
def main(config: DictConfig) -> None:
    """Generate instructions without loading or running a GPU renderer."""
    from one_scan_is_enough.instruction.pipeline import generate

    OmegaConf.resolve(config)
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Missing configuration: {sorted(missing)}")
    generate(config, Path(to_absolute_path(config.input.scene_dir)))


if __name__ == "__main__":
    main()
