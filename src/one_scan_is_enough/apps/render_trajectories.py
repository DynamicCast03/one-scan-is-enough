"""Hydra entry point for parallel GPU rendering of trajectory observations."""

from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="apps/render_trajectories")
def main(config: DictConfig) -> None:
    """Render training frames, route frames and goal views without model requests."""
    from one_scan_is_enough.rendering.pipeline import render

    OmegaConf.resolve(config)
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Missing configuration: {sorted(missing)}")
    render(config, Path(to_absolute_path(config.input.scene_dir)))


if __name__ == "__main__":
    main()
