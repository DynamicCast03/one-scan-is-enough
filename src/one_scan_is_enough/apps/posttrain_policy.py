"""Hydra entry point for PRPT-KS post-training; launch one process per GPU."""

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="apps/posttrain_policy")
def main(config: DictConfig) -> None:
    """Post-train the policy, for example with torchrun -m one_scan_is_enough.apps.posttrain_policy."""
    from one_scan_is_enough.posttrain.trainer import train

    OmegaConf.resolve(config)
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Missing configuration: {sorted(missing)}")
    for key in ("scene_dir", "base", "output_dir"):
        config[key] = to_absolute_path(config[key])
    train(config)


if __name__ == "__main__":
    main()
