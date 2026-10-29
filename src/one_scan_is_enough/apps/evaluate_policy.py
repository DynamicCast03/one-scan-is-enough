"""Hydra entry point for closed-loop evaluation; launch one process per GPU."""

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="apps/evaluate_policy")
def main(config: DictConfig) -> None:
    """Evaluate the policy, for example with torchrun -m one_scan_is_enough.apps.evaluate_policy."""
    from one_scan_is_enough.evaluation.closed_loop import evaluate

    OmegaConf.resolve(config)
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Missing configuration: {sorted(missing)}")
    for key in ("scene_dir", "trajectories", "checkpoint", "output_dir"):
        config[key] = to_absolute_path(config[key])
    evaluate(config)


if __name__ == "__main__":
    main()
