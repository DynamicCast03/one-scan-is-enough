"""Hydra entry point for training a PGSR Gaussian scene from export-pgsr outputs."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# Python modules of the CUDA extensions that PGSR imports.
EXTENSIONS = "import diff_plane_rasterization, simple_knn._C"


def true_flags(root: Path) -> set[str]:
    """Names of the PGSR boolean parameters that default to true and cannot be switched off."""
    tree = ast.parse((root / "arguments" / "__init__.py").read_text())
    return {
        node.targets[0].attr.lstrip("_")
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
    }


def train_arguments(train: dict, fixed: set[str]) -> list[str]:
    """Translate the train settings into PGSR command line arguments of the same names."""
    arguments = []
    for key, value in train.items():
        if isinstance(value, bool):
            if value:
                arguments.append(f"--{key}")
            elif key in fixed:
                raise ValueError(f"PGSR cannot switch off {key}, which defaults to true")
        elif isinstance(value, list):
            arguments += [f"--{key}", *map(str, value)]
        elif value is not None:
            arguments += [f"--{key}", str(value)]
    return arguments


def run(command: list[str], cwd: Path, env: dict, log_path: Path) -> int:
    """Run a command, copying its output to the terminal and a log file."""
    with open(log_path, "ab") as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        while chunk := process.stdout.read1(65536):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk)
        return process.wait()


@hydra.main(version_base="1.3", config_path="../config", config_name="apps/train_pgsr")
def main(config: DictConfig) -> None:
    """Train PGSR in its own interpreter and report the final Gaussian point cloud."""
    OmegaConf.resolve(config)
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Missing configuration: {sorted(missing)}")
    root = (
        Path(to_absolute_path(config.pgsr.root))
        if config.pgsr.root
        else (Path(__file__).resolve().parents[3] / "third_party" / "PGSR")
    )
    if not (root / "train.py").is_file():
        raise FileNotFoundError(f"PGSR checkout not found at {root}")
    python = config.pgsr.python or sys.executable
    env = dict(os.environ)
    if config.pgsr.pythonpath:
        paths = [to_absolute_path(config.pgsr.pythonpath), env.get("PYTHONPATH", "")]
        env["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
    check = subprocess.run(
        [python, "-c", EXTENSIONS], cwd=root, env=env, capture_output=True, text=True
    )
    if check.returncode != 0:
        raise RuntimeError(f"PGSR CUDA extensions are not importable by {python}:\n{check.stderr}")

    train = OmegaConf.to_container(config.train)
    model = Path(to_absolute_path(config.output.model_dir))
    command = [
        python,
        "-u",
        "train.py",
        "-s",
        to_absolute_path(config.input.scene_dir),
        "-m",
        str(model),
        *train_arguments(train, true_flags(root)),
    ]
    log_path = Path(HydraConfig.get().runtime.output_dir) / "pgsr.log"
    print(" ".join(command), flush=True)
    code = run(command, root, env, log_path)
    if code != 0:
        raise RuntimeError(f"PGSR training exited with code {code}, see {log_path}")
    ply = model / "point_cloud" / f"iteration_{train['iterations']}" / "point_cloud.ply"
    if not ply.is_file():
        raise FileNotFoundError(f"PGSR finished without writing {ply}")
    print(f"Gaussian point cloud: {ply}", flush=True)


if __name__ == "__main__":
    main()
