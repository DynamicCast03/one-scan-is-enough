"""Label rendered trajectories through concurrent model calls and one HDF5 writer."""

import fcntl
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from one_scan_is_enough.instruction.backend import InstructionBackend
from one_scan_is_enough.instruction.prompts import INSTRUCTION_TYPES
from one_scan_is_enough.scene import Scene
from one_scan_is_enough.trajectories import TrajectoryFile

logger = logging.getLogger(__name__)


def _annotate(
    client: InstructionBackend, instruction_type: str, route: list[bytes], goal: list[bytes]
) -> str:
    """Generate a navigation instruction from route frames and goal views in one model call."""
    kind = INSTRUCTION_TYPES[instruction_type]
    if kind.route_frames == "last":
        route = route[-1:]
    return client.complete(kind.prompt, route, goal)


def assign_types(keys: list[str], shares: dict[str, float], seed: int) -> dict[str, str]:
    """Give each trajectory one instruction type, with counts proportional to the shares.

    Counts round by largest remainder and the types are shuffled over the
    sorted keys, so the same keys, shares and seed give the same assignment.
    """
    unknown = set(shares) - set(INSTRUCTION_TYPES)
    if unknown or any(share < 0 for share in shares.values()) or sum(shares.values()) <= 0:
        raise ValueError(f"Invalid instruction type shares: {dict(shares)}")
    names = [name for name, share in shares.items() if share > 0]
    exact = np.array([shares[name] for name in names], dtype=np.float64)
    exact *= len(keys) / exact.sum()
    counts = np.floor(exact).astype(int)
    remainders = np.argsort(counts - exact, kind="stable")
    counts[remainders[: len(keys) - counts.sum()]] += 1
    types = np.repeat(names, counts)
    np.random.default_rng(seed).shuffle(types)
    return dict(zip(sorted(keys), types.tolist()))


def generate(config: DictConfig, scene_dir: Path) -> Path:
    """Call the model for unlabeled trajectories and write each complete instruction.

    Each trajectory is labeled with its type of assign_types. Trajectories whose
    instruction has another type, or none, are labeled again. Only the main
    thread opens trajectories.h5. It keeps up to the configured number of
    requests in flight and starts the next one as soon as any finishes.
    """
    scene = Scene(scene_dir)
    concurrency = config.llm.concurrency
    if concurrency <= 0:
        raise ValueError("Concurrency must be positive.")
    client = InstructionBackend(config.llm)
    with client, (scene_dir / ".trajectories.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        errors = []
        with TrajectoryFile(scene.trajectories_path, "r+") as data:
            if data.scene_name != scene.read_name():
                raise ValueError("Scene and trajectories have different names.")
            data.require("rendering")
            types = assign_types(data.keys(), config.instruction.types, config.seed)
            labels = {key: data.instruction(key) for key in data.keys()}
            pending = [
                key
                for key, label in labels.items()
                if config.instruction.relabel or label is None or label.type != types[key]
            ]
            logger.info(
                "Instruction types: %s; %d of %d trajectories to label",
                {name: list(types.values()).count(name) for name in config.instruction.types},
                len(pending),
                len(types),
            )
            names = iter(pending)
            running = {}
            with (
                ThreadPoolExecutor(max_workers=concurrency) as pool,
                tqdm(total=len(pending), desc="Label trajectories") as progress,
            ):
                while True:
                    # Refill free request slots so that one slow request does not idle the others.
                    for name in islice(names, concurrency - len(running)):
                        route, goal = data.instruction_images(name)
                        future = pool.submit(_annotate, client, types[name], route, goal)
                        running[future] = name
                    if not running:
                        break
                    done, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in done:
                        name = running.pop(future)
                        progress.update()
                        try:
                            instruction = future.result()
                        except Exception:
                            logger.exception(
                                "Instruction generation failed for trajectory %s", name
                            )
                            errors.append(name)
                            continue
                        data.write_instruction(name, instruction, types[name])
                        logger.info("Instruction ready: %s", name)
        if errors:
            raise RuntimeError(
                f"Instruction generation failed for {len(errors)} trajectories: {errors[:8]}"
            )
    return scene.trajectories_path
