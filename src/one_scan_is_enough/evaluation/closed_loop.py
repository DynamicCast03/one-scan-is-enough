"""Closed-loop evaluation of RealNav in a Gaussian scene.

Each episode starts at the first reference pose and runs as a Navigation of
one_scan_is_enough.simulation.execution: at every decision the policy predicts eight
waypoints from its observation history and the run executes some of them.
An episode succeeds when the final position lies within the success radius
of the goal.

Launch processes with torchrun; several can share one GPU. Ranks evaluate
disjoint episodes and append one JSON line per episode to their own file.
Episodes already recorded in the output directory are skipped, so an
interrupted run resumes. Rank 0 writes the summary once all ranks finish.
"""

import json
import logging
import math
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tqdm import tqdm

from one_scan_is_enough.policy.predictor import NavigationPolicy
from one_scan_is_enough.scene import Scene
from one_scan_is_enough.simulation.environment import SceneEnvironment
from one_scan_is_enough.trajectories import Episode, TrajectoryFile

logger = logging.getLogger(__name__)


def evaluate(config: DictConfig) -> dict | None:
    """Evaluate a checkpoint and return the summary on rank 0."""
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("gloo", timeout=timedelta(days=1))
    # Ranks share GPUs when there are more ranks than GPUs on a node.
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)) % torch.cuda.device_count())
    torch.cuda.set_device(device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with TrajectoryFile(config.trajectories) as file:
        episodes = file.episodes()
    done = {record["key"] for record in _records(output)}
    pending = [e for e in episodes[rank::world] if e.key not in done]
    logger.info("Rank %d: %d of %d episodes pending", rank, len(pending), len(episodes))
    if pending:
        environment = SceneEnvironment(Scene(config.scene_dir), config.environment, device)
        policy = NavigationPolicy.from_pretrained(config.checkpoint, device=str(device))
        with open(output / f"episodes-{rank:02d}.jsonl", "a", encoding="utf-8") as file:
            for episode in tqdm(pending, desc=f"Rank {rank}", unit="episode"):
                record = run_episode(policy, environment, episode, config.success_radius_m)
                file.write(json.dumps(record) + "\n")
                file.flush()

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        return None
    records = {record["key"]: record for record in _records(output)}
    summary = summarize([records[e.key] for e in episodes if e.key in records])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Summary: %s", summary)
    return summary


def run_episode(
    policy: NavigationPolicy,
    environment: SceneEnvironment,
    episode: Episode,
    success_radius_m: float,
) -> dict:
    """Run one episode and return its JSON record."""
    goal = episode.poses[-1, :2]
    session = policy.start(episode.instruction)
    run = environment.start(episode.poses[0], session.observe)
    decisions = 0
    while not run.done:
        run.step(session.predict().astype(np.float64))
        decisions += 1

    trajectory = np.array(run.trajectory)  # (K, 3) scene poses from the start.
    errors = np.linalg.norm(trajectory[:, :2] - goal, axis=1)
    success = bool(errors[-1] < success_radius_m)
    shortest = episode.path_length
    return {
        "key": episode.key,
        "instruction": episode.instruction,
        "instruction_type": episode.instruction_type,
        "success": success,
        "spl": success * shortest / max(shortest, run.path_length, 1e-9),
        "navigation_error_m": float(errors[-1]),
        # The start pose does not count toward oracle success.
        "oracle_success": bool((errors[1:] < success_radius_m).any()),
        "path_length_m": run.path_length,
        "shortest_path_m": shortest,
        "stopped": run.stopped,
        "decisions": decisions,
        "executed_waypoints": run.executed,
        "trajectory": trajectory.tolist(),
    }


def summarize(records: list[dict]) -> dict:
    """Average SR, SPL, NE and OS over episode records, overall and per instruction type."""
    summary = _averages(records)
    types = sorted({record["instruction_type"] for record in records})
    summary["by_type"] = {
        name: _averages([r for r in records if r["instruction_type"] == name]) for name in types
    }
    return summary


def _averages(records: list[dict]) -> dict:
    return {
        "episodes": len(records),
        "success_rate": _mean(records, "success"),
        "spl": _mean(records, "spl"),
        "navigation_error_m": _mean(records, "navigation_error_m"),
        "oracle_success_rate": _mean(records, "oracle_success"),
    }


def _mean(records: list[dict], key: str) -> float:
    return float(np.mean([float(record[key]) for record in records])) if records else math.nan


def _records(output: Path) -> list[dict]:
    records = []
    for path in sorted(output.glob("episodes-*.jsonl")):
        with open(path, encoding="utf-8") as file:
            records.extend(json.loads(line) for line in file if line.strip())
    return records
