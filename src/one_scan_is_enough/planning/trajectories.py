"""Sample reference trajectories on the planning map without file I/O.

Starts, goals and paths lie in the free cells of the planning map, the
feasible region shared with closed-loop runs. Pairs are sampled on distance
rings within one connected component and planned in both directions.
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from omegaconf import DictConfig
from scipy.spatial import cKDTree
from tqdm import tqdm

from one_scan_is_enough.planning import path_planner
from one_scan_is_enough.planning.path_sampling import sample_poses
from one_scan_is_enough.scene import BevMap

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """A planned reference path and the scene poses sampled along it."""

    path_xz: np.ndarray  # (N, 2), meters in the aligned frame.
    poses: np.ndarray  # (M, 3) scene poses of one_scan_is_enough.poses.


def sample_goal_on_distance_ring(
    passable_coords: np.ndarray,
    start_pixel: list[int],
    resolution: float,
    min_dist_m: float,
    band_m: float,
    rng: np.random.Generator,
    max_tries: int,
    max_distance_m: float,
) -> list[int] | None:
    """Sample a goal from passable pixels near a random radius around start.

    The sampled radius is uniform in [min_dist_m, min(max_dist_m(start), max_distance_m)],
    where max_dist_m(start) is the farthest Euclidean distance (in meters) from start to any passable pixel.
    Candidates are passable pixels with Euclidean distance within +/- band_m of the sampled radius.
    """
    if len(passable_coords) == 0:
        return None
    start_rc = np.asarray([int(start_pixel[0]), int(start_pixel[1])], dtype=np.float64)
    coords = passable_coords.astype(np.float64)
    d_px = np.hypot(coords[:, 0] - start_rc[0], coords[:, 1] - start_rc[1])
    max_dist_m = min(float(d_px.max()) * float(resolution), max_distance_m)
    if max_dist_m <= float(min_dist_m):
        return None
    band_px = max(1.0, float(band_m) / max(1e-12, float(resolution)))
    min_px = float(min_dist_m) / max(1e-12, float(resolution))
    max_px = float(max_dist_m) / max(1e-12, float(resolution))
    for _ in range(max(1, int(max_tries))):
        target_px = float(rng.uniform(min_px, max_px))
        idx = np.flatnonzero(np.abs(d_px - target_px) <= band_px)
        if idx.size == 0:
            continue
        pick = passable_coords[int(idx[int(rng.integers(0, idx.size))])]
        return [int(pick[0]), int(pick[1])]
    return None


def snap_goal_pixel(
    goal_pixel: list[int],
    distance_field: np.ndarray,
    goal_near_obstacle_px: int,
    goal_snap_coords: np.ndarray,
    goal_snap_tree: cKDTree,
) -> list[int]:
    """Keep a goal near obstacles, else move it to the nearest goal cell near obstacles."""
    (row, col) = (int(goal_pixel[0]), int(goal_pixel[1]))
    if distance_field[row, col] <= float(goal_near_obstacle_px):
        return [row, col]
    (_, idx) = goal_snap_tree.query(np.asarray([row, col], dtype=np.float64), k=1)
    snapped = goal_snap_coords[int(idx)]
    return [int(snapped[0]), int(snapped[1])]


def build_direction(
    start_pixel: list[int],
    ring_goal_pixel: list[int],
    distance_field: np.ndarray,
    goal_near_obstacle_px: int,
    goal_snap_coords: np.ndarray,
    goal_snap_tree: cKDTree,
    safe_component_labels: np.ndarray,
    drop_clear_path_prob: float,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    goal_pixel = snap_goal_pixel(
        ring_goal_pixel, distance_field, goal_near_obstacle_px, goal_snap_coords, goal_snap_tree
    )
    start_label = int(safe_component_labels[int(start_pixel[0]), int(start_pixel[1])])
    goal_label = int(safe_component_labels[int(goal_pixel[0]), int(goal_pixel[1])])
    if start_label == 0 or goal_label == 0 or start_label != goal_label:
        return None
    if path_planner.line_clearance_safe(start_pixel, goal_pixel, distance_field, 1) and (
        float(rng.random()) < float(drop_clear_path_prob)
    ):
        return None
    return {
        "start_pixel": [int(start_pixel[0]), int(start_pixel[1])],
        "goal_pixel": [int(goal_pixel[0]), int(goal_pixel[1])],
        "ring_goal_pixel": [int(ring_goal_pixel[0]), int(ring_goal_pixel[1])],
    }


def select_diverse_pair_tasks(
    candidates: list[dict[str, Any]],
    target_n: int,
    completed: list[dict[str, Any]],
    pair_weight: float,
) -> list[dict[str, Any]]:
    if target_n <= 0 or len(candidates) == 0:
        return []
    n = len(candidates)
    take_n = min(int(target_n), n)
    ids = np.asarray([int(c["candidate_id"]) for c in candidates], dtype=np.int64)
    starts = np.asarray(
        [[float(c["start_pixel"][0]), float(c["start_pixel"][1])] for c in candidates],
        dtype=np.float64,
    )
    goals = np.asarray(
        [
            [
                float(c["ring_goal_pixel"][0]),
                float(c["ring_goal_pixel"][1]),
            ]
            for c in candidates
        ],
        dtype=np.float64,
    )
    d = np.hypot(starts[:, 0] - goals[:, 0], starts[:, 1] - goals[:, 1])
    (q1, q2, q3) = np.quantile(d, [0.25, 0.5, 0.75]).tolist()
    bucket_ids = np.searchsorted(np.asarray([q1, q2, q3], dtype=np.float64), d, side="left")
    bucket_masks = [bucket_ids == i for i in range(4)]
    base = int(take_n // 4)
    quotas = [base + (1 if i < int(take_n % 4) else 0) for i in range(4)]
    active = np.ones(n, dtype=bool)
    chosen_idx = []
    has_refs = len(completed) > 0
    ds = np.full(n, np.inf, dtype=np.float64)
    dg = np.full(n, np.inf, dtype=np.float64)
    dp = np.full(n, np.inf, dtype=np.float64)
    if has_refs:
        for c in tqdm(
            completed, total=len(completed), desc="Init diversity cache", unit="ref", leave=False
        ):
            s = np.hypot(
                starts[:, 0] - float(c["start_pixel"][0]), starts[:, 1] - float(c["start_pixel"][1])
            )
            g = np.hypot(
                goals[:, 0] - float(c["ring_goal_pixel"][0]),
                goals[:, 1] - float(c["ring_goal_pixel"][1]),
            )
            ds = np.minimum(ds, s)
            dg = np.minimum(dg, g)
            dp = np.minimum(dp, 0.5 * (s + g))
        score = ds + dg + pair_weight * dp
    else:
        score = d.copy()
    all_mask = np.ones(n, dtype=bool)

    def pick_best(mask: np.ndarray) -> int:
        idx = np.flatnonzero(active & mask)
        if idx.size == 0:
            return -1
        sc = score[idx]
        best = float(np.max(sc))
        top = idx[sc == best]
        if top.size == 1:
            return int(top[0])
        return int(top[int(np.argmin(ids[top]))])

    def push_ref(i: int) -> None:
        nonlocal has_refs, ds, dg, dp, score
        active[i] = False
        chosen_idx.append(int(i))
        s = np.hypot(starts[:, 0] - starts[i, 0], starts[:, 1] - starts[i, 1])
        g = np.hypot(goals[:, 0] - goals[i, 0], goals[:, 1] - goals[i, 1])
        if has_refs:
            ds = np.minimum(ds, s)
            dg = np.minimum(dg, g)
            dp = np.minimum(dp, 0.5 * (s + g))
        else:
            (ds, dg, dp, has_refs) = (s, g, 0.5 * (s + g), True)
        score = ds + dg + pair_weight * dp
        score[~active] = -np.inf

    with tqdm(total=take_n, desc="Select diverse pair tasks", unit="task", leave=False) as pbar:
        for i in range(4):
            while int(quotas[i]) > 0:
                idx = pick_best(bucket_masks[i])
                if idx < 0:
                    break
                push_ref(idx)
                quotas[i] -= 1
                pbar.update(1)
        while len(chosen_idx) < take_n:
            idx = pick_best(all_mask)
            if idx < 0:
                break
            push_ref(idx)
            pbar.update(1)
    return [candidates[i] for i in chosen_idx]


def plan_direction(
    bev: BevMap,
    cache: path_planner.PlannerCache,
    config: DictConfig,
    start_pixel: list[int],
    goal_pixel: list[int],
) -> np.ndarray | None:
    """Plan one direction; reject unreachable pairs and paths below the minimum length."""
    path = path_planner.astar(cache, tuple(start_pixel), tuple(goal_pixel))
    if path is None or len(path) < 2:
        return None
    path = path_planner.simplify(cache, path)
    world = [
        [c * bev.resolution + bev.origin_xz[0], r * bev.resolution + bev.origin_xz[1]]
        for r, c in path
    ]
    length = sum(
        math.hypot(world[i][0] - world[i - 1][0], world[i][1] - world[i - 1][1])
        for i in range(1, len(world))
    )
    if length < config.ring_min_ref_geodesic_m:
        return None
    return np.asarray(world)


def eval_task(
    task: dict, bev: BevMap, cache: path_planner.PlannerCache, config: DictConfig
) -> dict | None:
    """Plan all directions of one candidate; reject incomplete pairs."""
    results = []
    for candidate in task["directions"]:
        path = plan_direction(bev, cache, config, candidate["start_pixel"], candidate["goal_pixel"])
        if path is None:
            return None
        results.append(path)
    return {**task, "paths": results}


def plan_trajectories(
    bev: BevMap,
    config: DictConfig,
    planner_config: DictConfig,
    sampling_config: DictConfig,
    rng: np.random.Generator,
    workers: int,
) -> list[Trajectory]:
    """Sample ring pairs in both directions on a planning map; return paths and poses.

    Starts are free cells of the planning map, and goals are free cells of the
    start's connected component. Goals farther than goal_near_obstacle_max_m
    from obstacles move to the nearest such cell.
    """
    if config.num_trajectories <= 0 or config.num_trajectories % 2:
        raise ValueError("Trajectory count must be positive and even for bidirectional pairs.")
    cache = path_planner.planner_cache(bev, planner_config)
    passable_mask = cache.free_mask
    passable_coords = np.argwhere(passable_mask)
    if len(passable_coords) < 2:
        raise ValueError("Not enough free cells on the planning map")
    _, safe_component_labels = cv2.connectedComponents(
        passable_mask.astype(np.uint8), connectivity=8
    )
    component_coords = {
        label: passable_coords[safe_component_labels[tuple(passable_coords.T)] == label]
        for label in np.unique(safe_component_labels[tuple(passable_coords.T)])
    }
    distance_field = cache.distance_field
    goal_near_obstacle_px = max(
        1, int(np.ceil(config.goal_near_obstacle_max_m / float(bev.resolution)))
    )
    goal_snap_coords = np.argwhere(
        passable_mask & (distance_field <= goal_near_obstacle_px)
    ).astype(np.int32)
    if len(goal_snap_coords) == 0:
        raise ValueError("No passable goal pixels near obstacles")
    goal_snap_tree = cKDTree(goal_snap_coords.astype(np.float64))
    requested_units = config.num_trajectories // 2
    max_pair_n = max(
        int(requested_units), int(math.ceil(float(requested_units) * config.candidate_multiplier))
    )
    pair_tasks: list[dict[str, Any]] = []
    next_candidate_id = 0

    def make_direction(start, goal):
        return build_direction(
            start,
            goal,
            distance_field,
            goal_near_obstacle_px,
            goal_snap_coords,
            goal_snap_tree,
            safe_component_labels,
            config.drop_clear_path_prob,
            rng,
        )

    def append_pair_candidates(target_total: int) -> None:
        nonlocal next_candidate_id
        attempts = 0
        keep_ratio = max(0.001, 1.0 - config.drop_clear_path_prob)
        attempt_limit = max(
            target_total - len(pair_tasks),
            math.ceil(
                (target_total - len(pair_tasks)) * config.ring_attempt_multiplier / keep_ratio
            ),
        )
        while len(pair_tasks) < target_total:
            if attempts >= attempt_limit:
                break
            attempts += 1
            start = passable_coords[int(rng.integers(0, len(passable_coords)))].tolist()
            goal = sample_goal_on_distance_ring(
                component_coords[safe_component_labels[start[0], start[1]]],
                start,
                bev.resolution,
                config.ring_min_start_goal_dist_m,
                config.ring_band_m,
                rng,
                config.ring_max_tries,
                config.ring_max_start_goal_dist_m,
            )
            if goal is None:
                continue
            directions = [make_direction(start, goal), make_direction(goal, start)]
            if any(direction is None for direction in directions):
                continue
            pair_tasks.append(
                {
                    "candidate_id": next_candidate_id,
                    "start_pixel": start,
                    "ring_goal_pixel": goal,
                    "directions": directions,
                }
            )
            next_candidate_id += 1

    append_pair_candidates(int(max_pair_n))
    logger.info(
        "Sampling done: goal_snap_coords=%d pair_tasks=%d",
        len(goal_snap_coords),
        len(pair_tasks),
    )
    completed = []
    remaining_tasks = pair_tasks
    batch_round = 0
    requested_n = int(requested_units)
    logger.info(
        "Planning started: workers=%d candidate_tasks=%d requested=%d",
        int(workers),
        len(pair_tasks),
        requested_n,
    )
    # Threads share the planner cache; the compiled A* releases the GIL.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        while remaining_tasks and len(completed) < requested_n:
            need = requested_n - len(completed)
            batch_tasks = select_diverse_pair_tasks(
                remaining_tasks,
                target_n=need,
                completed=completed,
                pair_weight=config.diversity_pair_weight,
            )
            if len(batch_tasks) == 0:
                break
            picked = {int(t["candidate_id"]) for t in batch_tasks}
            remaining_tasks = [t for t in remaining_tasks if int(t["candidate_id"]) not in picked]
            logger.info(
                "Planning batch started: round=%d need=%d picked=%d remaining_after_pick=%d",
                int(batch_round),
                int(need),
                len(batch_tasks),
                len(remaining_tasks),
            )
            batch_completed = []
            futs = [ex.submit(eval_task, t, bev, cache, config) for t in batch_tasks]
            for f in tqdm(
                as_completed(futs),
                total=len(futs),
                desc=f"Planning batch {batch_round}",
                unit="task",
            ):
                r = f.result()
                if r is not None:
                    batch_completed.append(r)
            batch_completed.sort(key=lambda task: task["candidate_id"])
            completed.extend(batch_completed)
            logger.info(
                "Planning batch finished: round=%d reachable_batch=%d reachable_total=%d/%d",
                int(batch_round),
                len(batch_completed),
                len(completed),
                int(requested_n),
            )
            if len(completed) < requested_n and len(remaining_tasks) == 0:
                before = len(pair_tasks)
                refill_target = before + max(
                    int(requested_n - len(completed)),
                    int(
                        math.ceil(float(requested_n - len(completed)) * config.candidate_multiplier)
                    ),
                )
                append_pair_candidates(refill_target)
                if len(pair_tasks) == before:
                    break
                remaining_tasks = pair_tasks[before:]
                logger.info(
                    "Planning refill appended: round=%d added=%d reachable_total=%d/%d",
                    int(batch_round),
                    int(len(pair_tasks) - before),
                    len(completed),
                    int(requested_n),
                )
            batch_round += 1
    completed.sort(key=lambda x: int(x["candidate_id"]))
    selected_tasks = completed[:requested_n]
    selected = [path for task in selected_tasks for path in task["paths"]]
    logger.info(
        "Planning finished: selected=%d requested=%d attempted=%d total_candidates=%d",
        len(selected),
        config.num_trajectories,
        int(len(pair_tasks) - len(remaining_tasks)),
        len(pair_tasks),
    )
    if len(selected) < config.num_trajectories:
        raise RuntimeError(
            f"Not enough planned paths: selected={len(selected)} requested={config.num_trajectories} attempted={int(len(pair_tasks) - len(remaining_tasks))}"
        )
    with ThreadPoolExecutor(max_workers=workers) as ex:
        poses = list(
            tqdm(
                ex.map(lambda path: sample_poses(path, sampling_config), selected),
                total=len(selected),
                desc="Sample poses",
                unit="trajectory",
            )
        )
    return [Trajectory(path, pose) for path, pose in zip(selected, poses)]
