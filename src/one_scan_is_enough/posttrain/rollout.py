"""PRPT-KS rollouts: Pareto-admissible rollouts, preference pairs and advantages.

At each decision the policy proposes eight waypoints. Their Pareto utility has
three axes read off the scene: progress along the reference path, safety
(staying on the planning map) and a correct stop at the goal. The
admissibility test rejects a policy action that leaves the planning map or
loses more progress than a tolerance; once the agent is near the goal, the
planner acts as well. A planner action then replaces the policy action, and
the pair becomes a preference pair weighted by Pareto dominance. Executed
actions become PPO samples whose advantages come from GAE over the progress
potential.
"""

import math
import random
from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from one_scan_is_enough.policy.predictor import NavigationPolicy
from one_scan_is_enough.simulation.environment import SceneEnvironment
from one_scan_is_enough.simulation.execution import Move, Navigation
from one_scan_is_enough.trajectories import Episode

# Progress potential R_s * f(completion) * exp(-deviation), with R_s a fraction
# of the success reward, f affine in completion and a deviation dead zone.
POTENTIAL_SCALE = 0.9
COMPLETION_OFFSET = 0.1
COMPLETION_SLOPE = 0.9
DEVIATION_DEAD_ZONE_M = 0.8


class EpisodeSampler:
    """Draw episodes without replacement, favoring long ones as training progresses.

    An episode of normalized length l in [0, 1] has the weight
    exp(strength * (2 * progress - 1) * l). Each restart redraws the order.
    """

    def __init__(self, episodes: list[Episode], strength: float, seed: int):
        self.episodes = episodes
        self.strength = strength
        self.rng = random.Random(seed)
        lengths = np.array([episode.length for episode in episodes])
        span = lengths.max() - lengths.min()
        self.scores = (lengths - lengths.min()) / span if span > 0 else np.zeros(len(lengths))
        self.restart(0.0)

    def restart(self, progress: float) -> None:
        """Redraw the order for a training progress in [0, 1]."""
        self.progress = progress
        indices = list(range(len(self.episodes)))
        weights = [math.exp(self.strength * (2 * progress - 1) * s) for s in self.scores]
        self.order = []
        while indices:
            index = self.rng.choices(range(len(indices)), weights=weights)[0]
            self.order.append(indices.pop(index))
            weights.pop(index)

    def next(self) -> Episode:
        """Return the next episode, restarting at the same progress when exhausted."""
        if not self.order:
            self.restart(self.progress)
        return self.episodes[self.order.pop(0)]


@dataclass
class Decision:
    """One training sample from a rollout.

    PPO samples hold the executed action and the action whose likelihood
    anchors the ratio. Preference samples hold the planner action as action
    and the rejected policy action as reference. Waypoints are metric.
    """

    episode: int  # Index into the rollout frames.
    frames: int  # Observations available at the decision.
    instruction: str
    action: np.ndarray  # (8, 3)
    reference: np.ndarray  # (8, 3)
    preference: bool
    reward: float = 0.0
    advantage: float = 0.0
    reference_logit: float = 0.0
    weight: float = 1.0
    progress: float = 0.0
    planner: bool = False


@dataclass
class Rollout:
    """Decisions and observation histories of the episodes in one buffer."""

    decisions: list[Decision]
    frames: list[list[np.ndarray]]
    successes: list[bool]


@dataclass
class _Utility:
    """The Pareto utility of an action: progress, safety and stop axes."""

    value_delta: float
    collision: float
    goal_stop: float
    value_after: float

    def dominates(self, other: "_Utility") -> bool:
        mine = (self.value_delta, self.collision, self.goal_stop)
        theirs = (other.value_delta, other.collision, other.goal_stop)
        return all(a >= b for a, b in zip(mine, theirs)) and any(
            a > b for a, b in zip(mine, theirs)
        )


class RolloutWorker:
    """Run the policy in one scene and turn the episodes into training decisions."""

    def __init__(
        self,
        policy: NavigationPolicy,
        environment: SceneEnvironment,
        sampler: EpisodeSampler,
        config: DictConfig,
    ):
        self.policy = policy
        self.environment = environment
        self.sampler = sampler
        self.config = config
        model_config = policy.model.config
        self.scale = np.asarray(model_config.waypoint_scale)

    def collect(self, steps: int, sigma: float) -> Rollout:
        """Roll out whole episodes until the buffer holds at least steps decisions.

        Advantages are divided by the largest magnitude in the buffer.
        sigma is the likelihood scale of the preference reference logits.
        """
        rollout = Rollout([], [], [])
        bar = tqdm(total=steps, desc="Rollout", unit="decision")
        while len(rollout.decisions) < steps:
            decisions, frames, success = self._episode(
                len(rollout.frames), self.sampler.next(), sigma
            )
            rollout.decisions.extend(decisions)
            rollout.frames.append(frames)
            rollout.successes.append(success)
            bar.update(min(len(decisions), steps - bar.n))
        bar.close()
        ppo = [d for d in rollout.decisions if not d.preference]
        if ppo:
            peak = max(abs(d.advantage) for d in ppo) + 1e-8
            for decision in ppo:
                decision.advantage /= peak
        return rollout

    def _episode(
        self, index: int, episode: Episode, sigma: float
    ) -> tuple[list[Decision], list[np.ndarray], bool]:
        config = self.config
        radius = config.reward.success_radius_m
        reference = episode.poses[:, :2]
        goal = reference[-1]
        frames = []
        session = self.policy.start(episode.instruction)

        def observe(frame: np.ndarray) -> None:
            frames.append(frame)
            session.observe(frame)

        run = self.environment.start(episode.poses[0], observe)
        decisions, values, next_values = [], [], []
        takeover = success = False
        while not run.done:
            pose = run.pose
            value_before = self._value(*self._progress(pose[:2], reference))
            takeover = takeover or np.linalg.norm(pose[:2] - goal) < radius
            policy_action = session.predict().astype(np.float64)
            policy_utility = self._utility(run, policy_action, value_before, reference)
            if takeover:
                planner = True
                action = self.environment.plan(pose, goal)
            else:
                # Safety and progress are tested separately, so neither offsets the other.
                progress = config.reward.gamma * policy_utility.value_after - value_before
                planner = (
                    policy_utility.collision < 0 or progress < -config.takeover.progress_tolerance
                )
                if planner:
                    action = self.environment.plan_along(
                        pose, reference, config.takeover.lookahead, config.takeover.increment
                    )
            if planner:
                planner_utility = self._utility(run, action, value_before, reference)
                decisions.append(
                    Decision(
                        index,
                        len(frames),
                        episode.instruction,
                        action,
                        policy_action,
                        preference=True,
                        reference_logit=self._log_likelihood(action, policy_action, sigma),
                        weight=self._preference_weight(planner_utility, policy_utility),
                        planner=True,
                    )
                )
            else:
                action = policy_action
            decision = Decision(
                index, len(frames), episode.instruction, action, action, preference=False
            )
            move = run.step(action)
            completion, distance = self._progress(_end(move, pose), reference)
            decision.progress, decision.planner = completion, planner
            reached = move.stopped and np.linalg.norm(run.pose[:2] - goal) < radius
            decision.reward = config.reward.success * reached - self._penalty(move.poses)
            values.append(value_before)
            next_values.append(self._value(completion, distance))
            decisions.append(decision)
            success = success or reached
        self._advantages([d for d in decisions if not d.preference], values, next_values)
        return decisions, frames, success

    def _utility(
        self,
        run: Navigation,
        action: np.ndarray,
        value_before: float,
        reference: np.ndarray,
    ) -> _Utility:
        """Score the executed part of an action without moving the agent."""
        move = run.preview(action)
        after = _end(move, run.pose)
        value_after = self._value(*self._progress(after, reference))
        reward = self.config.reward
        reached = move.stopped and np.linalg.norm(after - reference[-1]) < reward.success_radius_m
        return _Utility(
            value_delta=value_after - value_before,
            collision=-self._penalty(move.poses),
            goal_stop=reward.success if reached else 0.0,
            value_after=value_after,
        )

    def _penalty(self, points: np.ndarray) -> float:
        """Return the obstacle penalty when any point leaves the planning map."""
        outside = any(self.environment.outside(point[:2]) for point in points)
        return self.config.reward.obstacle_penalty if outside else 0.0

    @staticmethod
    def _progress(position: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
        """Return the completion and distance at the nearest reference position."""
        distances = np.linalg.norm(reference - position, axis=1)
        nearest = int(np.argmin(distances))
        completion = nearest / (len(reference) - 1) if len(reference) > 1 else 1.0
        return completion, float(distances[nearest])

    def _value(self, completion: float, distance: float) -> float:
        """Progress potential of a state; deviations within the dead zone cost nothing."""
        distance = 0.0 if distance < DEVIATION_DEAD_ZONE_M else distance
        completion = min(max(completion, 0.0), 1.0)
        return (
            POTENTIAL_SCALE
            * self.config.reward.success
            * (COMPLETION_OFFSET + COMPLETION_SLOPE * completion)
            * math.exp(-distance)
        )

    def _log_likelihood(self, chosen: np.ndarray, rejected: np.ndarray, sigma: float) -> float:
        """Gaussian log likelihood of chosen under rejected in normalized waypoint space."""
        difference = (chosen - rejected) * self.scale
        return float(-np.square(difference).sum() / (2.0 * sigma * sigma))

    def _preference_weight(self, planner: _Utility, policy: _Utility) -> float:
        weights = self.config.preference_weights
        if planner.dominates(policy):
            return weights.goal_stop if planner.goal_stop > 0 else weights.dominant
        return weights.other

    def _advantages(self, decisions: list[Decision], values: list, next_values: list) -> None:
        """Assign GAE advantages to the PPO decisions of one episode."""
        gamma, lambda_ = self.config.reward.gamma, self.config.reward.gae_lambda
        gae = 0.0
        for index in reversed(range(len(decisions))):
            last = index == len(decisions) - 1
            delta = decisions[index].reward + gamma * next_values[index] * (not last)
            delta -= values[index]
            gae = delta + gamma * lambda_ * (not last) * gae
            decisions[index].advantage = gae


def _end(move: Move, pose: np.ndarray) -> np.ndarray:
    """Return the XZ position after a move from a pose, before snapping."""
    return move.poses[-1, :2] if len(move.poses) else pose[:2]
