"""PRPT-KS post-training of RealNav in one scene.

Rank 0 rolls out the current policy until a buffer is full and broadcasts it.
Every rank then trains on its share of the buffer. Each micro-batch packs
rollout decisions with reference-trajectory steps of the same scene. The loss
combines a clipped PPO objective under a Gaussian likelihood, a weighted DPO
objective on planner preference pairs, and behavior cloning on the reference
steps. Training ends after one pass over the reference steps.

The adamw optimizer runs under DeepSpeed ZeRO-1 with fp32 master weights.
The adamw8bit optimizer keeps 8-bit Adam states and updates the bf16
weights with stochastic rounding in place of fp32 master weights, which
fits full post-training on one 80 GB GPU.
"""

import json
import logging
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator, DeepSpeedPlugin, InitProcessGroupKwargs
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from one_scan_is_enough.policy.data import DecisionSteps, pack
from one_scan_is_enough.policy.inputs import encode, history_indices
from one_scan_is_enough.policy.model import RealNavModel
from one_scan_is_enough.policy.predictor import NavigationPolicy
from one_scan_is_enough.posttrain.rollout import Decision, EpisodeSampler, Rollout, RolloutWorker
from one_scan_is_enough.scene import Scene
from one_scan_is_enough.simulation.environment import SceneEnvironment
from one_scan_is_enough.trajectories import TrajectoryFile

logger = logging.getLogger(__name__)


class AnchorStream:
    """One shuffled pass over reference decision steps, sharded across ranks."""

    def __init__(self, total: int, seed: int, rank: int, world: int):
        order = list(range(total))
        # The offset keeps this order independent of the other streams seeded from seed.
        random.Random(seed + 2_000_003).shuffle(order)
        self.order = order[rank::world]
        self.position = 0

    def take(self, count: int) -> list[int]:
        """Return up to count step indices; fewer once the pass is exhausted."""
        indices = self.order[self.position : self.position + count]
        self.position += len(indices)
        return indices


class RoundBatches(Dataset):
    """The micro-batches of one round, each packing rollout decisions with reference steps.

    Item i is the i-th micro-batch of the round. Batches go to ranks in turn and
    take reference steps from the anchor stream of their rank in that order, so
    loader workers can build them in any order.
    """

    def __init__(
        self,
        rollout: Rollout,
        steps: DecisionSteps,
        processor,
        anchors: list[AnchorStream],
        order: list[int],
        rollout_batch: int,
        anchor_batch: int,
    ):
        self.rollout = rollout
        self.steps = steps
        self.processor = processor
        batches = [order[i : i + rollout_batch] for i in range(0, len(order), rollout_batch)]
        # Ranks get equal numbers of batches, as accelerate pads its shards.
        world = len(anchors)
        batches += batches[: -len(batches) % world]
        self.batches = [
            (batch, anchors[index % world].take(anchor_batch))
            for index, batch in enumerate(batches)
        ]

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        indices, anchor_indices = self.batches[index]
        decisions = [self.rollout.decisions[i] for i in indices]
        rollout = [self._sample(decision) for decision in decisions]
        anchors = [self.steps[i] for i in anchor_indices]
        batch = pack(rollout + anchors)
        count = len(anchors)
        batch["reference"] = torch.stack(
            [torch.from_numpy(d.reference).float() for d in decisions]
            + [sample["waypoints"] for sample in anchors]
        )
        batch["ppo"] = torch.tensor([not d.preference for d in decisions] + [False] * count)
        batch["preference"] = torch.tensor([d.preference for d in decisions] + [False] * count)
        batch["anchor"] = torch.tensor([False] * len(decisions) + [True] * count)
        batch["advantages"] = torch.tensor([d.advantage for d in decisions] + [0.0] * count)
        batch["reference_logits"] = torch.tensor(
            [d.reference_logit for d in decisions] + [0.0] * count
        )
        batch["weights"] = torch.tensor([d.weight for d in decisions] + [1.0] * count)
        return batch

    def _sample(self, decision: Decision) -> dict[str, torch.Tensor]:
        frames = self.rollout.frames[decision.episode]
        ids = history_indices(decision.frames)
        sample = encode(self.processor, decision.instruction, [frames[i] for i in ids])
        sample["waypoints"] = torch.from_numpy(decision.action).float()
        return sample


def _single(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return items[0]


def prpt_loss(
    model, normalize, batch: dict[str, torch.Tensor], config: DictConfig, sigma: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the weighted loss and its terms for one packed micro-batch.

    Likelihoods are Gaussian with scale sigma around the predicted waypoints
    in normalized space. PPO ratios compare the executed action under the
    prediction with its likelihood under the rollout reference action.
    """
    output = model(
        input_ids=batch["input_ids"],
        pixel_values=batch["pixel_values"],
        image_grid_thw=batch["image_grid_thw"],
        lengths=batch["lengths"],
    )
    predicted = normalize(output.waypoints)
    action = normalize(batch["waypoints"])
    reference = normalize(batch["reference"])

    def log_likelihood(target, mean):
        return -(target - mean).square().sum(dim=(1, 2)) / (2.0 * sigma * sigma)

    zero = predicted.sum() * 0.0
    ppo, preference, anchor = batch["ppo"], batch["preference"], batch["anchor"]
    rl = bc = dpo = zero
    ratio = torch.zeros(0, device=predicted.device)
    if ppo.any():
        ratio = torch.exp(
            log_likelihood(action[ppo], predicted[ppo])
            - log_likelihood(action[ppo], reference[ppo])
        )
        advantages = batch["advantages"][ppo]
        clipped = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        rl = -torch.min(ratio * advantages, clipped * advantages).mean()
    if anchor.any():
        bc = F.mse_loss(predicted[anchor], action[anchor])
    if preference.any():
        margin = (
            log_likelihood(action[preference], predicted[preference])
            - log_likelihood(reference[preference], predicted[preference])
            - batch["reference_logits"][preference]
        )
        losses = -F.logsigmoid(config.dpo_beta * margin)
        dpo = (losses * batch["weights"][preference]).mean()
    loss = config.rl * rl + config.bc * bc + config.dpo * dpo
    return loss, {
        "loss": loss.item(),
        "rl_loss": rl.item(),
        "bc_loss": bc.item(),
        "dpo_loss": dpo.item(),
        "ratio": ratio.mean().item() if len(ratio) else 0.0,
        "ppo": int(ppo.sum()),
        "preference": int(preference.sum()),
        "anchor": int(anchor.sum()),
    }


def learning_rate_ratio(step: int, total: int, warmup_ratio: float, min_ratio: float) -> float:
    """Linear warmup over the first steps, then cosine decay to min_ratio."""
    warmup = math.ceil(total * warmup_ratio)
    if warmup > 0 and step <= warmup:
        return step / warmup
    progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def likelihood_scale(round_index: int, rounds: int, start: float, end: float) -> float:
    """Cosine decay of the likelihood scale from start to end over the rollout rounds."""
    progress = min(max(round_index, 0) / max(1, rounds - 1), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def module_parameters(model: RealNavModel) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    """Partition named parameters into vision, language and head groups."""
    groups = {"vision": [], "language": [], "head": []}
    for name, parameter in model.named_parameters():
        if name.startswith("model.visual."):
            groups["vision"].append((name, parameter))
        elif name.startswith("action_head."):
            groups["head"].append((name, parameter))
        else:
            groups["language"].append((name, parameter))
    return groups


def train(config: DictConfig) -> Path:
    """Post-train the base policy in the configured scene and save it to the output directory."""
    training, loss_config = config.training, config.loss
    world = int(os.environ.get("WORLD_SIZE", 1))
    plugin = None
    if training.optimizer == "adamw":
        deepspeed = {
            "zero_optimization": {"stage": 1},
            "bf16": {"enabled": True},
            "train_micro_batch_size_per_gpu": training.rollout_batch,
            "gradient_accumulation_steps": training.gradient_accumulation_steps,
            "train_batch_size": training.rollout_batch
            * training.gradient_accumulation_steps
            * world,
        }
        plugin = DeepSpeedPlugin(hf_ds_config=deepspeed)
    elif training.optimizer != "adamw8bit":
        raise ValueError(f"Unknown optimizer: {training.optimizer}")
    accelerator = Accelerator(
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        mixed_precision="bf16",
        deepspeed_plugin=plugin,
        # Other ranks wait for rank 0 while it rolls out a buffer.
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))],
    )
    rank, main = accelerator.process_index, accelerator.is_main_process
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
    output = Path(config.output_dir)

    processor = AutoProcessor.from_pretrained(config.base)
    model = RealNavModel.from_pretrained(
        config.base, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False
    if training.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    groups = []
    for module, parameters in module_parameters(model).items():
        rate = training.learning_rates[module]
        for _, parameter in parameters:
            parameter.requires_grad = rate > 0
        if rate > 0:
            groups.append({"params": [p for _, p in parameters], "lr": rate})
    if training.optimizer == "adamw8bit":
        from torchao.prototype.low_bit_optim import AdamW8bit

        optimizer = AdamW8bit(groups, weight_decay=0.0, bf16_stochastic_round=True)
        for group in optimizer.param_groups:
            group["lr"] = torch.tensor(group["lr"])  # Updated in place, see below.
    else:
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    engine, optimizer = accelerator.prepare(model, optimizer)
    core = accelerator.unwrap_model(engine)
    base_rates = [float(group["lr"]) for group in optimizer.param_groups]

    scene = Scene(config.scene_dir)
    steps = DecisionSteps(scene.trajectories_path, processor, training.anchor_end_repeat)
    # Every rank holds the anchor streams of all ranks to lay out whole rounds.
    anchors = [AnchorStream(len(steps), config.seed, r, world) for r in range(world)]
    per_step = training.gradient_accumulation_steps * world
    total_steps = training.max_steps or max(
        1, math.ceil(len(steps) / (training.anchor_batch * per_step))
    )
    steps_per_round = math.ceil(config.rollout.buffer_steps / (training.rollout_batch * per_step))
    rounds = max(1, math.ceil(total_steps / steps_per_round))
    worker = None
    if main:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yaml").write_text(OmegaConf.to_yaml(config))
        environment = SceneEnvironment(scene, config.environment, accelerator.device)
        with TrajectoryFile(scene.trajectories_path) as file:
            episodes = file.episodes()
        sampler = EpisodeSampler(episodes, config.rollout.length_curriculum, config.seed)
        worker = RolloutWorker(
            NavigationPolicy(core, processor), environment, sampler, config.rollout
        )
        logger.info(
            "Reference steps: %d, optimizer steps: %d, estimated rounds: %d",
            len(steps),
            total_steps,
            rounds,
        )

    step = round_index = 0
    while step < total_steps:
        # The rollout uses the scale of the previous round; the update uses this round's.
        scales = [
            likelihood_scale(i, rounds, loss_config.sigma_start, loss_config.sigma_end)
            for i in (round_index - 1, round_index)
        ]
        started = time.monotonic()
        payload = [None]
        if main:
            worker.sampler.restart(min(step / total_steps, 1.0))
            core.eval()
            payload = [worker.collect(config.rollout.buffer_steps, scales[0])]
            core.train()
        if world > 1:
            dist.broadcast_object_list(payload, src=0)
        rollout = payload[0]
        order = list(range(len(rollout.decisions)))
        engine.train()
        random.Random(config.seed + step).shuffle(order)
        batches = RoundBatches(
            rollout,
            steps,
            processor,
            anchors,
            order,
            training.rollout_batch,
            training.anchor_batch,
        )
        # Workers fork per round and share the rollout frames copy-on-write.
        loader = accelerator.prepare_data_loader(
            DataLoader(
                batches,
                batch_size=1,
                collate_fn=_single,
                num_workers=training.loader_workers,
                pin_memory=True,
            )
        )
        rolled_out = time.monotonic()
        metrics = []
        for batch in loader:
            lr_ratio = learning_rate_ratio(
                step + 1, total_steps, training.warmup_ratio, training.min_lr_ratio
            )
            for group, rate in zip(optimizer.param_groups, base_rates):
                if isinstance(group["lr"], torch.Tensor):
                    group["lr"].fill_(rate * lr_ratio)
                else:
                    group["lr"] = rate * lr_ratio
            with accelerator.accumulate(engine):
                loss, terms = prpt_loss(engine, core.normalize, batch, loss_config, scales[1])
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            metrics.append(terms)
            if accelerator.sync_gradients:
                step += 1
                if step >= total_steps:
                    break
        round_index += 1
        if main:
            seconds = (rolled_out - started, time.monotonic() - rolled_out)
            _log_round(
                output,
                round_index,
                step,
                total_steps,
                rollout,
                metrics,
                scales[1],
                lr_ratio,
                seconds,
            )
        if round_index % training.save_every == 0:
            _save(accelerator, engine, processor, output / "checkpoints" / f"round-{round_index}")
    _save(accelerator, engine, processor, output)
    accelerator.end_training()
    return output


def _log_round(
    output, round_index, step, total_steps, rollout, metrics, sigma, lr_ratio, seconds
) -> None:
    record = {
        "round": round_index,
        "step": step,
        "total_steps": total_steps,
        "sigma": sigma,
        "lr_ratio": lr_ratio,
        # Wall time of the rollout and of the updates of the round.
        "rollout_seconds": seconds[0],
        "update_seconds": seconds[1],
    }
    executed = [d for d in rollout.decisions if not d.preference]
    episodes = len(rollout.frames)
    record.update(
        episodes=episodes,
        decisions=len(rollout.decisions),
        success_rate=sum(rollout.successes) / episodes,
        episode_return=sum(d.reward for d in executed) / episodes,
        progress=sum(d.progress for d in executed) / len(executed),
        planner_rate=sum(d.planner for d in executed) / len(executed),
    )
    summary = "success {success_rate:.3f}, progress {progress:.3f}, ".format(**record)
    summary += "planner {planner_rate:.3f}, ".format(**record)
    for key in ("loss", "rl_loss", "bc_loss", "dpo_loss", "ratio"):
        record[key] = sum(m[key] for m in metrics) / max(1, len(metrics))
    for key in ("ppo", "preference", "anchor"):
        record[key] = sum(m[key] for m in metrics)
    with (output / "metrics.jsonl").open("a") as file:
        file.write(json.dumps(record) + "\n")
    logger.info(
        "Round %d step %d/%d: %sloss %.4f (rl %.4f, bc %.4f, dpo %.4f); "
        "rollout %.0f s, updates %.0f s",
        round_index,
        step,
        total_steps,
        summary,
        record["loss"],
        record["rl_loss"],
        record["bc_loss"],
        record["dpo_loss"],
        *seconds,
    )


def _save(accelerator: Accelerator, engine, processor, directory: Path) -> None:
    accelerator.wait_for_everyone()
    state = accelerator.get_state_dict(engine)
    model = accelerator.unwrap_model(engine)
    model.config.use_cache = True
    model.save_pretrained(
        directory,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state,
    )
    model.config.use_cache = False
    if accelerator.is_main_process:
        processor.save_pretrained(directory)
