"""Model inputs and waypoint labels shared by policy training and inference.

Waypoints and trajectory actions follow one_scan_is_enough.poses, relative to the pose
at the decision step.
"""

import numpy as np
import torch

from one_scan_is_enough.poses import accumulate

NAV_TOKENS = ("<nav1>", "<nav2>", "<nav3>")
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
HISTORY_FRAMES = 18
# Each trajectory action holds two increments; four actions give eight waypoints.
HORIZON_ACTIONS = 4
WAYPOINTS = 2 * HORIZON_ACTIONS

_TASK = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a video of "
    "historical observations and an image of the current observation {images}. Your assigned "
    "task is: '{task}'. Based on this sequence of images, predict your future trajectory."
)


def prompt(instruction: str, frames: int) -> str:
    """Return the chat text for one decision step with one image placeholder per frame.

    Frame labels and placeholders are joined without separators.
    """
    images = "".join(f"<Observation {i}/{frames}>{IMAGE}" for i in range(1, frames + 1))
    return (
        f"<|im_start|>user\n{_TASK.format(images=images, task=instruction)}<|im_end|>\n"
        f"<|im_start|>assistant\n{''.join(NAV_TOKENS)}<|im_end|>\n"
    )


def history_indices(count: int) -> list[int]:
    """Return the observed frame indices fed to the policy, oldest first.

    All frames are used up to HISTORY_FRAMES; longer histories are sampled
    uniformly and always keep the first and the current frame.
    """
    if count < 1:
        raise ValueError("The history needs at least one frame.")
    if count <= HISTORY_FRAMES:
        return list(range(count))
    return np.linspace(0, count - 1, HISTORY_FRAMES, dtype=int).tolist()


def encode(processor, instruction: str, frames: list[np.ndarray]) -> dict[str, torch.Tensor]:
    """Return the model inputs of one decision step.

    Args:
        processor: The Qwen3-VL processor of the policy.
        instruction: The navigation instruction.
        frames: The (H, W, 3) uint8 RGB history frames that history_indices
            selects, oldest first.

    Returns:
        input_ids (L,), pixel_values (P, C) and image_grid_thw (F, 3).
    """
    pixels = processor.image_processor(images=frames, return_tensors="pt")
    return {
        "input_ids": token_ids(processor, instruction, pixels["image_grid_thw"]),
        "pixel_values": pixels["pixel_values"],
        "image_grid_thw": pixels["image_grid_thw"],
    }


def token_ids(processor, instruction: str, grids: torch.Tensor) -> torch.Tensor:
    """Return the (L,) prompt token ids for history frames with image grids (F, 3).

    Each image placeholder expands to one token per merged image patch, as the
    Qwen3-VL processor expands it. Inference with cached frame features needs
    the grids without the pixels, so all callers use this expansion.
    """
    merge = processor.image_processor.merge_size**2
    parts = prompt(instruction, len(grids)).split(IMAGE_PAD)
    text = "".join(
        part + IMAGE_PAD * (int(size.prod()) // merge) for part, size in zip(parts, grids)
    )
    return processor.tokenizer(text + parts[-1], return_tensors="pt")["input_ids"][0]


def waypoint_labels(actions: np.ndarray, step: int) -> np.ndarray:
    """Return (8, 3) float32 waypoints from the four actions starting at a frame.

    Actions are (T, 2, 3) increments. Increments past the trajectory end are
    zero, which repeats the last pose.
    """
    steps = np.zeros((WAYPOINTS, 3), dtype=np.float64)
    upcoming = actions[step : step + HORIZON_ACTIONS].reshape(-1, 3)
    steps[: len(upcoming)] = upcoming
    return accumulate(steps).astype(np.float32)
