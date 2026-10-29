"""Closed-loop RealNav inference over a growing observation history."""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

from one_scan_is_enough.policy.inputs import history_indices, token_ids
from one_scan_is_enough.policy.model import FrameCache, RealNavModel


class NavigationPolicy:
    """A loaded RealNav model with the image processor it was trained with."""

    def __init__(self, model: RealNavModel, processor):
        self.model = model.eval()
        self.processor = processor

    @classmethod
    def from_pretrained(
        cls, path: str | Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16
    ) -> "NavigationPolicy":
        model = RealNavModel.from_pretrained(
            path, dtype=dtype, attn_implementation="flash_attention_2"
        )
        return cls(model.to(device), AutoProcessor.from_pretrained(path))

    def start(self, instruction: str, cache_frames: int = 256) -> "PolicySession":
        """Start the inference session of one navigation run; sessions share no frames or caches."""
        return PolicySession(self, instruction, FrameCache(cache_frames))


class PolicySession:
    """Observation history and visual feature cache of one navigation run.

    Call observe for every new RGB frame, in time order, and predict at each
    decision step. Each prediction sees up to 18 history frames.
    """

    def __init__(self, policy: NavigationPolicy, instruction: str, cache: FrameCache):
        self.policy = policy
        self.instruction = instruction
        self.cache = cache
        self.frames: list[np.ndarray] = []

    def observe(self, frame: np.ndarray) -> None:
        """Append an (H, W, 3) uint8 RGB frame; the policy was trained on 640x480."""
        self.frames.append(frame)

    @torch.inference_mode()
    def predict(self) -> np.ndarray:
        """Return (8, 3) float32 metric waypoints relative to the latest frame."""
        model, processor = self.policy.model, self.policy.processor
        ids = history_indices(len(self.frames))
        missing = [frame_id for frame_id in ids if frame_id not in self.cache]
        grids = {}
        pixel_values = torch.empty(0)
        if missing:
            pixels = processor.image_processor(
                images=[self.frames[frame_id] for frame_id in missing], return_tensors="pt"
            )
            pixel_values = pixels["pixel_values"]
            grids = dict(zip(missing, pixels["image_grid_thw"]))
        grid = torch.stack([grids[i] if i in grids else self.cache.grid(i) for i in ids])
        input_ids = token_ids(processor, self.instruction, grid)[None]
        device = model.device
        with model.frame_cache(self.cache, ids):
            output = model(
                input_ids=input_ids.to(device),
                pixel_values=pixel_values.to(device),
                image_grid_thw=grid.to(device),
            )
        return output.waypoints[0].cpu().numpy()
