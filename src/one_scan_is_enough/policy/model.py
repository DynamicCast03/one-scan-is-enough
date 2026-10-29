"""RealNav: Qwen3-VL with three navigation tokens and an MLP waypoint head.

The model config carries the navigation token ids and the waypoint normalization
(normalized = metric * scale + bias), so checkpoints load without extra files.
Inputs and outputs follow the waypoint axes of one_scan_is_enough.policy.inputs.
"""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers.utils import ModelOutput

from one_scan_is_enough.policy.inputs import NAV_TOKENS, WAYPOINTS


class WaypointHead(nn.Module):
    """Project each navigation token state, then map their concatenation to waypoints."""

    def __init__(self, hidden_size: int, width: int = 256):
        super().__init__()
        self.proj1 = nn.Sequential(
            nn.Linear(hidden_size, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        flat = len(NAV_TOKENS) * width
        self.proj2 = nn.Sequential(nn.Linear(flat, flat), nn.GELU(), nn.Linear(flat, WAYPOINTS * 3))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Map (B, 3, hidden) token states to (B, 8, 3) normalized waypoints."""
        return self.proj2(self.proj1(states).flatten(1)).view(-1, WAYPOINTS, 3)


class FrameCache:
    """Visual features of recently seen frames keyed by caller-chosen frame ids.

    Qwen3-VL encodes every image on its own, so a cached feature equals a fresh
    encoding of the same pixels. Entries beyond the capacity are evicted from
    the least recently used end; ids of the current call are never evicted.
    """

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._entries: OrderedDict[int, tuple[torch.Tensor, torch.Tensor, list]] = OrderedDict()
        self._request: list[int] | None = None

    def __contains__(self, frame_id: int) -> bool:
        return frame_id in self._entries

    def grid(self, frame_id: int) -> torch.Tensor:
        """Return the (3,) CPU image grid of a cached frame."""
        return self._entries[frame_id][0]

    def features(self, encode, pixel_values, image_grid_thw):
        """Encode the uncached requested frames and return features of all of them.

        pixel_values holds the uncached frames in request order; image_grid_thw
        holds the grid of every requested frame.
        """
        ids = self._request
        missing = [index for index, frame_id in enumerate(ids) if frame_id not in self._entries]
        if missing:
            grids = image_grid_thw[missing]
            embeds, deepstack = encode(pixel_values, grids)
            sizes = [len(e) for e in embeds]
            layers = [layer.split(sizes) for layer in deepstack]
            for position, index in enumerate(missing):
                self._entries[ids[index]] = (
                    grids[position].cpu(),
                    embeds[position],
                    [layer[position] for layer in layers],
                )
        for frame_id in ids:
            self._entries.move_to_end(frame_id)
        while len(self._entries) > max(self.capacity, len(ids)):
            self._entries.popitem(last=False)
        entries = [self._entries[frame_id] for frame_id in ids]
        deepstack = [torch.cat(layer) for layer in zip(*(entry[2] for entry in entries))]
        return tuple(entry[1] for entry in entries), deepstack


class _Backbone(Qwen3VLModel):
    """Qwen3-VL backbone whose image features can come from a FrameCache."""

    def __init__(self, config):
        super().__init__(config)
        self.frame_cache: FrameCache | None = None

    def get_image_features(self, pixel_values, image_grid_thw=None):
        if self.frame_cache is None:
            return super().get_image_features(pixel_values, image_grid_thw)
        return self.frame_cache.features(super().get_image_features, pixel_values, image_grid_thw)


@dataclass
class RealNavOutput(ModelOutput):
    waypoints: torch.Tensor | None = None  # (S, 8, 3) metric waypoints, float32.


class RealNavModel(Qwen3VLForConditionalGeneration):
    """Predict eight metric waypoints from the final states of the navigation tokens.

    The language model head is unused.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = _Backbone(config)
        self.action_head = WaypointHead(config.text_config.hidden_size)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        lengths: torch.LongTensor | None = None,
        **kwargs,
    ) -> RealNavOutput:
        """Predict waypoints for every sequence in input_ids.

        input_ids is (B, L) for B unpadded sequences of equal length, or (1, N)
        for packed sequences whose token counts are given by lengths (S,).
        Packing requires flash_attention_2.
        """
        if lengths is not None:
            kwargs.update(self._packing(input_ids, image_grid_thw, lengths))
        elif kwargs.get("attention_mask") is None:
            # Without a mask, transformers infers sequence boundaries from the
            # multimodal positions and splits sdpa and eager attention wrongly.
            kwargs["attention_mask"] = torch.ones_like(input_ids)
        hidden = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            **kwargs,
        ).last_hidden_state
        nav = torch.isin(input_ids, torch.tensor(self.config.nav_token_ids, device=hidden.device))
        states = hidden[nav].view(-1, len(NAV_TOKENS), hidden.shape[-1])
        predicted = self.action_head(states).float()
        scale, bias = self._normalization(predicted.device)
        return RealNavOutput(waypoints=(predicted - bias) / scale)

    def normalize(self, waypoints: torch.Tensor) -> torch.Tensor:
        """Map metric waypoints (..., 3) to the float32 space of the head output."""
        scale, bias = self._normalization(waypoints.device)
        return waypoints.float() * scale + bias

    @contextmanager
    def frame_cache(self, cache: FrameCache, frame_ids: list[int]):
        """Serve image features of frame_ids from the cache within the context.

        Inside the context, pixel_values passed to forward hold only the frames
        missing from the cache, in frame_ids order, and image_grid_thw holds
        all frames.
        """
        cache._request = frame_ids
        self.model.frame_cache = cache
        try:
            yield
        finally:
            self.model.frame_cache = None
            cache._request = None

    def _normalization(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.tensor(self.config.waypoint_scale, dtype=torch.float32, device=device)
        bias = torch.tensor(self.config.waypoint_bias, dtype=torch.float32, device=device)
        return scale, bias

    def _packing(self, input_ids, image_grid_thw, lengths) -> dict:
        """Return per-sequence multimodal positions and varlen attention boundaries."""
        sequences = input_ids[0].split(lengths.tolist())
        images = [int((s == self.config.vision_start_token_id).sum()) for s in sequences]
        positions = [
            self.model.get_rope_index(sequence[None], grid, None, None)[0]
            for sequence, grid in zip(sequences, image_grid_thw.split(images))
        ]
        boundaries = F.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
        longest = int(lengths.max())
        return {
            "position_ids": torch.cat(positions, dim=2),
            "cu_seq_lens_q": boundaries,
            "cu_seq_lens_k": boundaries,
            "max_length_q": longest,
            "max_length_k": longest,
        }
