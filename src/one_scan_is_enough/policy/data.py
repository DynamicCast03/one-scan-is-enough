"""Decision-step samples from trajectory files and packed training batches."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from one_scan_is_enough.policy.inputs import (
    HORIZON_ACTIONS,
    encode,
    history_indices,
    waypoint_labels,
)
from one_scan_is_enough.trajectories import TrajectoryFile


class DecisionSteps(Dataset):
    """One sample per frame of every labeled trajectory in a complete trajectory file.

    Trajectories without an instruction are skipped. A sample holds the
    processed prompt and history frames up to that frame and its metric
    waypoint labels. Frames whose four-action horizon reaches the trajectory
    end appear end_repeat times. Each loader worker opens its own file handle.
    """

    def __init__(self, path: str | Path, processor, end_repeat: int = 1):
        self.path = Path(path)
        self.processor = processor
        self.steps: list[tuple[str, int]] = []  # (trajectory key, frame)
        with TrajectoryFile(self.path) as data:
            data.require("rendering")
            for key in data.keys():
                if data.instruction(key) is None:
                    continue
                frames = len(data.actions(key))
                for step in range(frames):
                    repeat = end_repeat if step >= frames - HORIZON_ACTIONS else 1
                    self.steps.extend([(key, step)] * repeat)
        self._data: TrajectoryFile | None = None

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        key, step = self.steps[index]
        if self._data is None:
            self._data = TrajectoryFile(self.path)
        sample = encode(
            self.processor,
            self._data.instruction(key).text,
            self._data.frames(key, history_indices(step + 1)),
        )
        sample["waypoints"] = torch.from_numpy(waypoint_labels(self._data.actions(key), step))
        return sample

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_data"] = None
        return state


def pack(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate samples into one packed sequence for RealNavModel.forward."""
    return {
        "input_ids": torch.cat([s["input_ids"] for s in samples])[None],
        "lengths": torch.tensor([len(s["input_ids"]) for s in samples]),
        "pixel_values": torch.cat([s["pixel_values"] for s in samples]),
        "image_grid_thw": torch.cat([s["image_grid_thw"] for s in samples]),
        "waypoints": torch.stack([s["waypoints"] for s in samples]),
    }
