"""Trajectory files: Scan2Instruction episodes with their images, instructions and actions.

A trajectory file goes through these stages:

    planning   -> trajectories with a planned path, poses and actions
    rendering  -> training frames, route frames and goal views per trajectory
    labeling   -> an instruction per trajectory, written one at a time

A trajectory is labeled when it has a non-empty instruction. Poses and
actions follow one_scan_is_enough.poses.

Only this module knows the HDF5 layout. One process writes a file at a time,
and no other process reads it during the write.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from one_scan_is_enough.poses import increments


@dataclass
class Episode:
    """A planned reference trajectory with its instruction."""

    key: str
    instruction: str
    poses: np.ndarray  # (M, 3) scene poses; the first is the start, the last the goal.
    path_xz: np.ndarray  # (N, 2) planned reference path in meters.
    instruction_type: str  # A key of one_scan_is_enough.instruction.prompts.INSTRUCTION_TYPES.

    @property
    def length(self) -> float:
        """Length of the reference pose sequence in meters."""
        return _polyline_length(self.poses[:, :2])

    @property
    def path_length(self) -> float:
        """Length of the planned reference path in meters, the shortest path of SPL."""
        return _polyline_length(self.path_xz)


@dataclass
class Instruction:
    """The instruction of a labeled trajectory."""

    text: str
    type: str  # A key of one_scan_is_enough.instruction.prompts.INSTRUCTION_TYPES.


class TrajectoryFile:
    """One open trajectory file.

    Open the file in a context or call close. Each process and each loader
    worker must open its own instance. Opening a new file in a write mode
    starts it with no trajectories and no complete stage.
    """

    def __init__(self, path: str | Path, mode: str = "r", **attributes):
        """Open a trajectory file.

        Args:
            path: The HDF5 file.
            mode: An h5py file mode.
            **attributes: File attributes for a new file, such as scene_name.
                They are not written to an existing file.
        """
        self.path = Path(path)
        self.file = h5py.File(self.path, mode)
        if mode != "r" and "trajectories" not in self.file:
            self.file.attrs["planning_complete"] = False
            self.file.attrs["rendering_complete"] = False
            self.file.attrs.update(attributes)
            self.file.create_group("trajectories", track_order=True)

    def __enter__(self) -> "TrajectoryFile":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.file.close()

    @property
    def scene_name(self) -> str:
        return self.file.attrs["scene_name"]

    def complete(self, stage: str) -> bool:
        """Whether the planning or rendering stage completed for the whole file."""
        return bool(self.file.attrs[f"{stage}_complete"])

    def require(self, stage: str) -> None:
        """Raise ValueError unless the planning or rendering stage is complete."""
        if not self.complete(stage):
            raise ValueError(f"The {stage} stage of {self.path} is incomplete.")

    def finish(self, stage: str) -> None:
        """Mark the planning or rendering stage complete for the whole file."""
        self.file.attrs[f"{stage}_complete"] = True

    def keys(self) -> list[str]:
        """Return the trajectory keys in sorted order."""
        return sorted(self.file["trajectories"])

    # ---- Planning ----

    def add_planned(self, key: str, path_xz: np.ndarray, poses: np.ndarray) -> None:
        """Add a trajectory with its path (N, 2), scene poses (M, 3) and their actions."""
        group = self.file["trajectories"].create_group(key)
        group.create_dataset("path_xz", data=path_xz)
        group.create_dataset("poses", data=poses)
        group.create_dataset("actions", data=increments(poses))

    def poses(self, key: str) -> np.ndarray:
        """Return the (M, 3) scene poses of a planned trajectory."""
        return self.file[f"trajectories/{key}/poses"][...]

    def episodes(self) -> list[Episode]:
        """Return every labeled planned trajectory in key order."""
        episodes = []
        for key in self.keys():
            instruction = self.instruction(key)
            if instruction is not None:
                group = self.file[f"trajectories/{key}"]
                episodes.append(
                    Episode(
                        key,
                        instruction.text,
                        group["poses"][...],
                        group["path_xz"][...],
                        instruction.type,
                    )
                )
        return episodes

    # ---- Rendering ----

    def add_images(
        self,
        key: str,
        training: list[bytes],
        route: list[bytes],
        goal: tuple[np.ndarray, np.ndarray, list[bytes]],
    ) -> None:
        """Write the rendered JPEG images of one trajectory.

        The trajectory group is created when the file does not have it.

        Args:
            key: The trajectory key.
            training: Frames of the training images, one per action at the even
                poses.
            route: Frames with route markers for labeling.
            goal: The aligned XYZ look-at target (3,), aligned XYZ camera
                centers (G, 3) and frames of the goal views.
        """
        group = self.file["trajectories"].require_group(key)
        _write_jpegs(group.create_group("training_images"), training)
        _write_jpegs(group.create_group("instruction_images/route"), route)
        target, cameras, frames = goal
        views = group.create_group("instruction_images/goal")
        views.create_dataset("target", data=target)
        views.create_dataset("cameras", data=cameras.reshape(-1, 3))
        _write_jpegs(views, frames)

    def copy_images(self, source: "TrajectoryFile", keys: list[str]) -> None:
        """Copy the rendered images of trajectories from another file into this file."""
        for key in keys:
            group = source.file[f"trajectories/{key}"]
            destination = self.file[f"trajectories/{key}"]
            source.file.copy(group["training_images"], destination)
            source.file.copy(group["instruction_images"], destination)

    def instruction_images(self, key: str) -> tuple[list[bytes], list[bytes]]:
        """Return the route frames and goal views of a trajectory as JPEG bytes."""
        images = self.file[f"trajectories/{key}/instruction_images"]
        return _read_jpegs(images["route"]), _read_jpegs(images["goal"])

    def frames(self, key: str, indices: list[int]) -> list[np.ndarray]:
        """Read training frames as (H, W, 3) uint8 RGB arrays in requested order.

        Only the requested JPEG frames are read and decoded. Repeated indices
        remain repeated in the output.
        """
        group = self.file[f"trajectories/{key}/training_images/rgb"]
        offsets = group["offsets"][...]
        frames = []
        for index in indices:
            if not 0 <= index < len(offsets) - 1:
                raise IndexError(index)
            start, end = map(int, offsets[index : index + 2])
            frames.append(decode_jpeg(group["data"][start:end].tobytes()))
        return frames

    def actions(self, key: str) -> np.ndarray:
        """Return the (T, 2, 3) actions of a trajectory, one per training frame."""
        return self.file[f"trajectories/{key}/actions"][...]

    # ---- Labeling ----

    def instruction(self, key: str) -> Instruction | None:
        """Return the instruction of a trajectory, or None when it is not labeled."""
        group = self.file[f"trajectories/{key}"]
        if "instruction/text" not in group:
            return None
        text = group["instruction/text"].asstr()[()]
        if not text.strip():
            return None
        return Instruction(text, group["instruction/type"].asstr()[()])

    def write_instruction(self, key: str, text: str, instruction_type: str) -> None:
        """Write the instruction of a trajectory, replacing an earlier one, and flush the file."""
        group = self.file[f"trajectories/{key}"]
        if "instruction" in group:
            del group["instruction"]
        string = h5py.string_dtype("utf-8")
        group.create_dataset("instruction/text", data=text, dtype=string)
        group.create_dataset("instruction/type", data=instruction_type, dtype=string)
        self.file.flush()


def decode_jpeg(data: bytes) -> np.ndarray:
    """Decode a JPEG into an (H, W, 3) uint8 RGB array with Pillow."""
    with Image.open(BytesIO(data)) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def _write_jpegs(group: h5py.Group, frames: list[bytes]) -> None:
    """Pack a JPEG sequence into rgb/data with frame boundaries in rgb/offsets."""
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.uint64), np.cumsum([len(frame) for frame in frames], dtype=np.uint64))
    )
    group.create_dataset("rgb/offsets", data=offsets.astype(np.uint64))
    group.create_dataset("rgb/data", data=np.frombuffer(b"".join(frames), dtype=np.uint8))
    group["rgb"].attrs["encoding"] = "jpeg"


def _read_jpegs(group: h5py.Group) -> list[bytes]:
    """Load a packed JPEG sequence without decoding the image pixels."""
    offsets = group["rgb/offsets"][...]
    data = group["rgb/data"]
    return [data[int(start) : int(end)].tobytes() for start, end in zip(offsets[:-1], offsets[1:])]


def _polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
