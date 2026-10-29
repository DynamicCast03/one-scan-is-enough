"""Read and write calibrated scene packages with PLY assets and HDF5 maps.

Gaussian geometry uses the source world frame. BEV maps use the aligned
frame. Lengths are meters and angles are radians.
HDF5 updates require one writer and no concurrent readers of the same file.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class BevMap:
    """Maps with rows along +Z and columns along +X; 255 means free."""

    raw: np.ndarray  # (H, W), uint8, before safety filtering and manual edits.
    navigable: np.ndarray  # (H, W), uint8, final safe and edited map.
    origin_xz: np.ndarray  # (2,), coordinate of pixel (0, 0) in meters.
    resolution: float  # Meters per pixel.


def write_bev(group: h5py.Group, bev: BevMap) -> None:
    """Write one BEV into an empty HDF5 group."""
    group.create_dataset("raw", data=bev.raw, compression="gzip", shuffle=True)
    group.create_dataset("navigable", data=bev.navigable, compression="gzip", shuffle=True)
    group.create_dataset("origin_xz", data=bev.origin_xz)
    group.create_dataset("resolution", data=bev.resolution)


def read_bev(group: h5py.Group) -> BevMap:
    """Load a BEV independent of the HDF5 file lifetime."""
    return BevMap(
        group["raw"][...],
        group["navigable"][...],
        group["origin_xz"][...],
        float(group["resolution"][()]),
    )


class Scene:
    """Access a calibrated scene package by its fixed directory layout.

    No file handles remain open between calls. Callers can pass the Gaussian
    path directly to third-party tools. Creating a package is owned by
    the preparation workflow; this class writes and reads its HDF5 content.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    @property
    def gaussian_path(self) -> Path:
        return self.directory / "gaussians.ply"

    @property
    def data_path(self) -> Path:
        return self.directory / "scene.h5"

    @property
    def trajectories_path(self) -> Path:
        return self.directory / "trajectories.h5"

    def write(
        self,
        name: str,
        transform: np.ndarray,
        bev: BevMap,
    ) -> None:
        """Create the scene map after calibration; never replace an existing file."""
        with h5py.File(self.data_path, "x") as file:
            file.attrs["scene_name"] = name
            file.create_dataset("alignment/world_to_aligned", data=transform)
            write_bev(file.create_group("bev"), bev)

    def read_geometry(self) -> tuple[np.ndarray, BevMap]:
        """Read the world-to-aligned transform and navigation maps."""
        with h5py.File(self.data_path, "r") as file:
            return file["alignment/world_to_aligned"][...], read_bev(file["bev"])

    def read_name(self) -> str:
        """Read the scene name that trajectory files of this scene also carry."""
        with h5py.File(self.data_path, "r") as file:
            return file.attrs["scene_name"]
