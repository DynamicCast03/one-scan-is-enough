"""Read standard 3DGS geometry without loading a renderer."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from scipy.special import expit
from tqdm import tqdm

SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class GaussianGeometry:
    """Decoded Gaussian geometry in the source world frame, before alignment."""

    xyz: np.ndarray  # (N, 3), centers in meters.
    scales: np.ndarray  # (N, 3), standard deviations in meters.
    rotations: np.ndarray  # (N, 3, 3), local-to-world rotations.
    opacities: np.ndarray  # (N,), probabilities after sigmoid activation.
    colors: np.ndarray | None = None  # (N, 3), view-independent RGB in [0, 1].


def decode_geometry(
    vertex, *, scale_max: float | None = None, with_colors: bool = False
) -> GaussianGeometry:
    """Decode valid vertices, keeping scale norms at or below scale_max meters.

    With with_colors, the degree-zero SH term is converted to RGB.
    """
    xyz = np.column_stack([vertex[key] for key in ("x", "y", "z")]).astype(np.float32)
    scales = np.exp(np.column_stack([vertex[f"scale_{i}"] for i in range(3)]).astype(np.float32))
    quats = np.column_stack([vertex[f"rot_{i}"] for i in range(4)]).astype(np.float32)
    opacities = expit(np.asarray(vertex["opacity"], dtype=np.float32))
    valid = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(quats).all(axis=1)
        & np.isfinite(opacities)
        & (np.linalg.norm(quats, axis=1) > 0)
    )
    if scale_max is not None:
        valid &= np.linalg.norm(scales, axis=1) <= scale_max
    # A whole batch can be removed by the scale cutoff.
    rotations = (
        Rotation.from_quat(quats[valid][:, [1, 2, 3, 0]]).as_matrix().astype(np.float32)
        if np.any(valid)
        else np.empty((0, 3, 3), dtype=np.float32)
    )
    colors = None
    if with_colors:
        dc = np.column_stack([vertex[f"f_dc_{i}"] for i in range(3)]).astype(np.float32)
        colors = np.clip(0.5 + SH_C0 * dc[valid], 0.0, 1.0)
    return GaussianGeometry(xyz[valid], scales[valid], rotations, opacities[valid], colors)


def iter_geometry(
    path: str | Path,
    batch_size: int,
    *,
    scale_max: float | None = None,
    with_colors: bool = False,
):
    """Yield decoded batches from a memory-mapped binary Gaussian PLY.

    Binary scalar vertex records are required to bound input memory. Each
    yielded batch owns its decoded arrays; higher-order SH terms are never read.
    If set, scale_max excludes vertices with larger scale norms, in meters.
    """
    with Path(path).open("rb") as stream:
        stream.readline()
        if not stream.readline().startswith(b"format binary_"):
            raise ValueError("Scene preparation requires a binary Gaussian PLY.")
    data = PlyData.read(path, mmap="r")
    vertex = data["vertex"].data
    if not isinstance(vertex, np.memmap):
        raise ValueError("Scene preparation requires a memory-mappable binary PLY.")
    for start in range(0, len(vertex), batch_size):
        yield decode_geometry(
            vertex[start : start + batch_size], scale_max=scale_max, with_colors=with_colors
        )


def compute_scale_cap(path: str | Path, batch_size: int, quantile: float) -> float:
    """Return the exact scale-norm quantile across all valid Gaussian records.

    The quantile is computed before opacity filtering.
    Only one float32 norm per record is retained between input batches.
    """
    norms = [
        np.linalg.norm(batch.scales, axis=1)
        for batch in tqdm(iter_geometry(path, batch_size), desc="Gaussian scales", unit="batch")
    ]
    values = np.concatenate(norms)
    del norms
    return float(np.quantile(values, quantile, overwrite_input=True))
