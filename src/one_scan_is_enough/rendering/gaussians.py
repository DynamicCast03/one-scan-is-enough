"""Load standard 3D Gaussian Splatting PLY files for rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


@dataclass(frozen=True)
class GaussianScene:
    """GPU tensors describing a static Gaussian scene."""

    means: torch.Tensor  # (N, 3), world positions in meters.
    quaternions: torch.Tensor  # (N, 4), WXYZ rotations.
    scales: torch.Tensor  # (N, 3), positive standard deviations.
    opacities: torch.Tensor  # (N,), activated alpha values.
    spherical_harmonics: torch.Tensor  # (N, K, 3), SH color coefficients.
    sh_degree: int


def _sorted_properties(
    vertex: object,
    prefix: str,
) -> list[str]:
    names = [
        property_definition.name
        for property_definition in vertex.properties
        if property_definition.name.startswith(prefix)
    ]
    return sorted(names, key=lambda name: int(name.rsplit("_", maxsplit=1)[1]))


def load_gaussian_scene(path: Path, device: torch.device) -> GaussianScene:
    """Load one Gaussian PLY and move all render parameters onto one GPU."""

    ply = PlyData.read(path)
    vertex = ply.elements[0]
    means = np.stack(
        [
            np.asarray(vertex["x"]),
            np.asarray(vertex["y"]),
            np.asarray(vertex["z"]),
        ],
        axis=1,
    ).astype(np.float32)
    dc = np.stack(
        [
            np.asarray(vertex["f_dc_0"]),
            np.asarray(vertex["f_dc_1"]),
            np.asarray(vertex["f_dc_2"]),
        ],
        axis=1,
    ).astype(np.float32)

    rest_names = _sorted_properties(vertex, "f_rest_")
    coefficient_count = len(rest_names) // 3 + 1
    sh_degree = int(round(coefficient_count**0.5)) - 1
    if (sh_degree + 1) ** 2 != coefficient_count:
        raise ValueError(f"{path} has an invalid number of spherical-harmonic coefficients")
    rest = (
        np.stack([np.asarray(vertex[name]) for name in rest_names], axis=1).astype(np.float32)
        if rest_names
        else np.empty((len(means), 0), dtype=np.float32)
    )
    rest = rest.reshape(len(means), 3, coefficient_count - 1).transpose(0, 2, 1)
    spherical_harmonics = np.concatenate([dc[:, None, :], rest], axis=1)

    scale_names = _sorted_properties(vertex, "scale_")
    rotation_names = _sorted_properties(vertex, "rot_")
    scales = np.stack(
        [np.asarray(vertex[name]) for name in scale_names],
        axis=1,
    ).astype(np.float32)
    quaternions = np.stack(
        [np.asarray(vertex[name]) for name in rotation_names],
        axis=1,
    ).astype(np.float32)
    opacity_logits = np.asarray(vertex["opacity"], dtype=np.float32)

    return GaussianScene(
        means=torch.from_numpy(means).to(device),
        quaternions=torch.from_numpy(quaternions).to(device),
        scales=torch.from_numpy(scales).to(device).exp(),
        opacities=torch.from_numpy(opacity_logits).to(device).sigmoid(),
        spherical_harmonics=torch.from_numpy(spherical_harmonics).to(device),
        sh_degree=sh_degree,
    )
